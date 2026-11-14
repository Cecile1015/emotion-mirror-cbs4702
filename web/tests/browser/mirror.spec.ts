import {test, expect, Page} from '@playwright/test';
async function fakeSpeech(page: Page, mode='normal') {
  await page.addInitScript(({mode}) => {
    class TestRecognition {
      onstart: (()=>void)|null=null; onend: (()=>void)|null=null; onresult: ((e:unknown)=>void)|null=null; onerror: ((e:unknown)=>void)|null=null;
      start() {this.onstart?.();if(mode==='denied') {this.onerror?.({error:'not-allowed'});return;} if(mode==='silent') {this.onerror?.({error:'no-speech'});return;} this.onresult?.({results:[{isFinal:true,0:{transcript:'I feel happy'}},{isFinal:false,0:{transcript:'today'}}]});}
      stop(){this.onend?.();} abort(){}
    }
    Object.defineProperty(window,'SpeechRecognition',{value:TestRecognition,configurable:true});
  },{mode});
}
test('real local API comparison and responsive results', async ({page}) => {
  const errors: string[]=[];page.on('pageerror',e=>errors.push(e.message));
  await page.goto('/');await page.getByRole('button',{name:'Open the emotion mirror'}).click();await expect(page.getByRole('button',{name:/models configured/})).toBeVisible();
  await page.getByLabel('Your words').fill('i feel lonely today');
  await page.getByRole('button',{name:'Compare emotions',exact:true}).click();
  await expect(page.getByRole('heading',{name:'One sentence. Different readings.'})).toBeVisible();
  await expect(page.locator('.result-card')).toHaveCount(2);await expect(page.locator('.emotion').first()).toHaveText('sadness');
  await page.getByLabel('Your words').fill('i feel excited today');await expect(page.getByText('Your input or model selection has changed.',{exact:false})).toBeVisible();
  await page.setViewportSize({width:390,height:844});
  expect(await page.evaluate(()=>document.documentElement.scrollWidth <= innerWidth)).toBeTruthy();
  expect(errors).toEqual([]);
});
test('speech is reviewable and never automatically submits', async ({page}) => {
  await fakeSpeech(page);let requests=0;page.on('request',r=>{if(r.url().endsWith('/api/predict')) requests++;});
  await page.goto('/');await page.getByRole('button',{name:'Open the emotion mirror'}).click();await page.getByLabel('Transcription',{exact:true}).selectOption('browser');await page.getByRole('button',{name:/Start speaking/}).click();
  await expect(page.getByLabel('Your words')).toHaveValue('I feel happy today');
  await expect(page.getByRole('button',{name:'Compare emotions',exact:true})).toBeDisabled();
  await page.getByRole('button',{name:/Stop & review/}).click();
  await expect(page.getByLabel('Your words')).toBeEditable();
  await page.getByLabel('Your words').fill('I feel lonely today');expect(requests).toBe(0);
  await page.getByRole('button',{name:'Compare emotions',exact:true}).click();
  await expect(page.locator('.emotion').first()).toHaveText('sadness');expect(requests).toBe(1);
});
for(const mode of ['denied','silent']) test(`speech ${mode} leaves text fallback usable`, async ({page}) => {
  await fakeSpeech(page,mode);await page.goto('/');await page.getByRole('button',{name:'Open the emotion mirror'}).click();await page.getByLabel('Transcription',{exact:true}).selectOption('browser');await page.getByRole('button',{name:/Start speaking/}).click();
  await expect(page.locator('.speech-status')).toContainText(mode==='denied'?'Microphone access was denied':'No speech was detected');
  await page.getByLabel('Your words').fill('I feel happy today');await expect(page.getByLabel('Your words')).toBeEditable();
});
test('unsupported speech and rejected camera keep classification available', async ({page}) => {
  await page.addInitScript(()=>{Object.defineProperty(window,'SpeechRecognition',{value:undefined});Object.defineProperty(window,'webkitSpeechRecognition',{value:undefined});navigator.mediaDevices.getUserMedia=async()=>{throw new DOMException('denied','NotAllowedError');};});
  await page.goto('/');await page.getByRole('button',{name:'Open the emotion mirror'}).click();await page.getByLabel('Transcription',{exact:true}).selectOption('browser');await expect(page.getByRole('button',{name:/Start speaking/})).toBeDisabled();
  await page.getByRole('button',{name:'Turn mirror camera on'}).click();await expect(page.getByText('Camera unavailable or permission denied.',{exact:false})).toBeVisible();
  await page.getByLabel('Your words').fill('I feel happy today');await expect(page.getByRole('button',{name:'Compare emotions',exact:true})).toBeEnabled();
});
test('pending camera permission can be cancelled and late tracks are stopped', async ({page}) => {
  await page.addInitScript(()=>{
    const w=window as unknown as {releaseCamera:()=>void; stoppedTrack:boolean};w.stoppedTrack=false;
    navigator.mediaDevices.getUserMedia=()=>new Promise(resolve=>{w.releaseCamera=()=>resolve({getTracks:()=>[{stop:()=>{w.stoppedTrack=true;}}]} as unknown as MediaStream);});
  });
  await page.goto('/');await page.getByRole('button',{name:'Open the emotion mirror'}).click();await page.getByRole('button',{name:'Turn mirror camera on'}).click();await page.getByRole('button',{name:'Cancel camera request'}).click();
  await page.evaluate(()=>(window as unknown as {releaseCamera:()=>void}).releaseCamera());
  await expect.poll(()=>page.evaluate(()=>(window as unknown as {stoppedTrack:boolean}).stoppedTrack)).toBe(true);
});
test('partial DeepSeek failure preserves real local result rendering', async ({page}) => {
  await page.route('**/api/predict',async route=>{const response=await page.request.post('/api/predict',{data:{text:'I feel happy today',models:['tfidf_lr']}});const d=await response.json();d.errors=[{model_id:'llm',code:'prediction_failed',message:'DeepSeek temporarily unavailable.'}];d.all_agree=null;await route.fulfill({json:d});});
  await page.goto('/');await page.getByRole('button',{name:'Open the emotion mirror'}).click();await expect(page.getByRole('button',{name:/models configured/})).toBeVisible();
  await page.getByLabel('Your words').fill('I feel happy today');await page.getByRole('button',{name:'Compare emotions',exact:true}).click();
  await expect(page.locator('.emotion')).toHaveText('joy');await expect(page.getByRole('heading',{name:'No prediction'})).toBeVisible();await expect(page.getByText('Comparison incomplete')).toBeVisible();
});
test('research population switch and error-to-mirror transfer', async ({page}) => {
  await page.goto('/');await page.getByRole('button',{name:'Open the emotion mirror'}).click();await page.getByRole('button',{name:'The Ledger',exact:true}).click();
  await expect(page.getByRole('table').first()).toContainText('0.7129');
  await page.getByRole('button',{name:'Full test set · 2,000 texts'}).click();await expect(page.getByRole('table').first()).toContainText('Not evaluated');
  await page.getByRole('button',{name:'Crypt of Errors',exact:true}).click();await expect(page.locator('.case-card').first()).toBeVisible();
  const sentence=(await page.locator('.case-card blockquote').first().innerText()).slice(1,-1);
  await page.getByRole('button',{name:'Try these words in the mirror'}).first().click();await expect(page.getByLabel('Your words')).toHaveValue(sentence);
});
test('local recording releases the microphone and waits for transcript review', async ({page}) => {
  await page.addInitScript(()=>{
    const w=window as unknown as {microphoneStopped:boolean};w.microphoneStopped=false;
    navigator.mediaDevices.getUserMedia=async()=>({getTracks:()=>[{stop:()=>{w.microphoneStopped=true;}}]} as unknown as MediaStream);
    class Recorder {
      state='inactive';mimeType='audio/webm';onstop:(()=>void)|null=null;ondataavailable:((e:{data:Blob})=>void)|null=null;onerror:(()=>void)|null=null;
      static isTypeSupported(){return true;}start(){this.state='recording';}stop(){this.state='inactive';setTimeout(()=>{this.ondataavailable?.({data:new Blob(['test audio'],{type:this.mimeType})});this.onstop?.();},0);}
    }
    Object.defineProperty(window,'MediaRecorder',{value:Recorder,configurable:true});
  });
  let predictions=0;page.on('request',r=>{if(r.url().endsWith('/api/predict'))predictions++;});
  await page.route('**/api/transcribe',route=>route.fulfill({json:{text:'I feel happy today',provider:'local'}}));
  await page.goto('/');await page.getByRole('button',{name:'Open the emotion mirror'}).click();await expect(page.getByLabel('Transcription',{exact:true})).toHaveValue('local');
  await page.getByRole('button',{name:/Start speaking/}).click();await expect(page.getByRole('button',{name:'Compare emotions',exact:true})).toBeDisabled();
  await page.getByRole('button',{name:/Stop & review/}).click();await expect(page.getByLabel('Your words')).toHaveValue('I feel happy today');
  expect(await page.evaluate(()=>(window as unknown as {microphoneStopped:boolean}).microphoneStopped)).toBe(true);expect(predictions).toBe(0);
  await page.getByLabel('Your words').fill('I feel lonely today');await page.getByRole('button',{name:'Compare emotions',exact:true}).click();
  await expect(page.locator('.emotion').first()).toHaveText('sadness');
});
