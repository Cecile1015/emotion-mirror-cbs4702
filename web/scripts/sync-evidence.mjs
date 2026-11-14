import {copyFile,mkdir} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
const web=new URL('../',import.meta.url);await mkdir(new URL('data/',web),{recursive:true});
for(const name of ['ledger','error_analysis','models']){await copyFile(new URL(`../results/app/${name}.json`,web),new URL(`data/${name}.json`,web));console.log(`Updated ${name}.json from saved research.`);}
