import {copyFile,mkdir} from 'node:fs/promises';
import {execFileSync} from 'node:child_process';
import {existsSync} from 'node:fs';
import {fileURLToPath} from 'node:url';
const web=new URL('../',import.meta.url);await mkdir(new URL('data/',web),{recursive:true});
for(const name of ['ledger','models']){await copyFile(new URL(`../results/app/${name}.json`,web),new URL(`data/${name}.json`,web));console.log(`Updated ${name}.json from saved research.`);}

const root=fileURLToPath(new URL("../",web));
const localPython=fileURLToPath(new URL("../.venv/bin/python",web));
execFileSync(existsSync(localPython)?localPython:"python3",["scripts/prepare_error_gallery.py"],{cwd:root,stdio:"inherit"});
