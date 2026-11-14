import {PHASE_DEVELOPMENT_SERVER} from 'next/constants.js';
export default phase => ({reactStrictMode:true,poweredByHeader:false,distDir:phase===PHASE_DEVELOPMENT_SERVER?'.next-dev':'.next'});
