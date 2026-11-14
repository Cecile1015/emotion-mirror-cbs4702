import {speechBackend} from '@/lib/speechBackend';
export const runtime='nodejs';
export const dynamic='force-dynamic';
export function GET(){return speechBackend();}
