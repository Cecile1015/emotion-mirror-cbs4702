import {speechBackend} from '@/lib/speechBackend';
export const runtime='nodejs';
export const dynamic='force-dynamic';
export function POST(request:Request){return speechBackend(request);}
