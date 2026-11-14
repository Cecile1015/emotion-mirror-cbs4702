import { proxyBackend } from '@/lib/backend';
export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';
export function POST(request: Request) { return proxyBackend('predict', request); }
