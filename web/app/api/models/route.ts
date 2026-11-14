import { proxyBackend } from '@/lib/backend';
export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';
export function GET() { return proxyBackend('models'); }
