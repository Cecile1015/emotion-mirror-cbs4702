import ledger from '@/data/ledger.json';
import errors from '@/data/error_analysis.json';
export function GET() { return Response.json({ledger, errors}); }
