import type { Metadata } from 'next';
import './globals.css';
export const metadata: Metadata = {title: 'Boo, Darlings: emotion mirror', description: 'A voice-first comparison of six English emotion classifiers. A CBS4702 research project, inspired by Boo, Darling!'};
export default function Layout({children}: {children: React.ReactNode}) {
  return <html lang="en"><body>{children}</body></html>;
}
