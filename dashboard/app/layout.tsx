import type { Metadata } from 'next'
import './globals.css'
import ClientProviders from '@/components/ClientProviders'
import { Analytics } from '@vercel/analytics/next'

export const metadata: Metadata = {
  title: 'cxr-vision',
  description: 'Chest X-ray pathology detection — a full deep learning pipeline',
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link
          href="https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Hanken+Grotesk:wght@300;400;500;600;700;800&display=swap"
          rel="stylesheet"
        />
      </head>
      <body>
        <ClientProviders>
          {children}
        </ClientProviders>
        <Analytics />
      </body>
    </html>
  )
}