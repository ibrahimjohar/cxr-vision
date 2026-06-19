import Link from 'next/link'
import { GithubLogo, LinkedinLogo, InstagramLogo, EnvelopeSimple } from '@phosphor-icons/react/dist/ssr'

export default function Footer() {
  return (
    <footer style={{
      position: 'relative',
      zIndex: 1,
      borderTop: '1px solid var(--border)',
      width: '100%',
    }}>
      <div style={{
        padding: 'clamp(2.5rem, 5vw, 4rem) clamp(1.5rem, 5vw, 4rem)',
        maxWidth: '1400px',
        margin: '0 auto',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        flexWrap: 'wrap',
        gap: '1.5rem',
      }}>
        {/* brand */}
        <Link href="/" style={{ textDecoration: 'none' }}>
          <span style={{
            fontFamily: 'Instrument Serif, serif',
            fontSize: '1.5rem',
            fontWeight: 700,
            color: 'var(--text-primary)',
            letterSpacing: '0.04em',
          }}>
            cxr<span style={{ color: 'var(--accent-light)' }}>.</span>vision
          </span>
        </Link>

        {/* center — built with */}
        <p style={{
          fontFamily: 'Hanken Grotesk, sans-serif',
          fontSize: '0.75rem',
          fontWeight: 500,
          letterSpacing: '0.1em',
          textTransform: 'uppercase',
          color: 'var(--text-muted)',
        }}>
          OpenCV · PyTorch · Next.js · Modal
        </p>

        {/* social icons */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '1.25rem' }}>
          {[
            { href: 'https://github.com/ibrahimjohar', icon: GithubLogo, label: 'GitHub' },
            { href: 'https://linkedin.com/in/ibrahimjohar', icon: LinkedinLogo, label: 'LinkedIn' },
            { href: 'https://instagram.com/ibrahimjohar', icon: InstagramLogo, label: 'Instagram' },
            { href: 'mailto:ibrahim@example.com', icon: EnvelopeSimple, label: 'Email' },
          ].map(({ href, icon: Icon, label }) => (
            <a
              key={label}
              href={href}
              target={href.startsWith('mailto') ? undefined : '_blank'}
              rel="noopener noreferrer"
              aria-label={label}
              style={{
                color: 'var(--text-muted)',
                transition: 'color 0.2s, transform 0.2s',
                display: 'flex',
              }}
              className="footer-icon"
            >
              <Icon size={18} weight="light" />
            </a>
          ))}
        </div>
      </div>

      {/* bottom bar */}
      <div style={{
        borderTop: '1px solid var(--border)',
        padding: '1rem clamp(1.5rem, 5vw, 4rem)',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        flexWrap: 'wrap',
        gap: '0.5rem',
      }}>
        <p style={{
          fontFamily: 'Hanken Grotesk, sans-serif',
          fontSize: '0.7rem',
          color: 'var(--text-muted)',
          letterSpacing: '0.08em',
          textTransform: 'uppercase',
        }}>
          Ibrahim Johar
        </p>
        <p style={{
          fontFamily: 'Hanken Grotesk, sans-serif',
          fontSize: '0.7rem',
          color: 'var(--text-muted)',
          letterSpacing: '0.08em',
          textTransform: 'uppercase',
        }}>
          2026
        </p>
      </div>

      <style jsx>{`
        .footer-icon:hover {
          color: var(--text-primary) !important;
          transform: translateY(-2px);
        }
      `}</style>
    </footer>
  )
}