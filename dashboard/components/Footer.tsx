import Link from 'next/link'
import { ArrowUpRight } from '@phosphor-icons/react/dist/ssr'

const pages = [
  { href: '/overview', label: 'overview' },
  { href: '/preprocessing', label: 'preprocessing' },
  { href: '/classifier', label: 'classifier' },
  { href: '/vae', label: 'vae' },
  { href: '/unet', label: 'u-net' },
  { href: '/clip', label: 'clip' },
  { href: '/generative', label: 'generative' },
  { href: '/inference', label: 'inference' },
]

export default function Footer() {
  return (
    <footer style={{
      position: 'relative',
      zIndex: 1,
      borderTop: '1px solid var(--border)',
      padding: 'clamp(3rem, 6vw, 5rem) clamp(1.5rem, 5vw, 4rem)',
    }}>
      <div style={{ maxWidth: '1400px', margin: '0 auto' }}>
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
          gap: '3rem',
          marginBottom: '3rem',
        }}>
          {/* brand */}
          <div>
            <Link href="/" style={{ textDecoration: 'none' }}>
              <span style={{
                fontFamily: 'Instrument Serif, serif',
                fontSize: '1.6rem',
                fontWeight: 700,
                color: 'var(--text-primary)',
                letterSpacing: '0.04em',
                display: 'block',
                marginBottom: '1rem',
              }}>
                cxr<span style={{ color: 'var(--accent-light)' }}>.</span>vision
              </span>
            </Link>
            <p className="text-body" style={{ maxWidth: '260px', fontSize: '0.85rem' }}>
              A chest X-ray pathology detection pipeline built on the RSNA Pneumonia Detection Challenge dataset.
            </p>
          </div>

          {/* pages */}
          <div>
            <p className="text-subheading" style={{ marginBottom: '1.25rem' }}>pages</p>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem' }}>
              {pages.map(p => (
                <Link
                  key={p.href}
                  href={p.href}
                  style={{
                    fontFamily: 'Hanken Grotesk, sans-serif',
                    fontSize: '0.875rem',
                    color: 'var(--text-secondary)',
                    textDecoration: 'none',
                    transition: 'color 0.2s',
                    width: 'fit-content',
                  }}
                  className="footer-link"
                >
                  {p.label}
                </Link>
              ))}
            </div>
          </div>

          {/* project info */}
          <div>
            <p className="text-subheading" style={{ marginBottom: '1.25rem' }}>project</p>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem' }}>
              <a
                href="https://github.com/ibrahimjohar/cxr-vision"
                target="_blank"
                rel="noopener noreferrer"
                style={{
                  display: 'inline-flex', alignItems: 'center', gap: '0.3rem',
                  fontFamily: 'Hanken Grotesk, sans-serif',
                  fontSize: '0.875rem',
                  color: 'var(--text-secondary)',
                  textDecoration: 'none',
                  transition: 'color 0.2s',
                  width: 'fit-content',
                }}
                className="footer-link"
              >
                github repo <ArrowUpRight size={12} />
              </a>
              <p style={{ fontFamily: 'Hanken Grotesk, sans-serif', fontSize: '0.875rem', color: 'var(--text-muted)' }}>
                CS4045 · FAST NUCES Karachi
              </p>
              <p style={{ fontFamily: 'Hanken Grotesk, sans-serif', fontSize: '0.875rem', color: 'var(--text-muted)' }}>
                ResNet18 · VAE · DCGAN · DDPM · U-Net · CLIP
              </p>
            </div>
          </div>
        </div>

        {/* bottom bar */}
        <div style={{
          borderTop: '1px solid var(--border)',
          paddingTop: '1.5rem',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: '0.5rem',
        }}>
          <p className="text-subheading">
            built with PyTorch · Next.js · Modal
          </p>
          <p className="text-subheading">
            Ibrahim Johar · 2026
          </p>
        </div>
      </div>

      <style jsx>{`
        .footer-link:hover {
          color: var(--text-primary) !important;
        }
      `}</style>
    </footer>
  )
}