'use client'

import { useState, useEffect } from 'react'
import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { Sun, Moon, List, X } from '@phosphor-icons/react'

const links = [
  { href: '/', label: 'home' },
  { href: '/about', label: 'about' },
  { href: '/overview', label: 'overview' },
  { href: '/preprocessing', label: 'preprocessing' },
  { href: '/classifier', label: 'classifier' },
  { href: '/vae', label: 'vae' },
  { href: '/unet', label: 'u-net' },
  { href: '/clip', label: 'clip' },
  { href: '/generative', label: 'generative' },
  { href: '/inference', label: 'inference' },
  { href: '/about', label: 'about'}
]

interface NavbarProps {
  theme: 'dark' | 'light'
  onToggleTheme: () => void
}

export default function Navbar({ theme, onToggleTheme }: NavbarProps) {
  const pathname = usePathname()
  const [scrolled, setScrolled] = useState(false)
  const [menuOpen, setMenuOpen] = useState(false)
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    setTimeout(() => setMounted(true), 0)
    const onScroll = () => setScrolled(window.scrollY > 20)
    window.addEventListener('scroll', onScroll, { passive: true })
    return () => window.removeEventListener('scroll', onScroll)
  }, [])

  return (
    <>
      <nav style={{
        position: 'fixed',
        top: 0, left: 0, right: 0,
        height: 'var(--nav-height)',
        zIndex: 50,
        transition: 'background 0.3s ease, border-color 0.3s ease',
        background: scrolled ? 'var(--surface)' : 'transparent',
        backdropFilter: scrolled ? 'blur(16px)' : 'none',
        WebkitBackdropFilter: scrolled ? 'blur(16px)' : 'none',
        borderBottom: scrolled ? '1px solid var(--border)' : '1px solid transparent',
      }}>
        <div style={{
          maxWidth: '1400px',
          margin: '0 auto',
          padding: '0 clamp(1.5rem, 5vw, 4rem)',
          height: '100%',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
        }}>
          <Link href="/" style={{ textDecoration: 'none' }} onClick={() => setMenuOpen(false)}>
            <span style={{
              fontFamily: 'Instrument Serif, serif',
              fontSize: '1.7rem',
              fontWeight: 700,
              color: 'var(--text-primary)',
              letterSpacing: '0.04em',
              transition: 'color 0.4s ease',
            }}>
              cxr<span style={{ color: 'var(--accent-light)' }}>.</span>vision
            </span>
          </Link>

          <div className="nav-desktop" style={{
            display: 'flex',
            alignItems: 'center',
            gap: '2.5rem',
          }}>
            {links.slice(2).map(link => (
              <Link
                key={link.href}
                href={link.href}
                style={{
                  fontFamily: 'Hanken Grotesk, sans-serif',
                  fontSize: '0.875rem',
                  fontWeight: pathname === link.href ? 600 : 400,
                  letterSpacing: '0.01em',
                  textDecoration: 'none',
                  color: pathname === link.href ? 'var(--text-primary)' : 'var(--text-secondary)',
                  transition: 'color 0.2s ease',
                  position: 'relative',
                  whiteSpace: 'nowrap',
                }}
                className="nav-link"
              >
                {link.label}
                {pathname === link.href && (
                  <span style={{
                    position: 'absolute',
                    bottom: '-4px',
                    left: 0, right: 0,
                    height: '1px',
                    background: 'var(--accent-light)',
                    display: 'block',
                  }} />
                )}
              </Link>
            ))}
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '1.5rem' }}>
            <button
              onClick={onToggleTheme}
              style={{
                background: 'none', border: 'none',
                padding: '6px', cursor: 'pointer',
                color: 'var(--text-secondary)',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                transition: 'color 0.2s, transform 0.4s ease',
                borderRadius: '50%',
              }}
              aria-label="Toggle theme"
            >
              {mounted && (theme === 'dark'
                ? <Sun size={20} weight="light" />
                : <Moon size={20} weight="light" />
              )}
            </button>

            <button
              onClick={() => setMenuOpen(o => !o)}
              style={{
                background: 'none', border: 'none',
                padding: '6px', cursor: 'pointer',
                color: 'var(--text-secondary)',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                transition: 'color 0.2s',
              }}
              className="nav-menu-btn"
              aria-label="Toggle menu"
            >
              {menuOpen ? <X size={20} weight="light" /> : <List size={20} weight="light" />}
            </button>
          </div>
        </div>
      </nav>

      {menuOpen && (
        <div
          style={{
            position: 'fixed',
            top: 'var(--nav-height)',
            left: 0, right: 0,
            zIndex: 49,
            background: 'var(--bg-secondary)',
            borderBottom: '1px solid var(--border)',
            padding: '1.5rem clamp(1.5rem, 5vw, 4rem)',
            animation: 'menuSlideDown 0.3s cubic-bezier(0.87, 0, 0.13, 1) forwards',
          }}
        >
          {links.map(link => (
            <Link
              key={link.href}
              href={link.href}
              onClick={() => setMenuOpen(false)}
              style={{
                display: 'block',
                padding: '0.75rem 0',
                fontFamily: 'Hanken Grotesk, sans-serif',
                fontSize: '0.875rem',
                fontWeight: 400,
                letterSpacing: '0.01em',
                textDecoration: 'none',
                color: pathname === link.href ? 'var(--text-primary)' : 'var(--text-secondary)',
                borderBottom: '1px solid var(--border)',
                transition: 'color 0.2s ease',
              }}
            >
              {link.label}
            </Link>
          ))}
        </div>
      )}

      <style jsx>{`
        .nav-desktop { display: flex !important; }
        .nav-menu-btn { display: none !important; }
        .nav-link:hover { color: var(--text-primary) !important; }
        @media (max-width: 960px) {
          .nav-desktop { display: none !important; }
          .nav-menu-btn { display: flex !important; }
        }
      `}</style>
    </>
  )
}