'use client'

import { useEffect, useState, useCallback } from 'react'
import Lenis from 'lenis'
import Navbar from '@/components/Navbar'
import CustomCursor from '@/components/CustomCursor'
import AnimatedBg from '@/components/AnimatedBg'

export default function ClientProviders({ children }: { children: React.ReactNode }) {
  const [theme, setTheme] = useState<'dark' | 'light'>('dark')

  useEffect(() => {
    const lenis = new Lenis({
      duration: 1.2,
      easing: (t) => Math.min(1, 1.001 - Math.pow(2, -10 * t)),
      smoothWheel: true,
    })
    function raf(time: number) {
      lenis.raf(time)
      requestAnimationFrame(raf)
    }
    requestAnimationFrame(raf)
    return () => lenis.destroy()
  }, [])

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
  }, [theme])

  const toggleTheme = useCallback(() => {
    const next = theme === 'dark' ? 'light' : 'dark'

    // use view transitions api if available
    if (!document.startViewTransition) {
      setTheme(next)
      return
    }

    document.startViewTransition(() => {
      setTheme(next)
      document.documentElement.setAttribute('data-theme', next)
    })
  }, [theme])

  return (
    <>
      <AnimatedBg />
      <CustomCursor />
      <Navbar theme={theme} onToggleTheme={toggleTheme} />
      <main className="page-wrapper">
        {children}
      </main>

      <style jsx global>{`
        ::view-transition-old(root) {
          animation: none;
        }
        ::view-transition-new(root) {
          animation: clipReveal 0.7s cubic-bezier(0.87, 0, 0.13, 1) forwards;
        }
        @keyframes clipReveal {
          0%   { clip-path: circle(0% at calc(100% - 320px) 32px); }
          100% { clip-path: circle(150% at calc(100% - 320px) 32px); }
        }
        @media (max-width: 960px) {
          @keyframes clipReveal {
            0%   { clip-path: circle(0% at calc(100% - 56px) 32px); }
            100% { clip-path: circle(150% at calc(100% - 56px) 32px); }
          }
        }
      `}</style>
    </>
  )
}