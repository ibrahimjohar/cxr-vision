'use client'

import { useEffect, useState } from 'react'
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

  const toggleTheme = () => setTheme(t => t === 'dark' ? 'light' : 'dark')

  return (
    <>
      <AnimatedBg />
      <CustomCursor />
      <Navbar theme={theme} onToggleTheme={toggleTheme} />
      <main className="page-wrapper">
        {children}
      </main>
    </>
  )
}