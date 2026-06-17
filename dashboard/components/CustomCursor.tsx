'use client'

import { useEffect, useRef, useState } from 'react'
import { motion, useMotionValue, useSpring } from 'framer-motion'

export default function CustomCursor() {
  const cursorX = useMotionValue(-100)
  const cursorY = useMotionValue(-100)
  const [hovered, setHovered] = useState(false)
  const [visible, setVisible] = useState(false)

  // check once on mount — avoids setState inside effect
  const isTouch = useRef(true)

  const springConfig = { damping: 28, stiffness: 300, mass: 0.5 }
  const springX = useSpring(cursorX, springConfig)
  const springY = useSpring(cursorY, springConfig)

  useEffect(() => {
    isTouch.current = window.matchMedia('(hover: none)').matches
    if (isTouch.current) return

    const onMove = (e: MouseEvent) => {
      cursorX.set(e.clientX)
      cursorY.set(e.clientY)
      setVisible(true)
    }

    const onOver = (e: MouseEvent) => {
      const target = e.target as HTMLElement
      const isInteractive = !!(
        target.closest('a') ||
        target.closest('button') ||
        target.closest('[data-cursor-hover]')
      )
      setHovered(isInteractive)
    }

    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseover', onOver)
    return () => {
      window.removeEventListener('mousemove', onMove)
      window.removeEventListener('mouseover', onOver)
    }
  }, [])

  if (typeof window !== 'undefined' && window.matchMedia('(hover: none)').matches) return null

  return (
    <>
      <motion.div
        style={{
          position: 'fixed',
          top: 0,
          left: 0,
          x: springX,
          y: springY,
          translateX: '-50%',
          translateY: '-50%',
          zIndex: 9999,
          pointerEvents: 'none',
          width: hovered ? 40 : 24,
          height: hovered ? 40 : 24,
          border: '1px solid var(--accent)',
          borderRadius: '50%',
          opacity: visible ? (hovered ? 0.6 : 0.3) : 0,
          transition: 'width 0.2s ease, height 0.2s ease, opacity 0.2s ease',
          mixBlendMode: 'difference',
        }}
      />
      <motion.div
        style={{
          position: 'fixed',
          top: 0,
          left: 0,
          x: cursorX,
          y: cursorY,
          translateX: '-50%',
          translateY: '-50%',
          zIndex: 9999,
          pointerEvents: 'none',
          width: 4,
          height: 4,
          background: 'var(--accent)',
          borderRadius: '50%',
          opacity: visible ? 1 : 0,
          transition: 'opacity 0.2s ease',
        }}
      />
    </>
  )
}