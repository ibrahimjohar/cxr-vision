'use client'

import { useEffect, useRef, useState } from 'react'
import { motion, useMotionValue, useSpring } from 'framer-motion'

export default function CustomCursor() {
  const cursorX = useMotionValue(-100)
  const cursorY = useMotionValue(-100)
  const [hovered, setHovered] = useState(false)
  const [visible, setVisible] = useState(false)
  const [enabled, setEnabled] = useState(false)
  const isTouch = useRef(true)

  const springConfig = { damping: 28, stiffness: 300, mass: 0.5 }
  const springX = useSpring(cursorX, springConfig)
  const springY = useSpring(cursorY, springConfig)

  useEffect(() => {
    isTouch.current = window.matchMedia('(hover: none)').matches
    if (isTouch.current) return
    setEnabled(true)

    const onMove = (e: MouseEvent) => {
      cursorX.set(e.clientX)
      cursorY.set(e.clientY)
      setVisible(true)
    }

    const onOver = (e: MouseEvent) => {
      const target = e.target as HTMLElement
      setHovered(!!(
        target.closest('a') ||
        target.closest('button') ||
        target.closest('[data-cursor-hover]')
      ))
    }

    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseover', onOver)
    return () => {
      window.removeEventListener('mousemove', onMove)
      window.removeEventListener('mouseover', onOver)
    }
  }, [])

  if (!enabled) return null

  return (
    <>
      {/* outer ring — mix-blend-mode exclusion gives inverted color effect */}
      <motion.div
        style={{
          position: 'fixed',
          top: 0, left: 0,
          x: springX, y: springY,
          translateX: '-50%', translateY: '-50%',
          zIndex: 9999,
          pointerEvents: 'none',
          width: hovered ? 48 : 32,
          height: hovered ? 48 : 32,
          background: '#FFEDDF',
          borderRadius: '50%',
          opacity: visible ? 1 : 0,
          mixBlendMode: 'exclusion',
          transition: 'width 0.25s cubic-bezier(0.87, 0, 0.13, 1), height 0.25s cubic-bezier(0.87, 0, 0.13, 1), opacity 0.2s ease',
        }}
      />
    </>
  )
}