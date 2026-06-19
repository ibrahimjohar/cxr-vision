'use client'

import Image from 'next/image'
import { motion, AnimatePresence } from 'framer-motion'
import { useState, useRef, useCallback, useEffect } from 'react'
import { ArrowLeft, ArrowRight } from '@phosphor-icons/react'
import Link from 'next/link'

const techniques = [
  {
    id: 'raw',
    label: 'Raw Input',
    tag: '00',
    src: '/results/images/preprocess_raw.png',
    color: 'rgba(255,237,223,0.5)',
    heading: 'Original DICOM',
    body: 'Frontal chest radiograph converted from DICOM format. Raw images vary significantly in brightness and contrast across scanners and acquisition settings, making direct model training difficult without normalization.',
    params: [
      { label: 'Format', value: 'DICOM → PNG' },
      { label: 'Size', value: '224 × 224 px' },
      { label: 'Channels', value: 'Grayscale' },
    ],
  },
  {
    id: 'clahe',
    label: 'CLAHE',
    tag: '01',
    src: '/results/images/preprocess_clahe.png',
    color: 'var(--accent-light)',
    heading: 'Contrast Limited Adaptive Histogram Equalization',
    body: 'CLAHE divides the image into small tiles and equalizes each tile\'s histogram independently, then interpolates between tiles. Unlike global histogram equalization, it prevents noise amplification via contrast clipping. Essential for making subtle opacities visible in chest X-rays where pathology hides in low-contrast regions.',
    params: [
      { label: 'Clip limit', value: '2.0' },
      { label: 'Tile grid', value: '8 × 8' },
      { label: 'Method', value: 'Local adaptive' },
    ],
  },
  {
    id: 'canny',
    label: 'Canny Edges',
    tag: '02',
    src: '/results/images/preprocess_canny.png',
    color: '#9B6DCC',
    heading: 'Canny Edge Detection',
    body: 'Multi-stage algorithm: Gaussian smoothing to reduce noise, gradient computation via Sobel operators, non-maximum suppression to thin edges, and double hysteresis thresholding to finalize edges. Produces clean single-pixel-wide edges that clearly delineate lung boundaries, rib contours, and the cardiac silhouette.',
    params: [
      { label: 'Low threshold', value: '50' },
      { label: 'High threshold', value: '150' },
      { label: 'Aperture', value: '3 (Sobel kernel)' },
    ],
  },
  {
    id: 'sobel',
    label: 'Sobel Gradient',
    tag: '03',
    src: '/results/images/preprocess_sobel.png',
    color: '#C084FC',
    heading: 'Sobel Gradient Magnitude',
    body: 'Computes intensity gradient using two 3×3 kernels — one horizontal (Gx), one vertical (Gy). The gradient magnitude √(Gx² + Gy²) encodes edge strength at every pixel, capturing structural boundaries in all orientations. Responds to both rib transitions and subtle opacity gradients that mark pneumonia infiltrates.',
    params: [
      { label: 'Kernel', value: '3 × 3 Sobel' },
      { label: 'Direction', value: 'X + Y combined' },
      { label: 'Output', value: 'Magnitude map' },
    ],
  },
]

const fadeUp = (delay = 0) => ({
  initial: { opacity: 0, y: 18 },
  whileInView: { opacity: 1, y: 0 },
  viewport: { once: true, amount: 0.1 } as const,
  transition: { duration: 0.55, delay },
})

// draggable before/after slider
function CompareSlider({ srcA, srcB, labelA, labelB }: { srcA: string; srcB: string; labelA: string; labelB: string }) {
  const containerRef = useRef<HTMLDivElement>(null)
  const [pct, setPct] = useState(50)

  const updatePct = useCallback((clientX: number) => {
    if (!containerRef.current) return
    const rect = containerRef.current.getBoundingClientRect()
    const x = Math.max(0, Math.min(clientX - rect.left, rect.width))
    setPct((x / rect.width) * 100)
  }, [])

  const onMouseMove = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    updatePct(e.clientX)
  }, [updatePct])

  const onTouchMove = useCallback((e: React.TouchEvent<HTMLDivElement>) => {
    updatePct(e.touches[0].clientX)
  }, [updatePct])

  return (
    <div
      ref={containerRef}
      onMouseMove={onMouseMove}
      onTouchMove={onTouchMove}
      onTouchStart={(e) => updatePct(e.touches[0].clientX)}
      style={{
        position: 'relative',
        width: '100%',
        aspectRatio: '1',
        borderRadius: '10px',
        overflow: 'hidden',
        border: '1px solid var(--border)',
        userSelect: 'none',
        cursor: 'default',
      }}
    >
      {/* image B (right, underneath) */}
      <div style={{ position: 'absolute', inset: 0 }}>
        <Image src={srcB} alt={labelB} fill sizes="600px" style={{ objectFit: 'cover', filter: 'grayscale(20%)' }} />
      </div>

      {/* image A (left, clipped) */}
      <div style={{ position: 'absolute', inset: 0, clipPath: `inset(0 ${100 - pct}% 0 0)` }}>
        <Image src={srcA} alt={labelA} fill sizes="600px" style={{ objectFit: 'cover' }} />
      </div>

      {/* divider line */}
      <div style={{
        position: 'absolute',
        top: 0, bottom: 0,
        left: `${pct}%`,
        width: '2px',
        background: 'var(--accent-light)',
        transform: 'translateX(-50%)',
        pointerEvents: 'none',
      }}>
        {/* handle */}
        <div style={{
          position: 'absolute',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          width: '36px',
          height: '36px',
          borderRadius: '50%',
          background: 'var(--accent-light)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          boxShadow: '0 4px 16px rgba(58,1,92,0.5)',
        }}>
          <div style={{ display: 'flex', gap: '2px', alignItems: 'center' }}>
            <ArrowLeft size={10} color="#fff" weight="bold" />
            <ArrowRight size={10} color="#fff" weight="bold" />
          </div>
        </div>
      </div>

      {/* labels */}
      <div style={{ position: 'absolute', top: '1rem', left: '1rem', pointerEvents: 'none' }}>
        <span style={{ fontFamily: 'Hanken Grotesk', fontSize: '0.6rem', fontWeight: 700, letterSpacing: '0.12em', textTransform: 'uppercase', color: '#fff', background: 'rgba(8,5,16,0.7)', padding: '0.25rem 0.5rem', borderRadius: '3px' }}>
          {labelA}
        </span>
      </div>
      <div style={{ position: 'absolute', top: '1rem', right: '1rem', pointerEvents: 'none' }}>
        <span style={{ fontFamily: 'Hanken Grotesk', fontSize: '0.6rem', fontWeight: 700, letterSpacing: '0.12em', textTransform: 'uppercase', color: '#fff', background: 'rgba(8,5,16,0.7)', padding: '0.25rem 0.5rem', borderRadius: '3px' }}>
          {labelB}
        </span>
      </div>
    </div>
  )
}

export default function PreprocessingPage() {
  const [active, setActive] = useState(1) // default: CLAHE
  const current = techniques[active]
  const prev = techniques[active > 0 ? active - 1 : active]

  return (
    <div>
      {/* header */}
      <section className="section" style={{ paddingBottom: '2rem' }}>
        <motion.div {...fadeUp(0)}>
          <p className="text-subheading" style={{ marginBottom: '0.75rem' }}>Stage 01 · Classical CV</p>
        </motion.div>
        <motion.div {...fadeUp(0.06)}>
          <h1 style={{ marginBottom: '1.5rem' }}>
            <span className="text-display" style={{ color: 'var(--text-primary)', display: 'block', fontSize: 'clamp(2.5rem, 6vw, 5rem)' }}>
              Image
            </span>
            <span style={{
              fontFamily: 'Instrument Serif, serif',
              fontStyle: 'italic',
              fontSize: 'clamp(2.5rem, 6vw, 5rem)',
              fontWeight: 400,
              background: 'var(--gradient-accent)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              backgroundClip: 'text',
              display: 'block',
              lineHeight: 1.05,
            }}>
              Preprocessing.
            </span>
          </h1>
        </motion.div>
        <motion.div {...fadeUp(0.12)}>
          <p className="text-body" style={{ maxWidth: '560px', fontSize: '1.05rem', lineHeight: 1.8 }}>
            Before any deep learning model sees a pixel, classical computer vision techniques
            normalize contrast, suppress noise, and extract structural priors.
            These operations run on every image in the 26,684-sample pipeline.
          </p>
        </motion.div>
      </section>

      <div className="divider" />

      {/* main interactive section */}
      <section className="section">

        {/* technique tabs */}
        <motion.div {...fadeUp(0)} style={{ display: 'flex', gap: '0', marginBottom: '3rem', border: '1px solid var(--border)', borderRadius: '6px', overflow: 'hidden', width: 'fit-content' }}>
          {techniques.map((t, i) => (
            <button
              key={t.id}
              onClick={() => setActive(i)}
              style={{
                padding: '0.6rem 1.4rem',
                background: active === i ? 'linear-gradient(135deg, #3A015C, #7B2FBE)' : 'transparent',
                border: 'none',
                borderRight: i < techniques.length - 1 ? '1px solid var(--border)' : 'none',
                color: active === i ? '#fff' : 'var(--text-muted)',
                fontFamily: 'Hanken Grotesk, sans-serif',
                fontSize: '0.78rem',
                fontWeight: 600,
                letterSpacing: '0.06em',
                transition: 'background 0.2s, color 0.2s',
              }}
            >
              {t.label}
            </button>
          ))}
        </motion.div>

        {/* main two-column layout */}
        <div className="preprocess-main-grid" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '4rem', alignItems: 'start' }}>

          {/* left: image viewer */}
          <div>
            <AnimatePresence mode="wait">
              <motion.div
                key={current.id}
                initial={{ opacity: 0, scale: 0.97 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.97 }}
                transition={{ duration: 0.35, ease: [0.25, 0.1, 0.25, 1] }}
              >
                <div style={{
                  position: 'relative',
                  width: '100%',
                  aspectRatio: '1',
                  borderRadius: '10px',
                  overflow: 'hidden',
                  border: `1px solid ${current.color}`,
                  boxShadow: `0 0 40px rgba(123,47,190,0.15)`,
                }}>
                  <Image
                    src={current.src}
                    alt={current.label}
                    fill
                    sizes="600px"
                    style={{ objectFit: 'cover' }}
                  />
                  {/* scan overlay */}
                  <div style={{ position: 'absolute', inset: 0, background: 'linear-gradient(to top, rgba(8,5,16,0.4) 0%, transparent 40%)' }} />
                  <div style={{ position: 'absolute', bottom: '1rem', left: '1rem', right: '1rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span style={{ fontFamily: 'monospace', fontSize: '0.58rem', color: 'rgba(200,170,255,0.8)', letterSpacing: '0.08em' }}>
                      {current.tag} · {current.label.toUpperCase()}
                    </span>
                    <span style={{ fontFamily: 'monospace', fontSize: '0.58rem', color: 'rgba(200,170,255,0.8)', letterSpacing: '0.08em' }}>
                      224px
                    </span>
                  </div>
                </div>
              </motion.div>
            </AnimatePresence>

            {/* step arrows */}
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginTop: '1rem' }}>
              <button
                onClick={() => setActive(a => Math.max(0, a - 1))}
                disabled={active === 0}
                style={{
                  display: 'flex', alignItems: 'center', gap: '0.4rem',
                  background: 'transparent', border: '1px solid var(--border)',
                  borderRadius: '4px', padding: '0.5rem 0.9rem',
                  color: active === 0 ? 'var(--text-muted)' : 'var(--text-secondary)',
                  fontFamily: 'Hanken Grotesk', fontSize: '0.75rem', fontWeight: 600,
                  letterSpacing: '0.06em',
                  opacity: active === 0 ? 0.4 : 1,
                  transition: 'opacity 0.2s',
                }}
              >
                <ArrowLeft size={12} weight="bold" /> prev
              </button>

              {/* step dots */}
              <div style={{ display: 'flex', gap: '6px' }}>
                {techniques.map((_, i) => (
                  <button
                    key={i}
                    onClick={() => setActive(i)}
                    style={{
                      width: active === i ? '20px' : '6px',
                      height: '6px',
                      borderRadius: '3px',
                      background: active === i ? 'var(--accent-light)' : 'var(--border)',
                      border: 'none',
                      padding: 0,
                      transition: 'width 0.3s ease, background 0.3s ease',
                    }}
                  />
                ))}
              </div>

              <button
                onClick={() => setActive(a => Math.min(techniques.length - 1, a + 1))}
                disabled={active === techniques.length - 1}
                style={{
                  display: 'flex', alignItems: 'center', gap: '0.4rem',
                  background: 'transparent', border: '1px solid var(--border)',
                  borderRadius: '4px', padding: '0.5rem 0.9rem',
                  color: active === techniques.length - 1 ? 'var(--text-muted)' : 'var(--text-secondary)',
                  fontFamily: 'Hanken Grotesk', fontSize: '0.75rem', fontWeight: 600,
                  letterSpacing: '0.06em',
                  opacity: active === techniques.length - 1 ? 0.4 : 1,
                  transition: 'opacity 0.2s',
                }}
              >
                next <ArrowRight size={12} weight="bold" />
              </button>
            </div>
          </div>

          {/* right: technique info */}
          <div>
            <AnimatePresence mode="wait">
              <motion.div
                key={current.id + '_info'}
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -8 }}
                transition={{ duration: 0.3, ease: [0.25, 0.1, 0.25, 1] }}
              >
                {/* tag + step */}
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1.25rem' }}>
                  <div style={{ width: '2px', height: '2rem', background: current.color, borderRadius: '1px' }} />
                  <p className="text-subheading" style={{ color: current.color }}>{current.tag} · {current.label}</p>
                </div>

                <h2 style={{
                  fontFamily: 'Instrument Serif, serif',
                  fontSize: 'clamp(1.4rem, 2.5vw, 2rem)',
                  fontWeight: 400,
                  color: 'var(--text-primary)',
                  lineHeight: 1.2,
                  marginBottom: '1.25rem',
                }}>
                  {current.heading}
                </h2>

                <p className="text-body" style={{ fontSize: '1rem', lineHeight: 1.85, marginBottom: '2rem' }}>
                  {current.body}
                </p>

                {/* params */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0', border: '1px solid var(--border)', borderRadius: '6px', overflow: 'hidden', marginBottom: '2rem' }}>
                  {current.params.map((p, i) => (
                    <div key={p.label} style={{
                      display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                      padding: '0.75rem 1rem',
                      borderBottom: i < current.params.length - 1 ? '1px solid var(--border)' : 'none',
                      background: i % 2 === 0 ? 'transparent' : 'var(--surface)',
                    }}>
                      <p className="text-subheading">{p.label}</p>
                      <p style={{ fontFamily: 'Hanken Grotesk', fontSize: '0.875rem', fontWeight: 700, color: 'var(--text-primary)', letterSpacing: '0.02em' }}>
                        {p.value}
                      </p>
                    </div>
                  ))}
                </div>
              </motion.div>
            </AnimatePresence>
          </div>
        </div>
      </section>

      <div className="divider" />

      {/* pipeline flow + compare side by side */}
      <section className="section">
        <div className="pipeline-compare-grid" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '3rem', alignItems: 'start' }}>

          {/* left: heading + stacked pipeline steps */}
          <div>
            <motion.div {...fadeUp(0)}>
              <p className="text-subheading" style={{ marginBottom: '0.5rem' }}>Processing Pipeline</p>
            </motion.div>
            <motion.div {...fadeUp(0.06)}>
              <h2 style={{ marginBottom: '1.5rem' }}>
                <span className="text-heading" style={{ color: 'var(--text-primary)' }}>Ordered </span>
                <span className="text-heading font-display" style={{ fontStyle: 'italic', color: 'var(--text-primary)' }}>operations.</span>
              </h2>
            </motion.div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0', overflow: 'hidden', border: '1px solid var(--border)', borderRadius: '8px' }}>
            {techniques.map((t, i) => (
              <motion.div key={t.id} {...fadeUp(i * 0.07)}>
                <button
                  onClick={() => setActive(i)}
                  className="pipeline-step-btn"
                  style={{
                    width: '100%',
                    display: 'flex',
                    flexDirection: 'column',
                    padding: '1.25rem 1.5rem',
                    background: active === i ? 'rgba(123,47,190,0.12)' : 'transparent',
                    border: 'none',
                    borderBottom: i < techniques.length - 1 ? '1px solid var(--border)' : 'none',
                    transition: 'background 0.2s',
                    textAlign: 'left',
                    gap: '0.3rem',
                    cursor: 'pointer',
                    overflow: 'hidden',
                    position: 'relative',
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                      <p style={{ fontFamily: 'Instrument Serif', fontSize: '1.5rem', fontWeight: 300, color: active === i ? 'var(--accent-light)' : 'var(--text-muted)', lineHeight: 1, opacity: active === i ? 0.9 : 0.3, transition: 'color 0.2s, opacity 0.2s', minWidth: '2.5rem' }}>
                        {t.tag}
                      </p>
                      <div>
                        <p style={{ fontFamily: 'Hanken Grotesk', fontSize: '0.9rem', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '0.15rem' }}>
                          {t.label}
                        </p>
                        <p className="text-subheading" style={{ fontSize: '0.62rem' }}>
                          {i === 0 ? 'input' : i === 1 ? 'contrast enhancement' : i === 2 ? 'edge detection' : 'gradient magnitude'}
                        </p>
                      </div>
                    </div>
                    {active === i && (
                      <div style={{ width: '6px', height: '6px', borderRadius: '50%', background: 'var(--accent-light)', flexShrink: 0 }} />
                    )}
                  </div>
                  {active === i && (
                    <motion.div
                      layoutId="pipeline-indicator"
                      style={{ position: 'absolute', bottom: 0, left: 0, right: 0, height: '2px', background: 'var(--gradient-accent)' }}
                    />
                  )}
                </button>
              </motion.div>
            ))}
            </div>
          </div>

          {/* right: compare slider */}
          <div>
            <p className="text-subheading" style={{ marginBottom: '0.5rem' }}>Compare</p>
            <AnimatePresence mode="wait">
              <motion.div
                key={current.id + '_label'}
                initial={{ opacity: 0, y: 6 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -4 }}
                transition={{ duration: 0.25 }}
              >
                <h2 style={{ marginBottom: '0.5rem' }}>
                  <span className="text-heading" style={{ color: 'var(--text-primary)' }}>Raw vs </span>
                  <span className="text-heading font-display" style={{ fontStyle: 'italic', color: 'var(--text-primary)' }}>{current.label}.</span>
                </h2>
                <p className="text-body" style={{ fontSize: '0.875rem', marginBottom: '1.5rem' }}>
                  Hover over the image to reveal the difference.
                </p>
              </motion.div>
            </AnimatePresence>

            <AnimatePresence mode="wait">
              <motion.div
                key={current.id + '_compare'}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.25 }}
              >
                {current.id === 'raw' ? (
                  <div style={{ padding: '2rem', border: '1px solid var(--border)', borderRadius: '10px', textAlign: 'center' }}>
                    <p className="text-body" style={{ fontSize: '0.9rem' }}>
                      Select a technique on the left to compare.
                    </p>
                  </div>
                ) : (
                  <CompareSlider
                    srcA={techniques[0].src}
                    srcB={current.src}
                    labelA="Raw"
                    labelB={current.label}
                  />
                )}
              </motion.div>
            </AnimatePresence>
          </div>
        </div>
      </section>



      <div className="divider" />

      {/* all four images grid */}
      <section className="section">
        <motion.div {...fadeUp(0)}>
          <p className="text-subheading" style={{ marginBottom: '0.75rem' }}>All Outputs</p>
        </motion.div>
        <motion.div {...fadeUp(0.08)}>
          <h2 style={{ marginBottom: '3rem' }}>
            <span className="text-heading" style={{ color: 'var(--text-primary)' }}>Four views, </span>
            <span className="text-heading font-display" style={{ fontStyle: 'italic', color: 'var(--text-primary)' }}>one patient.</span>
          </h2>
        </motion.div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1rem' }}>
          {techniques.map((t, i) => (
            <motion.div key={t.id} {...fadeUp(i * 0.06)}>
              <button
                onClick={() => setActive(i)}
                style={{
                  width: '100%', background: 'none', border: 'none', padding: 0,
                  display: 'flex', flexDirection: 'column', gap: '0.75rem',
                  cursor: 'pointer',
                }}
                className="thumb-btn"
              >
                <div style={{
                  width: '100%', aspectRatio: '1',
                  borderRadius: '8px', overflow: 'hidden',
                  border: `1px solid ${active === i ? t.color : 'var(--border)'}`,
                  transition: 'border-color 0.2s, transform 0.2s, box-shadow 0.2s',
                  transform: active === i ? 'scale(1.03)' : 'scale(1)',
                  boxShadow: active === i ? `0 8px 24px rgba(123,47,190,0.2)` : 'none',
                }}>
                  <div style={{ position: 'relative', width: '100%', height: '100%' }}>
                    <Image src={t.src} alt={t.label} fill sizes="300px" style={{ objectFit: 'cover' }} />
                  </div>
                </div>
                <div style={{ textAlign: 'left' }}>
                  <p style={{ fontFamily: 'monospace', fontSize: '0.6rem', color: t.color, letterSpacing: '0.1em', marginBottom: '0.2rem' }}>{t.tag}</p>
                  <p style={{ fontFamily: 'Hanken Grotesk', fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-primary)' }}>{t.label}</p>
                </div>
              </button>
            </motion.div>
          ))}
        </div>
      </section>

      <div className="divider" />

      {/* nav to next page */}
      <section className="section" style={{ paddingTop: 'clamp(2rem, 5vw, 4rem)', paddingBottom: 'clamp(2rem, 5vw, 4rem)' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Link href="/overview" style={{ textDecoration: 'none' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--text-muted)', fontFamily: 'Hanken Grotesk', fontSize: '0.8rem', fontWeight: 600, letterSpacing: '0.06em' }}>
              <ArrowLeft size={13} weight="bold" /> overview
            </div>
          </Link>
          <Link href="/classifier" style={{ textDecoration: 'none' }}>
            <button style={{
              display: 'flex', alignItems: 'center', gap: '0.5rem',
              padding: '0.75rem 1.6rem',
              background: 'linear-gradient(135deg, #3A015C, #7B2FBE)',
              border: 'none', borderRadius: '3px',
              color: '#fff',
              fontFamily: 'Hanken Grotesk', fontSize: '0.8rem', fontWeight: 600,
              letterSpacing: '0.05em',
              boxShadow: '0 4px 24px rgba(123,47,190,0.3)',
            }}>
              next: classifier <ArrowRight size={13} weight="bold" />
            </button>
          </Link>
        </div>
      </section>

      <style jsx>{`
        .pipeline-compare-grid {
          align-items: start;
        }
        @media (max-width: 900px) {
          .pipeline-compare-grid {
            grid-template-columns: 1fr !important;
          }
        }
        .pipeline-step-btn:hover {
          background: rgba(123, 47, 190, 0.08) !important;
        }
        .thumb-btn:hover > div:first-child {
          transform: scale(1.04) !important;
        }
        @media (max-width: 900px) {
          .preprocess-main-grid {
            grid-template-columns: 1fr !important;
          }
        }
        @media (max-width: 640px) {
          div[style*="repeat(4, 1fr)"] {
            grid-template-columns: repeat(2, 1fr) !important;
          }
        }
      `}</style>
    </div>
  )
}