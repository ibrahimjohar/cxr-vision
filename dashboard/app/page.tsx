'use client'

import Link from 'next/link'
import Image from 'next/image'
import { motion, animate, useInView } from 'framer-motion'
import { ArrowRight, ArrowUpRight, Stethoscope, Sparkle, ArrowDown } from '@phosphor-icons/react'
import { useEffect, useRef, useState } from 'react'

const pipeline = [
  { step: '01', label: 'Classical CV', desc: 'CLAHE · Canny · Sobel', href: '/preprocessing' },
  { step: '02', label: 'Classification', desc: 'ResNet18 · 85.08% acc', href: '/classifier' },
  { step: '03', label: 'Anomaly Detection', desc: 'VAE · AUC 0.51', href: '/vae' },
  { step: '04', label: 'Segmentation', desc: 'Attention U-Net · Dice 0.40', href: '/unet' },
  { step: '05', label: 'Generative', desc: 'DCGAN · DDPM · FID 64.1', href: '/generative' },
  { step: '06', label: 'Zero-Shot CLIP', desc: 'CLIP · BiomedCLIP · AUC 0.84', href: '/clip' },
]

const stats = [
  { value: 26684, label: 'X-rays', sub: 'RSNA dataset', decimals: 0 },
  { value: 85, label: 'Accuracy', sub: 'ResNet18 classifier', suffix: '%', decimals: 0 },
  { value: 0.84, label: 'AUC', sub: 'BiomedCLIP zero-shot', decimals: 2 },
  { value: 64.1, label: 'FID', sub: 'Diffusion model', decimals: 1 },
]

const xrays = [
  { src: '/xrays/xray1.jpg', label: 'Normal · PA View' },
  { src: '/xrays/xray2.png', label: 'Opacity · PA View' },
  { src: '/xrays/xray3.jpg', label: 'Consolidation · PA' },
]

function CountUp({ value, decimals, suffix = '' }: { value: number; decimals: number; suffix?: string }) {
  const ref = useRef<HTMLSpanElement>(null)
  const inView = useInView(ref as React.RefObject<Element>, { once: true, margin: '-50px' })
  const [displayed, setDisplayed] = useState('0')

  useEffect(() => {
    if (!inView) return
    const controls = animate(0, value, {
      duration: 1.6,
      ease: [0.25, 0.1, 0.25, 1] as [number, number, number, number],
      onUpdate(v) {
        setDisplayed(decimals === 0 ? Math.floor(v).toLocaleString() : v.toFixed(decimals))
      },
    })
    return controls.stop
  }, [inView, value, decimals])

  return <span ref={ref}>{displayed}{suffix}</span>
}

function XrayCardStack() {
  const [active, setActive] = useState(0)
  const [hovered, setHovered] = useState(false)
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null)

  useEffect(() => {
    if (hovered) {
      if (intervalRef.current) clearInterval(intervalRef.current)
      return
    }
    intervalRef.current = setInterval(() => {
      setActive(prev => (prev + 1) % xrays.length)
    }, 3000)
    return () => { if (intervalRef.current) clearInterval(intervalRef.current) }
  }, [hovered])

  return (
    <motion.div
      initial={{ opacity: 0, y: 24 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, amount: 0.1 }}
      transition={{ duration: 0.6, delay: 0.25 }}
    >
      <div
        className="hero-xray-stack"
        style={{ position: 'relative', width: '440px', height: '530px', flexShrink: 0, cursor: 'pointer' }}
        onMouseEnter={() => setHovered(true)}
        onMouseLeave={() => setHovered(false)}
        onClick={() => setActive(prev => (prev + 1) % xrays.length)}
      >
        {/* all 3 cards always mounted — zIndex and transforms drive stacking */}
        {xrays.map((xray, i) => {
          const isActive = i === active
          // stackPos: 0 = front (active), 1 = mid, 2 = back
          const stackPos = ((i - active) + xrays.length) % xrays.length
          const rotations = [0, 4, 8]
          const offsets = [0, 16, 32]
          const rot = rotations[stackPos] ?? 8
          const off = offsets[stackPos] ?? 32

          return (
            <motion.div
              key={xray.src}
              animate={{
                rotate: rot,
                x: off,
                y: isActive
                  ? hovered ? -14 : 0
                  : off + (hovered ? -3 : 0),
                zIndex: 10 - stackPos,
                scale: 1 - stackPos * 0.018,
              }}
              transition={{ duration: 0.55, ease: [0.25, 0.1, 0.25, 1] }}
              style={{
                position: 'absolute',
                top: 0, left: 0,
                width: '360px', height: '440px',
                borderRadius: '10px',
                overflow: 'hidden',
                border: isActive
                  ? '1px solid rgba(123, 47, 190, 0.65)'
                  : `1px solid rgba(123, 47, 190, ${0.22 + (2 - stackPos) * 0.07})`,
                boxShadow: isActive
                  ? hovered
                    ? '0 28px 72px rgba(58,1,92,0.55), 0 0 0 1px rgba(123,47,190,0.4)'
                    : '0 10px 40px rgba(58,1,92,0.35)'
                  : 'none',
                transformOrigin: 'center center',
              }}
            >
              <Image
                src={xray.src}
                alt="chest x-ray"
                fill
                sizes="360px"
                priority={i === 0}
                style={{
                  objectFit: 'cover',
                  filter: 'grayscale(100%)',
                  opacity: isActive ? 0.76 : 0.42 - stackPos * 0.06,
                }}
              />
              <div style={{
                position: 'absolute', inset: 0,
                background: isActive
                  ? 'linear-gradient(160deg, rgba(58,1,92,0.14) 0%, rgba(14,8,24,0.26) 100%)'
                  : `linear-gradient(160deg, rgba(58,1,92,${0.38 + stackPos * 0.07}), rgba(14,8,24,${0.5 + stackPos * 0.07}))`,
              }} />

              {/* metadata label — only on active */}
              {isActive && (
                <div style={{ position: 'absolute', inset: 0, padding: '1.1rem', display: 'flex', flexDirection: 'column', justifyContent: 'space-between', pointerEvents: 'none' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span style={{ fontFamily: 'monospace', fontSize: '0.58rem', color: 'rgba(200,170,255,0.8)', letterSpacing: '0.08em' }}>PA VIEW</span>
                    <span style={{ fontFamily: 'monospace', fontSize: '0.58rem', color: 'rgba(200,170,255,0.8)', letterSpacing: '0.08em' }}>RSNA</span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span style={{ fontFamily: 'Hanken Grotesk', fontSize: '0.6rem', color: 'rgba(200,170,255,0.9)', letterSpacing: '0.1em', textTransform: 'uppercase' }}>
                      {xray.label}
                    </span>
                    <Sparkle size={11} style={{ color: 'rgba(200,170,255,0.7)' }} />
                  </div>
                </div>
              )}
            </motion.div>
          )
        })}

        {/* classifier badge */}
        <div style={{
          position: 'absolute',
          bottom: '20px', right: '-8px',
          zIndex: 20,
          padding: '0.55rem 0.9rem',
          background: 'var(--bg-secondary)',
          backdropFilter: 'blur(16px)',
          WebkitBackdropFilter: 'blur(16px)',
          border: '1px solid rgba(123, 47, 190, 0.4)',
          borderRadius: '6px',
        }}>
          <p style={{ fontFamily: 'Hanken Grotesk', fontSize: '0.58rem', color: 'var(--text-muted)', letterSpacing: '0.1em', textTransform: 'uppercase', marginBottom: '0.15rem' }}>
            classifier acc.
          </p>
          <p style={{ fontFamily: 'Hanken Grotesk', fontSize: '1.1rem', fontWeight: 800, color: 'var(--text-primary)', letterSpacing: '-0.02em' }}>
            85.08%
          </p>
        </div>
      </div>
    </motion.div>
  )
}

const fadeUp = (delay = 0) => ({
  initial: { opacity: 0, y: 18 },
  whileInView: { opacity: 1, y: 0 },
  viewport: { once: true, amount: 0.1 } as const,
  transition: { duration: 0.55, delay },
})

export default function HomePage() {
  return (
    <div style={{ position: 'relative' }}>

      {/* hero */}
      <section style={{
        minHeight: '100vh',
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'center',
        padding: 'calc(var(--nav-height) - 1.5rem) clamp(1.5rem, 5vw, 4rem) clamp(2rem, 4vw, 3rem)',
        position: 'relative',
        overflow: 'hidden',
      }}>

        {/* dot grid */}
        <div style={{
          position: 'absolute',
          inset: 0,
          backgroundImage: 'radial-gradient(circle, var(--text-muted) 1px, transparent 1px)',
          backgroundSize: '28px 28px',
          WebkitMaskImage: 'linear-gradient(to bottom, transparent 0%, rgba(0,0,0,0.3) 10px, black 50px)',
          maskImage: 'linear-gradient(to bottom, transparent 0%, rgba(0,0,0,0.3) 10px, black 50px)',
          opacity: 0.5,
          pointerEvents: 'none',
        }} />

        {/* grain */}
        <div style={{
          position: 'absolute',
          inset: 0,
          backgroundImage: `url("data:image/svg+xml,%3Csvg viewBox='0 0 512 512' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.08'/%3E%3C/svg%3E")`,
          pointerEvents: 'none',
          opacity: 0.4,
        }} />

        <div className="hero-grid" style={{
          maxWidth: '1400px',
          margin: '0 auto',
          width: '100%',
          display: 'grid',
          gridTemplateColumns: '1fr auto',
          gap: '4rem',
          alignItems: 'center',
          position: 'relative',
        }}>

          {/* left */}
          <div>
            <motion.div {...fadeUp(0)}>
              <div style={{
                display: 'inline-flex', alignItems: 'center', gap: '0.5rem',
                padding: '0.4rem 0.9rem',
                border: '1px solid var(--border)',
                borderRadius: '2px',
                marginBottom: '1.25rem',
              }}>
                <Stethoscope size={12} style={{ color: 'var(--accent-light)' }} />
                <span className="text-subheading" style={{ color: 'var(--text-muted)' }}>
                  medical imaging · deep learning
                </span>
              </div>
            </motion.div>

            <motion.div {...fadeUp(0.06)}>
              <h1 style={{ marginBottom: '1.25rem', lineHeight: 0.92 }}>
                <span className="text-display" style={{ color: 'var(--text-primary)', display: 'block' }}>
                  Chest X-Ray
                </span>
                <span style={{
                  fontFamily: 'Instrument Serif, serif',
                  fontStyle: 'italic',
                  fontSize: 'clamp(3.5rem, 9vw, 8rem)',
                  fontWeight: 400,
                  background: 'var(--gradient-accent)',
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent',
                  backgroundClip: 'text',
                  display: 'block',
                  overflow: 'visible',
                  paddingBottom: '0.12em',
                }}>
                  Pathology
                </span>
                <span className="text-display" style={{ color: 'var(--text-primary)', display: 'block' }}>
                  Detection.
                </span>
              </h1>
            </motion.div>

            <motion.div {...fadeUp(0.14)}>
              <p className="text-body" style={{ maxWidth: '500px', marginBottom: '2rem', fontSize: '1.05rem', lineHeight: 1.8 }}>
                Six-stage machine learning pipeline that identifies pneumonia in chest X-rays, combining
                classical image processing, convolutional networks, generative models, segmentation,
                and vision-language models. Trained on 26,684 radiographs from the RSNA dataset.
              </p>
            </motion.div>

            <motion.div {...fadeUp(0.2)} style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap' }}>
              <Link href="/overview" style={{ textDecoration: 'none' }}>
                <button style={{
                  display: 'flex', alignItems: 'center', gap: '0.5rem',
                  padding: '0.85rem 1.8rem',
                  background: 'linear-gradient(135deg, #3A015C, #7B2FBE)',
                  border: 'none', borderRadius: '3px',
                  color: '#fff',
                  fontFamily: 'Hanken Grotesk, sans-serif',
                  fontSize: '0.875rem', fontWeight: 600,
                  letterSpacing: '0.05em',
                  boxShadow: '0 4px 24px rgba(123, 47, 190, 0.35)',
                }}>
                  view results <ArrowRight size={14} weight="bold" />
                </button>
              </Link>
              <Link href="/inference" style={{ textDecoration: 'none' }}>
                <button style={{
                  display: 'flex', alignItems: 'center', gap: '0.5rem',
                  padding: '0.85rem 1.8rem',
                  background: 'var(--glass-btn-bg)',
                  backdropFilter: 'blur(12px)',
                  WebkitBackdropFilter: 'blur(12px)',
                  border: '1px solid var(--glass-btn-border)',
                  borderRadius: '3px',
                  color: 'var(--glass-btn-color)',
                  fontFamily: 'Hanken Grotesk, sans-serif',
                  fontSize: '0.875rem', fontWeight: 600,
                  letterSpacing: '0.05em',
                  outline: 'none',
                  WebkitAppearance: 'none',
                }}>
                  live inference <ArrowUpRight size={14} weight="bold" />
                </button>
              </Link>
            </motion.div>
          </div>

          {/* right */}
          <XrayCardStack />
        </div>

        {/* scroll indicator */}
        <motion.div
          {...fadeUp(0.5)}
          style={{
            position: 'absolute',
            bottom: '1.75rem',
            left: 'clamp(1.5rem, 5vw, 4rem)',
            display: 'flex', alignItems: 'center', gap: '0.75rem',
          }}
        >
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '4px' }}>
            <div style={{ width: '1px', height: '28px', background: 'linear-gradient(to bottom, transparent, var(--accent-light))' }} />
            <ArrowDown size={11} style={{ color: 'var(--accent-light)', opacity: 0.6 }} />
          </div>
          <p className="text-subheading">scroll</p>
        </motion.div>
      </section>

      <div className="divider" />

      {/* stats strip */}
      <section className="section" style={{ paddingTop: 'clamp(3rem, 6vw, 5rem)', paddingBottom: 'clamp(3rem, 6vw, 5rem)' }}>
        <div className="stats-grid" style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)' }}>
          {stats.map((stat, i) => (
            <motion.div key={stat.label} {...fadeUp(i * 0.07)}>
              <div className="accent-line" style={{
                padding: '2rem 2rem 2rem 2.5rem',
                borderRight: i < stats.length - 1 ? '1px solid var(--border)' : 'none',
                height: '100%',
              }}>
                <p style={{
                  fontFamily: 'Hanken Grotesk, sans-serif',
                  fontSize: 'clamp(2.2rem, 4vw, 3.5rem)',
                  fontWeight: 800,
                  letterSpacing: '-0.04em',
                  lineHeight: 1,
                  marginBottom: '0.3rem',
                  background: 'var(--gradient-accent)',
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent',
                  backgroundClip: 'text',
                }}>
                  <CountUp value={stat.value} decimals={stat.decimals} suffix={stat.suffix} />
                </p>
                <p style={{ fontFamily: 'Hanken Grotesk, sans-serif', fontSize: '1rem', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '0.2rem' }}>
                  {stat.label}
                </p>
                <p className="text-subheading">{stat.sub}</p>
              </div>
            </motion.div>
          ))}
        </div>
      </section>

      <div className="divider" />

      {/* pipeline grid */}
      <section className="section">
        <motion.div {...fadeUp(0)}>
          <p className="text-subheading" style={{ marginBottom: '0.75rem' }}>Pipeline</p>
        </motion.div>
        <motion.div {...fadeUp(0.08)}>
          <h2 style={{ marginBottom: '3rem', maxWidth: '560px' }}>
            <span className="text-heading" style={{ color: 'var(--text-primary)' }}>Six stages, </span>
            <span className="text-heading font-display" style={{ fontStyle: 'italic', color: 'var(--text-primary)' }}>
              one cohesive system.
            </span>
          </h2>
        </motion.div>

        <div className="pipeline-grid" style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(3, 1fr)',
          border: '1px solid var(--border)',
          borderRadius: '8px',
          overflow: 'hidden',
        }}>
          {pipeline.map((item, i) => (
            <motion.div key={item.step} {...fadeUp(i * 0.05)}>
              <Link href={item.href} style={{ textDecoration: 'none', display: 'block', height: '100%' }}>
                <div className="pipeline-card" style={{
                  padding: '2rem',
                  borderRight: (i + 1) % 3 !== 0 ? '1px solid var(--border)' : 'none',
                  borderBottom: i < 3 ? '1px solid var(--border)' : 'none',
                  height: '100%',
                  display: 'flex', flexDirection: 'column',
                  background: 'rgba(50, 0, 79, 0.08)',
                  backdropFilter: 'blur(12px)',
                  WebkitBackdropFilter: 'blur(12px)',
                  transition: 'background 0.25s ease, box-shadow 0.25s ease',
                }}>
                  <p style={{
                    fontFamily: 'Instrument Serif, serif',
                    fontSize: '3rem', fontWeight: 300,
                    color: 'var(--accent-light)', opacity: 0.25,
                    lineHeight: 1, marginBottom: '1.2rem',
                  }}>
                    {item.step}
                  </p>
                  <p style={{ fontFamily: 'Hanken Grotesk, sans-serif', fontSize: '1.05rem', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '0.4rem' }}>
                    {item.label}
                  </p>
                  <p className="text-subheading" style={{ marginBottom: 'auto' }}>{item.desc}</p>
                  <div className="pipeline-arrow" style={{
                    marginTop: '1.5rem',
                    display: 'flex', alignItems: 'center', gap: '0.4rem',
                    color: 'var(--accent-light)',
                    fontFamily: 'Hanken Grotesk, sans-serif',
                    fontSize: '0.75rem', fontWeight: 600,
                    letterSpacing: '0.08em',
                    opacity: 0.6,
                    transition: 'opacity 0.2s',
                  }}>
                    explore <ArrowRight size={12} weight="bold" />
                  </div>
                </div>
              </Link>
            </motion.div>
          ))}
        </div>
      </section>

      <div className="divider" />

      {/* dataset */}
      <section className="section">
        <div className="dataset-grid" style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
          gap: '5rem',
          alignItems: 'center',
        }}>
          <div>
            <motion.div {...fadeUp(0)}>
              <p className="text-subheading" style={{ marginBottom: '0.75rem' }}>Dataset</p>
            </motion.div>
            <motion.div {...fadeUp(0.08)}>
              <h2 style={{ marginBottom: '1.5rem' }}>
                <span className="text-heading" style={{ color: 'var(--text-primary)', display: 'block' }}>RSNA Pneumonia</span>
                <span className="text-heading font-display" style={{ fontStyle: 'italic', color: 'var(--text-primary)', display: 'block' }}>Detection Challenge</span>
              </h2>
            </motion.div>
            <motion.div {...fadeUp(0.12)}>
              <p className="text-body" style={{ fontSize: '1.05rem', lineHeight: 1.8 }}>
                26,684 frontal chest radiographs annotated with bounding boxes for pneumonia opacities.
                The dataset is heavily imbalanced at 77% normal and 23% pneumonia, making
                AUC the correct metric throughout. Accuracy alone misleads on imbalanced clinical data.
              </p>
            </motion.div>
          </div>

          <motion.div {...fadeUp(0.1)}>
            {[
              { label: 'Total images', value: '26,684', pct: 100 },
              { label: 'Normal', value: '20,672', pct: 77.5 },
              { label: 'Pneumonia (opacity)', value: '6,012', pct: 22.5 },
              { label: 'Training split', value: '21,347', pct: 80 },
              { label: 'Val + Test', value: '5,337', pct: 20 },
            ].map((row, i) => (
              <div key={row.label} style={{ padding: '1rem 0', borderBottom: '1px solid var(--border)' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                  <p className="text-subheading">{row.label}</p>
                  <p style={{ fontFamily: 'Hanken Grotesk', fontSize: '0.9rem', fontWeight: 700, color: 'var(--text-primary)' }}>{row.value}</p>
                </div>
                <div style={{ height: '2px', background: 'var(--border)', borderRadius: '1px', overflow: 'hidden' }}>
                  <motion.div
                    initial={{ width: 0 }}
                    whileInView={{ width: `${row.pct}%` }}
                    viewport={{ once: true, amount: 0.1 }}
                    transition={{ duration: 1.2, delay: i * 0.08, ease: [0.25, 0.1, 0.25, 1] as [number, number, number, number] }}
                    style={{ height: '100%', background: 'var(--gradient-accent)', borderRadius: '1px' }}
                  />
                </div>
              </div>
            ))}
          </motion.div>
        </div>
      </section>

      <div className="divider" />

      {/* cta */}
      <section className="section" style={{ textAlign: 'center', paddingTop: 'clamp(5rem, 10vw, 8rem)', paddingBottom: 'clamp(5rem, 10vw, 8rem)' }}>
        <motion.div {...fadeUp(0)}>
          <p className="text-subheading" style={{ marginBottom: '1rem' }}>Try it live</p>
        </motion.div>
        <motion.div {...fadeUp(0.08)}>
          <h2 style={{ marginBottom: '2.5rem' }}>
            <span className="text-heading" style={{ color: 'var(--text-primary)', display: 'block' }}>Upload an X-ray.</span>
            <span className="text-heading font-display" style={{ fontStyle: 'italic', color: 'var(--text-secondary)', display: 'block' }}>Get a full pipeline result.</span>
          </h2>
        </motion.div>
        <motion.div {...fadeUp(0.14)} style={{ display: 'flex', gap: '0.75rem', justifyContent: 'center', flexWrap: 'wrap' }}>
          <Link href="/inference" style={{ textDecoration: 'none' }}>
            <button style={{
              display: 'inline-flex', alignItems: 'center', gap: '0.5rem',
              padding: '1rem 2.5rem',
              background: 'linear-gradient(135deg, #3A015C, #7B2FBE)',
              border: 'none', borderRadius: '3px', color: '#fff',
              fontFamily: 'Hanken Grotesk, sans-serif',
              fontSize: '0.9rem', fontWeight: 600, letterSpacing: '0.05em',
              boxShadow: '0 4px 24px rgba(123, 47, 190, 0.35)',
            }}>
              run inference <ArrowRight size={14} weight="bold" />
            </button>
          </Link>
          <Link href="/overview" style={{ textDecoration: 'none' }}>
            <button style={{
              display: 'inline-flex', alignItems: 'center', gap: '0.5rem',
              padding: '1rem 2.5rem',
              background: 'var(--glass-btn-bg)',
              backdropFilter: 'blur(12px)', WebkitBackdropFilter: 'blur(12px)',
              border: '1px solid var(--glass-btn-border)', borderRadius: '3px',
              color: 'var(--glass-btn-color)',
              outline: 'none',
              WebkitAppearance: 'none',
              fontFamily: 'Hanken Grotesk, sans-serif',
              fontSize: '0.9rem', fontWeight: 600, letterSpacing: '0.05em',
            }}>
              view all results <ArrowUpRight size={14} weight="bold" />
            </button>
          </Link>
        </motion.div>
      </section>

      <style jsx>{`
        .pipeline-card:hover {
          background: rgba(50, 0, 79, 0.28) !important;
          box-shadow: 0 0 0 1px rgba(123, 47, 190, 0.35), inset 0 1px 0 rgba(255, 237, 223, 0.05) !important;
        }
        .pipeline-card:hover .pipeline-arrow {
          opacity: 1 !important;
        }
        @media (max-width: 900px) {
          .hero-grid {
            grid-template-columns: 1fr !important;
          }
          .hero-xray-stack {
            display: none !important;
          }
        }
        @media (max-width: 640px) {
          .stats-grid {
            grid-template-columns: repeat(2, 1fr) !important;
          }
          .pipeline-grid {
            grid-template-columns: repeat(1, 1fr) !important;
          }
        }
      `}</style>
    </div>
  )
}