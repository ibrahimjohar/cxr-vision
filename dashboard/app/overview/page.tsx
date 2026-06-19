'use client'

import Link from 'next/link'
import { motion, AnimatePresence } from 'framer-motion'
import { useRef, useState, useCallback, useEffect } from 'react'
import { createPortal } from 'react-dom'
import { ArrowRight, ArrowUpRight } from '@phosphor-icons/react'

const models = [
  {
    step: '01',
    name: 'ResNet18 Classifier',
    metric: 'Val Accuracy',
    value: '85.08%',
    href: '/classifier',
    verdict: 'strong',
    note: 'ImageNet pretrained, progressive unfreezing, early stopping epoch 16',
  },
  {
    step: '02',
    name: 'VAE Anomaly Detection',
    metric: 'AUC (MSE)',
    value: '0.51',
    href: '/vae',
    verdict: 'negative',
    note: 'Chance-level — global recon error fails on localized opacities. Documented method limitation.',
  },
  {
    step: '03',
    name: 'DCGAN',
    metric: 'FID',
    value: '89.56',
    href: '/generative',
    verdict: 'moderate',
    note: 'Stable adversarial equilibrium. Best checkpoint epoch 10. 224x224.',
  },
  {
    step: '04',
    name: 'DDPM Diffusion',
    metric: 'FID',
    value: '64.09',
    href: '/generative',
    verdict: 'strong',
    note: 'DDIM 50-step sampling. Outperforms GAN by ~25 FID points. 128x128.',
  },
  {
    step: '05',
    name: 'Attention U-Net',
    metric: 'Dice (opacity-only)',
    value: '0.40',
    href: '/unet',
    verdict: 'moderate',
    note: 'Honest localization metric. Overall 0.76 inflated by all-black normal masks.',
  },
  {
    step: '06',
    name: 'BiomedCLIP Zero-Shot',
    metric: 'AUC',
    value: '0.84',
    href: '/clip',
    verdict: 'strong',
    note: '+0.33 AUC over generic CLIP. Near-supervised performance with zero task-specific training.',
  },
]

const verdictColor: Record<string, string> = {
  strong: 'var(--accent-light)',
  moderate: '#9B6DCC',
  negative: 'var(--text-muted)',
}

const findings = [
  {
    tag: 'Classifier',
    heading: 'Transfer learning works.',
    body: 'ResNet18 with ImageNet weights and progressive unfreezing hits 85.08% on a heavily imbalanced medical dataset with no domain-specific pretraining. A strong baseline for 4GB VRAM.',
  },
  {
    tag: 'VAE',
    heading: 'Global recon error fails localized pathology.',
    body: 'A well-trained VAE (KL 0.058, recon loss 0.009) achieves chance-level AUC 0.51. Pneumonia opacities are too localized for pixel-level reconstruction error to separate distributions.',
  },
  {
    tag: 'Generative',
    heading: 'Diffusion beats GAN by 25 FID points.',
    body: 'DDPM FID 64.09 vs DCGAN FID 89.56. Score-based generation produces sharper, more diverse chest anatomy without adversarial training instability.',
  },
  {
    tag: 'U-Net',
    heading: 'Overall Dice is misleading.',
    body: 'Overall Dice 0.76 is inflated by all-black normal masks scoring 0.87. The honest opacity-only Dice is 0.40 -- weak-supervision from bounding boxes limits precise boundary learning.',
  },
  {
    tag: 'CLIP',
    heading: 'Domain pretraining is the key lever.',
    body: 'Generic CLIP AUC 0.51 vs BiomedCLIP AUC 0.84 -- both zero-shot, no RSNA training. The +0.33 AUC gap isolates domain-specific pretraining as the deciding factor.',
  },
  {
    tag: 'Evaluation',
    heading: 'Accuracy lies on imbalanced data.',
    body: 'Clinical prompts gave CLIP 77% accuracy but AUC 0.44. The model predicted "normal" for nearly every sample and still scored 77% due to class imbalance. AUC exposes this.',
  },
]

function FindingCard({ item, index, onClick }: { item: typeof findings[0]; index: number; onClick: () => void }) {
  const cardRef = useRef<HTMLDivElement>(null)
  const glowRef = useRef<HTMLDivElement>(null)

  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    if (!cardRef.current || !glowRef.current) return
    const rect = cardRef.current.getBoundingClientRect()
    const x = e.clientX - rect.left
    const y = e.clientY - rect.top
    glowRef.current.style.left = x + 'px'
    glowRef.current.style.top = y + 'px'
    glowRef.current.style.opacity = '1'
  }, [])

  const handleMouseLeave = useCallback(() => {
    if (!glowRef.current) return
    glowRef.current.style.opacity = '0'
  }, [])

  return (
    <motion.div
      initial={{ opacity: 0, y: 18 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, amount: 0.1 }}
      transition={{ duration: 0.55, delay: index * 0.06 }}
      style={{ height: '100%' }}
    >
      <div
        ref={cardRef}
        onClick={onClick}
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
        className="finding-card"
        style={{
          padding: '2rem',
          background: 'var(--bg)',
          height: '100%',
          display: 'flex',
          flexDirection: 'column',
          gap: '0.75rem',
          position: 'relative',
          overflow: 'hidden',
          cursor: 'pointer',
          transition: 'background 0.2s, transform 0.25s ease, box-shadow 0.25s ease',
        }}
      >
        {/* cursor glow */}
        <div
          ref={glowRef}
          style={{
            position: 'absolute',
            width: '140px',
            height: '140px',
            borderRadius: '50%',
            background: 'radial-gradient(circle, rgba(123,47,190,0.12) 0%, transparent 70%)',
            transform: 'translate(-50%, -50%)',
            pointerEvents: 'none',
            opacity: 0,
            transition: 'opacity 0.3s',
            zIndex: 0,
          }}
        />
        <div style={{ position: 'relative', zIndex: 1, display: 'flex', flexDirection: 'column', gap: '0.75rem', height: '100%' }}>
          <p className="text-subheading" style={{ color: 'var(--accent-light)' }}>{item.tag}</p>
          <p style={{ fontFamily: 'Hanken Grotesk, sans-serif', fontSize: '0.95rem', fontWeight: 600, color: 'var(--text-primary)', lineHeight: 1.4 }}>
            {item.heading}
          </p>
          <p className="text-body" style={{ fontSize: '0.875rem', lineHeight: 1.7 }}>
            {item.body}
          </p>
        </div>
      </div>
    </motion.div>
  )
}

function FindingsGrid() {
  const [expanded, setExpanded] = useState<number | null>(null)
  const [mounted, setMounted] = useState(false)
  const active = expanded !== null ? findings[expanded] : null

  useEffect(() => { setTimeout(() => setMounted(true), 0) }, [])

  const modal = (
    <AnimatePresence>
      {expanded !== null && active && (
        <>
          <div
            onClick={() => setExpanded(null)}
            style={{
              position: 'fixed',
              inset: 0,
              background: 'rgba(8, 5, 16, 0.75)',
              backdropFilter: 'blur(12px)',
              WebkitBackdropFilter: 'blur(12px)',
              zIndex: 9998,
            }}
          />
          <div style={{
            position: 'fixed',
            inset: 0,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 9999,
            pointerEvents: 'none',
          }}>
            <div style={{ pointerEvents: 'auto' }}>
              <motion.div
                initial={{ opacity: 0, scale: 0.92, y: 24 }}
                animate={{ opacity: 1, scale: 1, y: 0 }}
                exit={{ opacity: 0, scale: 0.94, y: 16 }}
                transition={{ duration: 0.35, ease: [0.16, 1, 0.3, 1] as [number,number,number,number] }}
                style={{
                  width: 'min(600px, calc(100vw - 3rem))',
                  background: 'var(--bg-secondary)',
                  border: '1px solid var(--border-accent)',
                  borderRadius: '12px',
                  padding: '2.5rem',
                  boxShadow: '0 32px 80px rgba(58,1,92,0.45)',
                }}
              >
                <p className="text-subheading" style={{ color: 'var(--accent-light)', marginBottom: '1rem' }}>
                  {active.tag}
                </p>
                <h3 style={{
                  fontFamily: 'Instrument Serif, serif',
                  fontSize: 'clamp(1.5rem, 3vw, 2.2rem)',
                  fontWeight: 400,
                  color: 'var(--text-primary)',
                  lineHeight: 1.2,
                  marginBottom: '1.25rem',
                }}>
                  {active.heading}
                </h3>
                <p className="text-body" style={{ fontSize: '1rem', lineHeight: 1.85 }}>
                  {active.body}
                </p>
                <button
                  onClick={() => setExpanded(null)}
                  style={{
                    marginTop: '2rem',
                    padding: '0.6rem 1.4rem',
                    background: 'var(--surface)',
                    border: '1px solid var(--border)',
                    borderRadius: '4px',
                    color: 'var(--text-secondary)',
                    fontFamily: 'Hanken Grotesk, sans-serif',
                    fontSize: '0.8rem',
                    fontWeight: 600,
                    letterSpacing: '0.06em',
                  }}
                >
                  close
                </button>
              </motion.div>
            </div>
          </div>
        </>
      )}
    </AnimatePresence>
  )

  return (
    <>
      <div
        className="findings-grid"
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(3, 1fr)',
          gap: '1px',
          background: 'var(--border)',
          borderRadius: '8px',
          overflow: 'hidden',
        }}
      >
        {findings.map((item, i) => (
          <FindingCard key={item.tag} item={item} index={i} onClick={() => setExpanded(i)} />
        ))}
      </div>
      {mounted && createPortal(modal, document.body)}
    </>
  )
}

const fadeUp = (delay = 0) => ({
  initial: { opacity: 0, y: 18 },
  whileInView: { opacity: 1, y: 0 },
  viewport: { once: true, amount: 0.1 } as const,
  transition: { duration: 0.55, delay },
})

export default function OverviewPage() {
  return (
    <div style={{ position: 'relative' }}>

      {/* page header */}
      <section className="section" style={{ paddingBottom: '2rem' }}>
        <motion.div {...fadeUp(0)}>
          <p className="text-subheading" style={{ marginBottom: '0.75rem' }}>Overview</p>
        </motion.div>
        <motion.div {...fadeUp(0.06)}>
          <h1 style={{ marginBottom: '1.5rem' }}>
            <span className="text-display" style={{ color: 'var(--text-primary)', display: 'block', fontSize: 'clamp(2.5rem, 6vw, 5rem)' }}>
              Pipeline
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
              Summary.
            </span>
          </h1>
        </motion.div>
        <motion.div {...fadeUp(0.12)}>
          <p className="text-body" style={{ maxWidth: '560px', fontSize: '1.05rem', lineHeight: 1.8 }}>
            Six models trained and evaluated on 26,684 RSNA chest radiographs.
            Results are reported honestly, including a documented negative result on VAE anomaly detection
            and a methodological trap exposed in zero-shot accuracy reporting.
          </p>
        </motion.div>
      </section>

      <div className="divider" />

      {/* dataset strip */}
      <section className="section" style={{ paddingTop: 'clamp(2.5rem, 5vw, 4rem)', paddingBottom: 'clamp(2.5rem, 5vw, 4rem)' }}>
        <motion.div {...fadeUp(0)}>
          <p className="text-subheading" style={{ marginBottom: '2rem' }}>Dataset — RSNA Pneumonia Detection Challenge</p>
        </motion.div>
        <div className="overview-dataset-grid" style={{ display: 'grid', gridTemplateColumns: 'repeat(6, 1fr)', gap: '0' }}>
          {[
            { label: 'Total', value: '26,684' },
            { label: 'Normal', value: '20,672' },
            { label: 'Pneumonia', value: '6,012' },
            { label: 'Train', value: '21,347' },
            { label: 'Val', value: '2,668' },
            { label: 'Test', value: '2,669' },
          ].map((item, i) => (
            <motion.div key={item.label} {...fadeUp(i * 0.05)}>
              <div style={{
                padding: '1.5rem 1.5rem 1.5rem 2rem',
                borderRight: i < 5 ? '1px solid var(--border)' : 'none',
                position: 'relative',
              }}>
                {i === 0 && (
                  <div style={{ position: 'absolute', left: 0, top: 0, bottom: 0, width: '2px', background: 'var(--gradient-accent)' }} />
                )}
                <p style={{
                  fontFamily: 'Hanken Grotesk, sans-serif',
                  fontSize: 'clamp(1.4rem, 2.5vw, 2rem)',
                  fontWeight: 800,
                  letterSpacing: '-0.03em',
                  color: 'var(--text-primary)',
                  marginBottom: '0.2rem',
                }}>
                  {item.value}
                </p>
                <p className="text-subheading">{item.label}</p>
              </div>
            </motion.div>
          ))}
        </div>

        <motion.div {...fadeUp(0.2)} style={{ marginTop: '2rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
            <p className="text-subheading">class distribution</p>
            <p className="text-subheading">77.5% normal · 22.5% pneumonia</p>
          </div>
          <div style={{ height: '6px', background: 'var(--border)', borderRadius: '3px', overflow: 'hidden' }}>
            <div style={{ display: 'flex', height: '100%' }}>
              <motion.div
                initial={{ width: 0 }}
                whileInView={{ width: '77.5%' }}
                viewport={{ once: true, amount: 0.1 }}
                transition={{ duration: 1.2, ease: [0.25, 0.1, 0.25, 1] as [number,number,number,number] }}
                style={{ height: '100%', background: 'var(--gradient-accent)', borderRadius: '3px 0 0 3px' }}
              />
              <motion.div
                initial={{ width: 0 }}
                whileInView={{ width: '22.5%' }}
                viewport={{ once: true, amount: 0.1 }}
                transition={{ duration: 1.2, delay: 0.1, ease: [0.25, 0.1, 0.25, 1] as [number,number,number,number] }}
                style={{ height: '100%', background: 'rgba(123,47,190,0.35)', borderRadius: '0 3px 3px 0' }}
              />
            </div>
          </div>
          <p className="text-body" style={{ marginTop: '0.75rem', fontSize: '0.875rem' }}>
            Heavily imbalanced — AUC is the primary metric throughout. Accuracy alone misleads on this distribution.
          </p>
        </motion.div>
      </section>

      <div className="divider" />

      {/* model results table */}
      <section className="section">
        <motion.div {...fadeUp(0)}>
          <p className="text-subheading" style={{ marginBottom: '0.75rem' }}>Model Results</p>
        </motion.div>
        <motion.div {...fadeUp(0.08)}>
          <h2 style={{ marginBottom: '3rem' }}>
            <span className="text-heading" style={{ color: 'var(--text-primary)' }}>Six models, </span>
            <span className="text-heading font-display" style={{ fontStyle: 'italic', color: 'var(--text-primary)' }}>honest evaluation.</span>
          </h2>
        </motion.div>

        <div style={{ display: 'flex', flexDirection: 'column', border: '1px solid var(--border)', borderRadius: '8px', overflow: 'hidden' }}>
          <div style={{
            display: 'grid',
            gridTemplateColumns: '3rem 1fr 140px 120px 1fr 80px',
            padding: '0.75rem 1.5rem',
            background: 'var(--surface)',
            borderBottom: '1px solid var(--border)',
            gap: '1rem',
            alignItems: 'center',
          }}>
            {['#', 'Model', 'Metric', 'Result', 'Notes', ''].map((h, i) => (
              <p key={i} className="text-subheading">{h}</p>
            ))}
          </div>

          {models.map((model, i) => (
            <motion.div key={model.step} {...fadeUp(i * 0.05)}>
              <Link href={model.href} style={{ textDecoration: 'none', display: 'block' }}>
                <div
                  className="overview-row"
                  style={{
                    display: 'grid',
                    gridTemplateColumns: '3rem 1fr 140px 120px 1fr 80px',
                    padding: '1.25rem 1.5rem',
                    borderBottom: i < models.length - 1 ? '1px solid var(--border)' : 'none',
                    gap: '1rem',
                    alignItems: 'center',
                    transition: 'background 0.2s',
                  }}
                >
                  <p style={{ fontFamily: 'Instrument Serif, serif', fontSize: '1.4rem', color: 'var(--accent-light)', opacity: 0.35, lineHeight: 1 }}>
                    {model.step}
                  </p>
                  <p style={{ fontFamily: 'Hanken Grotesk, sans-serif', fontSize: '0.95rem', fontWeight: 600, color: 'var(--text-primary)' }}>
                    {model.name}
                  </p>
                  <p className="text-subheading">{model.metric}</p>
                  <p style={{ fontFamily: 'Hanken Grotesk, sans-serif', fontSize: '1.1rem', fontWeight: 800, letterSpacing: '-0.02em', color: verdictColor[model.verdict] }}>
                    {model.value}
                  </p>
                  <p className="text-body" style={{ fontSize: '0.8rem', lineHeight: 1.5 }}>{model.note}</p>
                  <div className="overview-arrow" style={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end', color: 'var(--accent-light)', opacity: 0.5, transition: 'opacity 0.2s' }}>
                    <ArrowRight size={14} weight="bold" />
                  </div>
                </div>
              </Link>
            </motion.div>
          ))}
        </div>

        <motion.div {...fadeUp(0.3)} style={{ display: 'flex', gap: '2rem', marginTop: '1.25rem' }}>
          {[
            { color: 'var(--accent-light)', label: 'Strong result' },
            { color: '#9B6DCC', label: 'Moderate result' },
            { color: 'var(--text-muted)', label: 'Negative / chance-level' },
          ].map(item => (
            <div key={item.label} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <div style={{ width: '8px', height: '8px', borderRadius: '50%', background: item.color }} />
              <p className="text-subheading">{item.label}</p>
            </div>
          ))}
        </motion.div>
      </section>

      <div className="divider" />

      {/* key findings */}
      <section className="section">
        <motion.div {...fadeUp(0)}>
          <p className="text-subheading" style={{ marginBottom: '0.75rem' }}>Key Findings</p>
        </motion.div>
        <motion.div {...fadeUp(0.08)}>
          <h2 style={{ marginBottom: '3rem' }}>
            <span className="text-heading" style={{ color: 'var(--text-primary)' }}>What the numbers </span>
            <span className="text-heading font-display" style={{ fontStyle: 'italic', color: 'var(--text-primary)' }}>actually mean.</span>
          </h2>
        </motion.div>
        <FindingsGrid />
      </section>

      <div className="divider" />

      {/* page navigation */}
      <section className="section" style={{ paddingTop: 'clamp(3rem, 6vw, 5rem)', paddingBottom: 'clamp(3rem, 6vw, 5rem)' }}>
        <motion.div {...fadeUp(0)}>
          <p className="text-subheading" style={{ marginBottom: '2rem' }}>Explore each model</p>
        </motion.div>
        <div className="nav-pages-grid" style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1rem' }}>
          {[
            { label: 'Preprocessing', sub: 'Classical CV', href: '/preprocessing' },
            { label: 'Classifier', sub: 'ResNet18 · 85.08%', href: '/classifier' },
            { label: 'VAE', sub: 'Anomaly Detection · AUC 0.51', href: '/vae' },
            { label: 'U-Net', sub: 'Segmentation · Dice 0.40', href: '/unet' },
            { label: 'CLIP', sub: 'Zero-Shot · AUC 0.84', href: '/clip' },
            { label: 'Generative', sub: 'GAN · Diffusion · FID 64.1', href: '/generative' },
          ].map((page, i) => (
            <motion.div key={page.href} {...fadeUp(i * 0.05)}>
              <Link href={page.href} style={{ textDecoration: 'none', display: 'block' }}>
                <div className="nav-page-card" style={{
                  padding: '1.5rem',
                  border: '1px solid var(--border)',
                  borderRadius: '6px',
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  transition: 'border-color 0.2s, background 0.2s',
                }}>
                  <div>
                    <p style={{ fontFamily: 'Hanken Grotesk, sans-serif', fontSize: '0.95rem', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '0.2rem' }}>
                      {page.label}
                    </p>
                    <p className="text-subheading">{page.sub}</p>
                  </div>
                  <ArrowUpRight size={16} style={{ color: 'var(--accent-light)', opacity: 0.5 }} />
                </div>
              </Link>
            </motion.div>
          ))}
        </div>
      </section>

      <style jsx>{`
        .finding-card:hover {
          background: var(--surface) !important;
          box-shadow: inset 0 0 0 1px var(--border-accent);
          transform: scale(1.018);
        }
        [data-theme="light"] .finding-card:hover {
          background: rgba(58, 1, 92, 0.05) !important;
        }
        .overview-row:hover {
          background: var(--surface) !important;
        }
        .overview-row:hover .overview-arrow {
          opacity: 1 !important;
        }
        .nav-page-card:hover {
          border-color: var(--border-accent) !important;
          background: var(--surface) !important;
        }
        [data-theme="light"] .overview-row:hover {
          background: rgba(58, 1, 92, 0.04) !important;
        }
        @media (max-width: 900px) {
          .findings-grid {
            grid-template-columns: repeat(2, 1fr) !important;
          }
          .nav-pages-grid {
            grid-template-columns: repeat(2, 1fr) !important;
          }
        }
        @media (max-width: 640px) {
          .overview-dataset-grid {
            grid-template-columns: repeat(3, 1fr) !important;
          }
          .findings-grid {
            grid-template-columns: repeat(1, 1fr) !important;
          }
          .nav-pages-grid {
            grid-template-columns: repeat(1, 1fr) !important;
          }
        }
      `}</style>
    </div>
  )
}