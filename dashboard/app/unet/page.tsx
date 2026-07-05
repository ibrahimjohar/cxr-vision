'use client'

import Image from 'next/image'
import Link from 'next/link'
import { motion, AnimatePresence, useInView, animate } from 'framer-motion'
import { useRef, useState, useEffect } from 'react'
import { ArrowLeft, ArrowRight, CheckCircle } from '@phosphor-icons/react'

function CountUp({ to, decimals = 0, suffix = '', duration = 1.8 }: { to: number; decimals?: number; suffix?: string; duration?: number }) {
  const ref = useRef<HTMLSpanElement>(null)
  const inView = useInView(ref as React.RefObject<Element>, { once: true, margin: '-40px' })
  const [val, setVal] = useState('0')

  useEffect(() => {
    if (!inView) return
    const ctrl = animate(0, to, {
      duration,
      ease: [0.25, 0.1, 0.25, 1] as [number, number, number, number],
      onUpdate: (v) => setVal(decimals === 0 ? Math.floor(v).toLocaleString() : v.toFixed(decimals)),
    })
    return ctrl.stop
  }, [inView, to, decimals, duration])

  return <span ref={ref}>{val}{suffix}</span>
}

const fadeUp = (delay = 0) => ({
  initial: { opacity: 0, y: 18 },
  whileInView: { opacity: 1, y: 0 },
  viewport: { once: true, amount: 0.1 } as const,
  transition: { duration: 0.55, delay },
})

const metrics = [
  { value: 0.763, label: 'Dice (Overall)', sub: 'Inflated by all-black normal masks', decimals: 3 },
  { value: 0.4036, label: 'Dice (Opacity-only)', sub: 'Honest localization metric — n=601', decimals: 4 },
  { value: 0.87, label: 'Dice (Normal-only)', sub: 'False-positive suppression — n=2,068', decimals: 2 },
  { value: 16, label: 'Best Epoch', sub: 'Out of 20 scheduled', decimals: 0 },
]

const layers = [
  { label: 'Input', detail: '224 × 224 × 1 grayscale', type: 'io' },
  { label: 'Encoder Block 1', detail: '32 channels · conv + maxpool', type: 'backbone' },
  { label: 'Encoder Block 2', detail: '64 channels', type: 'backbone' },
  { label: 'Encoder Block 3', detail: '128 channels', type: 'backbone' },
  { label: 'Bottleneck', detail: '256 channels', type: 'reg' },
  { label: 'Attention Gates ×3', detail: 'Gate each skip connection before concatenation', type: 'head' },
  { label: 'Decoder Blocks 1–3', detail: 'Upsample + concat + conv, 256 → 32', type: 'backbone' },
  { label: 'Output', detail: '1×1 conv → sigmoid mask, 224 × 224 × 1', type: 'io' },
]

const layerColor: Record<string, string> = {
  io: 'var(--text-muted)',
  backbone: 'var(--accent-light)',
  reg: '#9B6DCC',
  head: '#C084FC',
}

const trainingConfig = [
  { label: 'Backbone features', value: '[32, 64, 128, 256] — reduced for 4GB VRAM' },
  { label: 'Supervision', value: 'RSNA bounding boxes → binary masks' },
  { label: 'Loss', value: 'Combined Dice + BCE' },
  { label: 'Epochs', value: '20 (best: 16)' },
  { label: 'Learning rate', value: '1e-4' },
  { label: 'Training data', value: 'All 21,347 images — opacity + normal' },
  { label: 'Parameters', value: '7,851,197' },
]

const maskSamples = [
  { epoch: 1, path: '/results/images/unet_masks_epoch001.png' },
  { epoch: 5, path: '/results/images/unet_masks_epoch005.png' },
  { epoch: 10, path: '/results/images/unet_masks_epoch010.png' },
  { epoch: 15, path: '/results/images/unet_masks_epoch015.png' },
  { epoch: 20, path: '/results/images/unet_masks_epoch020.png' },
]

const verdicts = [
  {
    heading: 'Overall Dice is misleading.',
    body: 'The 0.763 overall score is carried by 2,068 all-black normal masks scoring 0.87 just by predicting near-empty. That\u2019s not localization skill — it\u2019s the easy 77.5% of the dataset.',
  },
  {
    heading: 'Opacity-only Dice (0.40) is the honest number.',
    body: 'Restricted to the 601 cases that actually contain an opacity, the model captures the general region but struggles to match boundaries precisely — this is the metric that reflects real clinical usefulness.',
  },
  {
    heading: 'Coarse box supervision is the bottleneck, not capacity.',
    body: 'RSNA labels are rectangular bounding boxes, not pixel-accurate contours. Training on boxes teaches the network to predict blob-like rectangular regions, which caps the achievable Dice regardless of further training.',
  },
]

export default function UnetPage() {
  const [activeEpoch, setActiveEpoch] = useState(4) // default: epoch 20
  const [activeLayer, setActiveLayer] = useState<number | null>(null)
  const current = maskSamples[activeEpoch]

  return (
    <div>

      {/* header */}
      <section className="section" style={{ paddingBottom: '2rem' }}>
        <motion.div {...fadeUp(0)}>
          <p className="text-subheading" style={{ marginBottom: '0.75rem' }}>Stage 04 · Segmentation</p>
        </motion.div>
        <motion.div {...fadeUp(0.06)}>
          <h1 style={{ marginBottom: '1.5rem', lineHeight: 0.95 }}>
            <span className="text-display" style={{ color: 'var(--text-primary)', display: 'block', fontSize: 'clamp(2.5rem, 6vw, 5rem)' }}>
              Attention
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
              U-Net.
            </span>
          </h1>
        </motion.div>
        <motion.div {...fadeUp(0.12)}>
          <p className="text-body" style={{ maxWidth: '600px', fontSize: '1.05rem', lineHeight: 1.8 }}>
            A 4-level encoder/decoder with attention gates, trained to localize pneumonia opacities
            from real RSNA bounding boxes. Overall Dice looks strong at 0.76 — but that number is
            carried by empty normal masks. The honest opacity-only Dice is 0.40.
          </p>
        </motion.div>
      </section>

      <div className="divider" />

      {/* metrics strip */}
      <section className="section" style={{ paddingTop: 'clamp(2.5rem, 5vw, 4rem)', paddingBottom: 'clamp(2.5rem, 5vw, 4rem)' }}>
        <div className="metrics-grid" style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)' }}>
          {metrics.map((m, i) => (
            <motion.div key={m.label} {...fadeUp(i * 0.07)}>
              <div className="accent-line" style={{
                padding: '2rem 2rem 2rem 2.5rem',
                borderRight: i < metrics.length - 1 ? '1px solid var(--border)' : 'none',
                height: '100%',
              }}>
                <p style={{
                  fontFamily: 'Hanken Grotesk, sans-serif',
                  fontSize: 'clamp(2rem, 3.5vw, 3rem)',
                  fontWeight: 800,
                  letterSpacing: '-0.04em',
                  lineHeight: 1,
                  marginBottom: '0.3rem',
                  background: 'var(--gradient-accent)',
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent',
                  backgroundClip: 'text',
                }}>
                  <CountUp to={m.value} decimals={m.decimals} />
                </p>
                <p style={{ fontFamily: 'Hanken Grotesk', fontSize: '0.9rem', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '0.2rem' }}>{m.label}</p>
                <p className="text-subheading">{m.sub}</p>
              </div>
            </motion.div>
          ))}
        </div>
      </section>

      <div className="divider" />

      {/* architecture + training config */}
      <section className="section">
        <div className="arch-config-grid" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '4rem', alignItems: 'start' }}>

          {/* left: architecture */}
          <div>
            <motion.div {...fadeUp(0)}>
              <p className="text-subheading" style={{ marginBottom: '0.5rem' }}>Architecture</p>
            </motion.div>
            <motion.div {...fadeUp(0.06)}>
              <h2 style={{ marginBottom: '1.5rem' }}>
                <span className="text-heading" style={{ color: 'var(--text-primary)' }}>Encoder, </span>
                <span className="text-heading font-display" style={{ fontStyle: 'italic', color: 'var(--text-primary)' }}>gates, decoder.</span>
              </h2>
            </motion.div>
            <motion.div {...fadeUp(0.1)}>
              <p className="text-body" style={{ fontSize: '0.95rem', lineHeight: 1.75, marginBottom: '1.5rem' }}>
                Hover a layer to inspect it. Attention gates sit on every skip connection, deciding what
                the decoder is allowed to see from the encoder.
              </p>
            </motion.div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '0', border: '1px solid var(--border)', borderRadius: '8px', overflow: 'hidden' }}>
              {layers.map((layer, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, x: -12 }}
                  whileInView={{ opacity: 1, x: 0 }}
                  viewport={{ once: true, amount: 0.1 }}
                  transition={{ duration: 0.4, delay: i * 0.06 }}
                >
                  <div
                    className="layer-row"
                    onMouseEnter={() => setActiveLayer(i)}
                    onMouseLeave={() => setActiveLayer(null)}
                    style={{
                      display: 'flex', alignItems: 'center', gap: '1rem',
                      padding: '0.9rem 1.25rem',
                      borderBottom: i < layers.length - 1 ? '1px solid var(--border)' : 'none',
                      background: activeLayer === i ? 'var(--surface)' : 'transparent',
                      transition: 'background 0.2s',
                    }}
                  >
                    <div style={{ width: '3px', height: '2rem', borderRadius: '2px', background: layerColor[layer.type], flexShrink: 0, opacity: activeLayer === i ? 1 : 0.5, transition: 'opacity 0.2s' }} />
                    <div style={{ flex: 1 }}>
                      <p style={{ fontFamily: 'Hanken Grotesk', fontSize: '0.875rem', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '0.15rem' }}>{layer.label}</p>
                      <p className="text-subheading" style={{ fontSize: '0.62rem' }}>{layer.detail}</p>
                    </div>
                    <span style={{ fontFamily: 'Hanken Grotesk', fontSize: '0.6rem', fontWeight: 700, letterSpacing: '0.1em', textTransform: 'uppercase', color: layerColor[layer.type], opacity: 0.8, flexShrink: 0 }}>
                      {layer.type}
                    </span>
                  </div>
                </motion.div>
              ))}
            </div>

            <motion.div {...fadeUp(0.3)} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginTop: '1rem' }}>
              <div style={{ width: '1px', height: '24px', background: 'var(--gradient-accent)' }} />
              <p className="text-subheading">params: 7,851,197</p>
            </motion.div>
          </div>

          {/* right: training config */}
          <div>
            <motion.div {...fadeUp(0)}>
              <p className="text-subheading" style={{ marginBottom: '0.5rem' }}>Training Setup</p>
            </motion.div>
            <motion.div {...fadeUp(0.06)}>
              <h2 style={{ marginBottom: '1.5rem' }}>
                <span className="text-heading" style={{ color: 'var(--text-primary)' }}>Real boxes, </span>
                <span className="text-heading font-display" style={{ fontStyle: 'italic', color: 'var(--text-primary)' }}>combined loss.</span>
              </h2>
            </motion.div>
            <motion.div {...fadeUp(0.1)}>
              <p className="text-body" style={{ fontSize: '0.95rem', lineHeight: 1.75, marginBottom: '1.5rem' }}>
                Every RSNA bounding box was rasterized into a binary mask. Normal images get an
                all-black mask — the model must learn to predict nothing there too.
              </p>
            </motion.div>

            <motion.div {...fadeUp(0.15)} style={{ border: '1px solid var(--border)', borderRadius: '8px', overflow: 'hidden', marginBottom: '2rem' }}>
              {trainingConfig.map((row, i, arr) => (
                <div key={row.label} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '0.7rem 1.1rem', borderBottom: i < arr.length - 1 ? '1px solid var(--border)' : 'none', background: i % 2 === 0 ? 'transparent' : 'var(--surface)' }}>
                  <p className="text-subheading">{row.label}</p>
                  <p style={{ fontFamily: 'Hanken Grotesk', fontSize: '0.82rem', fontWeight: 700, color: 'var(--text-primary)', textAlign: 'right' }}>{row.value}</p>
                </div>
              ))}
            </motion.div>

            <motion.div {...fadeUp(0.2)} style={{ padding: '1.5rem', border: '1px solid var(--border-accent)', borderRadius: '8px', background: 'var(--surface)' }}>
              <p style={{ fontFamily: 'Hanken Grotesk', fontSize: '0.875rem', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '0.5rem' }}>
                What do the attention gates do?
              </p>
              <p className="text-body" style={{ fontSize: '0.875rem', lineHeight: 1.75 }}>
                Each gate learns to suppress irrelevant background — ribs, mediastinum, tubing — in the
                encoder&rsquo;s skip connections before they reach the decoder. In principle this should reduce
                false positives. The epoch progression below shows how much of that actually happens.
              </p>
            </motion.div>
          </div>
        </div>
      </section>

      <div className="divider" />

      {/* mask evolution + verdict */}
      <section className="section">
        <div className="mask-verdict-grid" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '4rem', alignItems: 'start' }}>

          {/* left: mask viewer */}
          <div>
            <motion.div {...fadeUp(0)}>
              <p className="text-subheading" style={{ marginBottom: '0.5rem' }}>Localization Over Training</p>
            </motion.div>
            <motion.div {...fadeUp(0.06)}>
              <h2 style={{ marginBottom: '1.5rem' }}>
                <span className="text-heading" style={{ color: 'var(--text-primary)' }}>From noise to </span>
                <span className="text-heading font-display" style={{ fontStyle: 'italic', color: 'var(--text-primary)' }}>near-boxes.</span>
              </h2>
            </motion.div>
            <motion.div {...fadeUp(0.1)}>
              <p className="text-body" style={{ fontSize: '0.95rem', lineHeight: 1.8, marginBottom: '1.5rem' }}>
                Four patients per epoch: input, ground-truth mask, predicted heatmap. Row one is a
                normal case — watch how much residual false-positive heat survives even at epoch 20.
              </p>
            </motion.div>

            {/* epoch tabs */}
            <motion.div {...fadeUp(0.14)} style={{ display: 'flex', gap: '0', marginBottom: '1.5rem', border: '1px solid var(--border)', borderRadius: '6px', overflow: 'hidden', width: 'fit-content' }}>
              {maskSamples.map((s, i) => (
                <button
                  key={s.epoch}
                  onClick={() => setActiveEpoch(i)}
                  style={{
                    padding: '0.6rem 1.4rem',
                    background: activeEpoch === i ? 'linear-gradient(135deg, #3A015C, #7B2FBE)' : 'transparent',
                    border: 'none',
                    borderRight: i < maskSamples.length - 1 ? '1px solid var(--border)' : 'none',
                    color: activeEpoch === i ? '#fff' : 'var(--text-muted)',
                    fontFamily: 'Hanken Grotesk, sans-serif',
                    fontSize: '0.78rem',
                    fontWeight: 600,
                    letterSpacing: '0.06em',
                    transition: 'background 0.2s, color 0.2s',
                  }}
                >
                  Epoch {s.epoch}
                </button>
              ))}
            </motion.div>

            <AnimatePresence mode="wait">
              <motion.div
                key={current.epoch}
                initial={{ opacity: 0, scale: 0.98 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.98 }}
                transition={{ duration: 0.35, ease: [0.25, 0.1, 0.25, 1] }}
              >
                <div style={{
                  position: 'relative',
                  width: '100%',
                  maxWidth: '460px',
                  aspectRatio: '857 / 1180',
                  borderRadius: '10px',
                  overflow: 'hidden',
                  border: '1px solid var(--border)',
                  background: '#fff',
                }}>
                  <Image
                    src={current.path}
                    alt={`U-Net mask predictions — epoch ${current.epoch}`}
                    fill
                    sizes="460px"
                    style={{ objectFit: 'contain' }}
                  />
                </div>
              </motion.div>
            </AnimatePresence>
          </div>

          {/* right: honest verdict cards */}
          <div>
            <motion.div {...fadeUp(0)}>
              <p className="text-subheading" style={{ marginBottom: '0.5rem' }}>What This Means</p>
            </motion.div>
            <motion.div {...fadeUp(0.06)}>
              <h2 style={{ marginBottom: '1.5rem' }}>
                <span className="text-heading" style={{ color: 'var(--text-primary)' }}>Honest </span>
                <span className="text-heading font-display" style={{ fontStyle: 'italic', color: 'var(--text-primary)' }}>reading.</span>
              </h2>
            </motion.div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '1px', border: '1px solid var(--border)', borderRadius: '8px', overflow: 'hidden' }}>
              {verdicts.map((item, i, arr) => (
                <motion.div key={item.heading} initial={{ opacity: 0, y: 10 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true, amount: 0.1 }} transition={{ duration: 0.45, delay: i * 0.07 }}>
                  <div style={{ padding: '1.25rem', borderBottom: i < arr.length - 1 ? '1px solid var(--border)' : 'none', display: 'flex', gap: '1rem', alignItems: 'flex-start' }}>
                    <div style={{ paddingTop: '2px', flexShrink: 0 }}>
                      <CheckCircle size={16} style={{ color: 'var(--accent-light)', opacity: 0.8 }} weight="fill" />
                    </div>
                    <div>
                      <p style={{ fontFamily: 'Hanken Grotesk', fontSize: '0.875rem', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '0.4rem', lineHeight: 1.4 }}>{item.heading}</p>
                      <p className="text-body" style={{ fontSize: '0.85rem', lineHeight: 1.7 }}>{item.body}</p>
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>
          </div>
        </div>
      </section>

      <div className="divider" />

      {/* page nav */}
      <section className="section" style={{ paddingTop: 'clamp(2rem, 5vw, 4rem)', paddingBottom: 'clamp(2rem, 5vw, 4rem)' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Link href="/vae" style={{ textDecoration: 'none' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--text-muted)', fontFamily: 'Hanken Grotesk', fontSize: '0.8rem', fontWeight: 600, letterSpacing: '0.06em' }}>
              <ArrowLeft size={13} weight="bold" /> vae
            </div>
          </Link>
          <Link href="/clip" style={{ textDecoration: 'none' }}>
            <button style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', padding: '0.75rem 1.6rem', background: 'linear-gradient(135deg, #3A015C, #7B2FBE)', border: 'none', borderRadius: '3px', color: '#fff', fontFamily: 'Hanken Grotesk', fontSize: '0.8rem', fontWeight: 600, letterSpacing: '0.05em', boxShadow: '0 4px 24px rgba(123,47,190,0.3)' }}>
              next: clip <ArrowRight size={13} weight="bold" />
            </button>
          </Link>
        </div>
      </section>

      <style jsx>{`
        .layer-row:hover { background: var(--surface) !important; }
        @media (max-width: 900px) {
          .arch-config-grid { grid-template-columns: 1fr !important; }
          .mask-verdict-grid { grid-template-columns: 1fr !important; }
          .metrics-grid { grid-template-columns: repeat(2, 1fr) !important; }
        }
      `}</style>
    </div>
  )
}