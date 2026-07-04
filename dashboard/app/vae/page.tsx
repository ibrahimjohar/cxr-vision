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
  { value: 0.5105, label: 'AUC (MSE)', sub: 'Anomaly detection — chance level', decimals: 4 },
  { value: 0.4829, label: 'AUC (SSIM)', sub: 'Alternative similarity score', decimals: 4 },
  { value: 0.0094, label: 'Val Recon Loss', sub: 'Best checkpoint, epoch 18', decimals: 4 },
  { value: 0.058, label: 'KL Divergence', sub: 'At convergence — nonzero, healthy', decimals: 3 },
]

const layers = [
  { label: 'Input', detail: '224 × 224 × 1 grayscale', type: 'io' },
  { label: 'Conv Encoder', detail: '4 conv layers · stride 2 · 224 → 14', type: 'backbone' },
  { label: 'Flatten', detail: '14 × 14 × 256 → 50,176-dim vector', type: 'reg' },
  { label: 'fc_mu / fc_logvar', detail: 'Linear → 128-dim latent parameters', type: 'head' },
  { label: 'Reparameterize', detail: 'z = μ + σ · ε, ε ~ N(0, 1)', type: 'reg' },
  { label: 'Conv Decoder', detail: '4 transpose-conv layers · 14 → 224', type: 'backbone' },
  { label: 'Output', detail: 'Reconstructed 224 × 224 × 1', type: 'io' },
]

const layerColor: Record<string, string> = {
  io: 'var(--text-muted)',
  backbone: 'var(--accent-light)',
  reg: '#9B6DCC',
  head: '#C084FC',
}

const trainingConfig = [
  { label: 'Latent dimension', value: '128' },
  { label: 'β (KL weight)', value: '0.1 — fixed, no annealing' },
  { label: 'Training data', value: 'Normal-only · 16,537 images' },
  { label: 'Epochs', value: '20 (best: 18)' },
  { label: 'Learning rate', value: '1e-3' },
  { label: 'Patience', value: '5 epochs' },
  { label: 'Parameters', value: '20,696,001' },
]

const reconSamples = [
  { epoch: 1, path: '/results/images/vae_recon_epoch001.png' },
  { epoch: 5, path: '/results/images/vae_recon_epoch005.png' },
  { epoch: 10, path: '/results/images/vae_recon_epoch010.png' },
  { epoch: 15, path: '/results/images/vae_recon_epoch015.png' },
  { epoch: 20, path: '/results/images/vae_recon_epoch020.png' },
]

const verdicts = [
  {
    heading: 'Reconstruction error barely separates the classes.',
    body: 'Normal and pneumonia reconstruction-error distributions overlap almost completely — a separation ratio of just 1.037×. The histogram shows two nearly identical curves, not two distinguishable populations.',
  },
  {
    heading: 'SSIM doesn\u2019t rescue it either.',
    body: 'Swapping MSE for structural similarity as the anomaly score gives AUC 0.4829 — no better than chance. The failure isn\u2019t a quirk of one distance metric; it\u2019s representational.',
  },
  {
    heading: 'A documented method limitation, not a bug.',
    body: 'A healthy, nonzero KL divergence (0.058) and low reconstruction loss (0.0094) confirm the VAE trained correctly. Global pixel-level error is simply too coarse to catch pathology confined to a small region of the lung field.',
  },
]

export default function VaePage() {
  const [activeEpoch, setActiveEpoch] = useState(4) // default: epoch 20
  const [activeLayer, setActiveLayer] = useState<number | null>(null)
  const current = reconSamples[activeEpoch]

  return (
    <div>

      {/* header */}
      <section className="section" style={{ paddingBottom: '2rem' }}>
        <motion.div {...fadeUp(0)}>
          <p className="text-subheading" style={{ marginBottom: '0.75rem' }}>Stage 03 · Anomaly Detection</p>
        </motion.div>
        <motion.div {...fadeUp(0.06)}>
          <h1 style={{ marginBottom: '1.5rem', lineHeight: 0.95 }}>
            <span className="text-display" style={{ color: 'var(--text-primary)', display: 'block', fontSize: 'clamp(2.5rem, 6vw, 5rem)' }}>
              Variational
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
              Autoencoder.
            </span>
          </h1>
        </motion.div>
        <motion.div {...fadeUp(0.12)}>
          <p className="text-body" style={{ maxWidth: '600px', fontSize: '1.05rem', lineHeight: 1.8 }}>
            Trained exclusively on normal chest X-rays to learn what healthy anatomy looks like, then used to
            flag pathology as high reconstruction error. Training itself succeeded — the anomaly-detection
            signal did not. AUC 0.51 is a documented negative result, not an implementation failure.
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
                <span className="text-heading font-display" style={{ fontStyle: 'italic', color: 'var(--text-primary)' }}>latent, decoder.</span>
              </h2>
            </motion.div>
            <motion.div {...fadeUp(0.1)}>
              <p className="text-body" style={{ fontSize: '0.95rem', lineHeight: 1.75, marginBottom: '1.5rem' }}>
                Hover a layer to inspect it. The bottleneck forces the model to compress each X-ray into 128 numbers.
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
              <p className="text-subheading">params: 20,696,001</p>
            </motion.div>
          </div>

          {/* right: training config */}
          <div>
            <motion.div {...fadeUp(0)}>
              <p className="text-subheading" style={{ marginBottom: '0.5rem' }}>Training Setup</p>
            </motion.div>
            <motion.div {...fadeUp(0.06)}>
              <h2 style={{ marginBottom: '1.5rem' }}>
                <span className="text-heading" style={{ color: 'var(--text-primary)' }}>Fixed β, </span>
                <span className="text-heading font-display" style={{ fontStyle: 'italic', color: 'var(--text-primary)' }}>normal-only.</span>
              </h2>
            </motion.div>
            <motion.div {...fadeUp(0.1)}>
              <p className="text-body" style={{ fontSize: '0.95rem', lineHeight: 1.75, marginBottom: '1.5rem' }}>
                No KL annealing — β = 0.1 from epoch one. The model only ever sees healthy anatomy during training.
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
                Why fixed β instead of annealing?
              </p>
              <p className="text-body" style={{ fontSize: '0.875rem', lineHeight: 1.75 }}>
                Annealing β from 0 usually helps avoid posterior collapse early in training. Here, a fixed
                β = 0.1 already converges to a healthy, nonzero KL (~0.058) — the latent space stayed
                informative without the extra scheduling complexity.
              </p>
            </motion.div>
          </div>
        </div>
      </section>

      <div className="divider" />

      {/* reconstruction evolution */}
      <section className="section">
        <motion.div {...fadeUp(0)}>
          <p className="text-subheading" style={{ marginBottom: '0.5rem' }}>Reconstruction Quality</p>
        </motion.div>
        <motion.div {...fadeUp(0.06)}>
          <h2 style={{ marginBottom: '1.5rem' }}>
            <span className="text-heading" style={{ color: 'var(--text-primary)' }}>Sharper over </span>
            <span className="text-heading font-display" style={{ fontStyle: 'italic', color: 'var(--text-primary)' }}>training — to a point.</span>
          </h2>
        </motion.div>
        <motion.div {...fadeUp(0.1)}>
          <p className="text-body" style={{ maxWidth: '640px', fontSize: '0.95rem', lineHeight: 1.8, marginBottom: '2rem' }}>
            Top row: real X-rays. Bottom row: VAE reconstructions at that epoch. Global anatomy sharpens
            quickly, but fine detail — the kind that would carry localized pathology — never fully recovers.
          </p>
        </motion.div>

        {/* epoch tabs */}
        <motion.div {...fadeUp(0.14)} style={{ display: 'flex', gap: '0', marginBottom: '1.5rem', border: '1px solid var(--border)', borderRadius: '6px', overflow: 'hidden', width: 'fit-content' }}>
          {reconSamples.map((s, i) => (
            <button
              key={s.epoch}
              onClick={() => setActiveEpoch(i)}
              style={{
                padding: '0.6rem 1.4rem',
                background: activeEpoch === i ? 'linear-gradient(135deg, #3A015C, #7B2FBE)' : 'transparent',
                border: 'none',
                borderRight: i < reconSamples.length - 1 ? '1px solid var(--border)' : 'none',
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
              aspectRatio: '1571 / 397',
              borderRadius: '10px',
              overflow: 'hidden',
              border: '1px solid var(--border)',
              background: '#fff',
            }}>
              <Image
                src={current.path}
                alt={`VAE reconstructions — epoch ${current.epoch}`}
                fill
                sizes="1200px"
                style={{ objectFit: 'contain' }}
              />
            </div>
          </motion.div>
        </AnimatePresence>
      </section>

      <div className="divider" />

      {/* anomaly separation */}
      <section className="section">
        <div className="separation-grid" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '4rem', alignItems: 'start' }}>

          {/* left: histogram */}
          <div>
            <motion.div {...fadeUp(0)}>
              <p className="text-subheading" style={{ marginBottom: '0.5rem' }}>Anomaly Separation</p>
            </motion.div>
            <motion.div {...fadeUp(0.06)}>
              <h2 style={{ marginBottom: '1.5rem' }}>
                <span className="text-heading" style={{ color: 'var(--text-primary)' }}>Two curves, </span>
                <span className="text-heading font-display" style={{ fontStyle: 'italic', color: 'var(--text-primary)' }}>one distribution.</span>
              </h2>
            </motion.div>

            <motion.div {...fadeUp(0.1)} style={{ border: '1px solid var(--border)', borderRadius: '10px', overflow: 'hidden', background: '#fff' }}>
              <div style={{ position: 'relative', width: '100%', aspectRatio: '789 / 490' }}>
                <Image
                  src="/results/images/vae_anomaly_separation.png"
                  alt="VAE anomaly separation histogram"
                  fill
                  sizes="700px"
                  style={{ objectFit: 'contain' }}
                />
              </div>
            </motion.div>

            <motion.div {...fadeUp(0.16)} style={{ display: 'flex', gap: '2rem', marginTop: '1.5rem' }}>
              <div>
                <p style={{ fontFamily: 'Hanken Grotesk', fontSize: '1.6rem', fontWeight: 800, color: 'var(--accent-light)', letterSpacing: '-0.03em' }}>
                  <CountUp to={1.037} decimals={3} suffix="×" />
                </p>
                <p className="text-subheading">separation ratio</p>
              </div>
              <div>
                <p style={{ fontFamily: 'Hanken Grotesk', fontSize: '1.6rem', fontWeight: 800, color: 'var(--text-muted)', letterSpacing: '-0.03em' }}>2,068 / 601</p>
                <p className="text-subheading">normal / pneumonia (val)</p>
              </div>
            </motion.div>
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
          <Link href="/classifier" style={{ textDecoration: 'none' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--text-muted)', fontFamily: 'Hanken Grotesk', fontSize: '0.8rem', fontWeight: 600, letterSpacing: '0.06em' }}>
              <ArrowLeft size={13} weight="bold" /> classifier
            </div>
          </Link>
          <Link href="/unet" style={{ textDecoration: 'none' }}>
            <button style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', padding: '0.75rem 1.6rem', background: 'linear-gradient(135deg, #3A015C, #7B2FBE)', border: 'none', borderRadius: '3px', color: '#fff', fontFamily: 'Hanken Grotesk', fontSize: '0.8rem', fontWeight: 600, letterSpacing: '0.05em', boxShadow: '0 4px 24px rgba(123,47,190,0.3)' }}>
              next: u-net <ArrowRight size={13} weight="bold" />
            </button>
          </Link>
        </div>
      </section>

      <style jsx>{`
        .layer-row:hover { background: var(--surface) !important; }
        @media (max-width: 900px) {
          .arch-config-grid { grid-template-columns: 1fr !important; }
          .separation-grid { grid-template-columns: 1fr !important; }
          .metrics-grid { grid-template-columns: repeat(2, 1fr) !important; }
        }
      `}</style>
    </div>
  )
}