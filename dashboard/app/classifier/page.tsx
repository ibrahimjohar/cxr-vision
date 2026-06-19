'use client'

import Link from 'next/link'
import { motion, useInView, animate } from 'framer-motion'
import { useRef, useState, useEffect } from 'react'
import { ArrowLeft, ArrowRight, CheckCircle, Circle } from '@phosphor-icons/react'

// counter animation — same pattern as home page, known working
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

// architecture layers
const layers = [
  { label: 'Input', detail: '224 × 224 × 1', type: 'io' },
  { label: 'ResNet18 Backbone', detail: '11.7M params · ImageNet pretrained', type: 'backbone' },
  { label: 'Dropout', detail: 'p = 0.3', type: 'reg' },
  { label: 'Linear 512 → 256', detail: 'ReLU activation', type: 'head' },
  { label: 'Dropout', detail: 'p = 0.3', type: 'reg' },
  { label: 'Linear 256 → 2', detail: 'Softmax · Normal / Pneumonia', type: 'head' },
  { label: 'Output', detail: 'Binary classification', type: 'io' },
]

const layerColor: Record<string, string> = {
  io: 'var(--text-muted)',
  backbone: 'var(--accent-light)',
  reg: '#9B6DCC',
  head: '#C084FC',
}

// training strategy steps
const strategy = [
  {
    phase: '01',
    label: 'Frozen backbone',
    epochs: 'Epochs 1–5',
    detail: 'ResNet18 weights locked. Only the classification head trains. Prevents destroying ImageNet features before the head stabilises.',
    color: 'var(--accent-light)',
  },
  {
    phase: '02',
    label: 'Progressive unfreeze',
    epochs: 'Epoch 6+',
    detail: 'Backbone unfrozen at 0.1× learning rate. Fine-tunes domain-specific features while the head is already calibrated. Reduces risk of catastrophic forgetting.',
    color: '#9B6DCC',
  },
  {
    phase: '03',
    label: 'Early stopping',
    epochs: 'Stopped epoch 16',
    detail: 'Best val accuracy epoch 11. Patience of 5 epochs. Prevents overfitting to the 21,347-sample training set.',
    color: '#C084FC',
  },
]

// key metrics
const metrics = [
  { value: 85.08, label: 'Val Accuracy', sub: 'Best checkpoint', suffix: '%', decimals: 2 },
  { value: 11, label: 'Best Epoch', sub: 'Out of 20 scheduled', decimals: 0 },
  { value: 16, label: 'Stopped Epoch', sub: 'Patience = 5', decimals: 0 },
  { value: 21347, label: 'Training Images', sub: '80% split', decimals: 0 },
]

export default function ClassifierPage() {
  const [activePhase, setActivePhase] = useState(0)
  const [activeLayer, setActiveLayer] = useState<number | null>(null)

  return (
    <div>

      {/* page header */}
      <section className="section" style={{ paddingBottom: '2rem' }}>
        <motion.div {...fadeUp(0)}>
          <p className="text-subheading" style={{ marginBottom: '0.75rem' }}>Stage 02 · Classification</p>
        </motion.div>
        <motion.div {...fadeUp(0.06)}>
          <h1 style={{ marginBottom: '1.5rem', lineHeight: 0.95 }}>
            <span className="text-display" style={{ color: 'var(--text-primary)', display: 'block', fontSize: 'clamp(2.5rem, 6vw, 5rem)' }}>
              ResNet18
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
              Classifier.
            </span>
          </h1>
        </motion.div>
        <motion.div {...fadeUp(0.12)}>
          <p className="text-body" style={{ maxWidth: '560px', fontSize: '1.05rem', lineHeight: 1.8 }}>
            Transfer learning from ImageNet to chest radiographs. ResNet18 with
            progressive layer unfreezing reaches 85.08% validation accuracy on
            the heavily imbalanced RSNA dataset — without any domain-specific pretraining.
          </p>
        </motion.div>
      </section>

      <div className="divider" />

      {/* key metrics strip */}
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
                  <CountUp to={m.value} decimals={m.decimals} suffix={m.suffix} />
                </p>
                <p style={{ fontFamily: 'Hanken Grotesk', fontSize: '0.9rem', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '0.2rem' }}>{m.label}</p>
                <p className="text-subheading">{m.sub}</p>
              </div>
            </motion.div>
          ))}
        </div>
      </section>

      <div className="divider" />

      {/* architecture + training strategy — two equal columns */}
      <section className="section">
        <div className="arch-strategy-grid" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '4rem', alignItems: 'start' }}>

          {/* left: architecture */}
          <div>
            <motion.div {...fadeUp(0)}>
              <p className="text-subheading" style={{ marginBottom: '0.5rem' }}>Architecture</p>
            </motion.div>
            <motion.div {...fadeUp(0.06)}>
              <h2 style={{ marginBottom: '1.5rem' }}>
                <span className="text-heading" style={{ color: 'var(--text-primary)' }}>Model </span>
                <span className="text-heading font-display" style={{ fontStyle: 'italic', color: 'var(--text-primary)' }}>layers.</span>
              </h2>
            </motion.div>
            <motion.div {...fadeUp(0.1)}>
              <p className="text-body" style={{ fontSize: '0.95rem', lineHeight: 1.75, marginBottom: '1.5rem' }}>
                Hover a layer to inspect it. The backbone is locked during early training then progressively unfrozen.
              </p>
            </motion.div>

            {/* layer diagram */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0', border: '1px solid var(--border)', borderRadius: '8px', overflow: 'hidden' }}>
              {layers.map((layer, i) => (
                <motion.div
                  key={layer.label}
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
                      display: 'flex',
                      alignItems: 'center',
                      gap: '1rem',
                      padding: '0.9rem 1.25rem',
                      borderBottom: i < layers.length - 1 ? '1px solid var(--border)' : 'none',
                      background: activeLayer === i ? 'var(--surface)' : 'transparent',
                      transition: 'background 0.2s',
                      cursor: 'default',
                    }}
                  >
                    {/* color indicator */}
                    <div style={{
                      width: '3px',
                      height: '2rem',
                      borderRadius: '2px',
                      background: layerColor[layer.type],
                      flexShrink: 0,
                      transition: 'opacity 0.2s',
                      opacity: activeLayer === i ? 1 : 0.5,
                    }} />
                    <div style={{ flex: 1, minWidth: 0 }}>
                      <p style={{ fontFamily: 'Hanken Grotesk', fontSize: '0.875rem', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '0.15rem' }}>
                        {layer.label}
                      </p>
                      <p className="text-subheading" style={{ fontSize: '0.62rem' }}>{layer.detail}</p>
                    </div>
                    {/* type badge */}
                    <span style={{
                      fontFamily: 'Hanken Grotesk',
                      fontSize: '0.6rem',
                      fontWeight: 700,
                      letterSpacing: '0.1em',
                      textTransform: 'uppercase',
                      color: layerColor[layer.type],
                      opacity: 0.8,
                      flexShrink: 0,
                    }}>
                      {layer.type}
                    </span>
                  </div>
                </motion.div>
              ))}
            </div>

            {/* flow indicator */}
            <motion.div {...fadeUp(0.3)} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginTop: '1rem' }}>
              <div style={{ width: '1px', height: '24px', background: 'var(--gradient-accent)' }} />
              <p className="text-subheading">data flows top to bottom</p>
            </motion.div>
          </div>

          {/* right: training strategy */}
          <div>
            <motion.div {...fadeUp(0)}>
              <p className="text-subheading" style={{ marginBottom: '0.5rem' }}>Training Strategy</p>
            </motion.div>
            <motion.div {...fadeUp(0.06)}>
              <h2 style={{ marginBottom: '1.5rem' }}>
                <span className="text-heading" style={{ color: 'var(--text-primary)' }}>How it </span>
                <span className="text-heading font-display" style={{ fontStyle: 'italic', color: 'var(--text-primary)' }}>learned.</span>
              </h2>
            </motion.div>
            <motion.div {...fadeUp(0.1)}>
              <p className="text-body" style={{ fontSize: '0.95rem', lineHeight: 1.75, marginBottom: '1.5rem' }}>
                Click a phase to read the reasoning behind each training decision.
              </p>
            </motion.div>

            {/* phase selector */}
            <div style={{ display: 'flex', gap: '0', border: '1px solid var(--border)', borderRadius: '6px', overflow: 'hidden', marginBottom: '1.5rem' }}>
              {strategy.map((s, i) => (
                <button
                  key={s.phase}
                  onClick={() => setActivePhase(i)}
                  style={{
                    flex: 1,
                    padding: '0.7rem 0.5rem',
                    background: activePhase === i ? 'linear-gradient(135deg, #3A015C, #7B2FBE)' : 'transparent',
                    border: 'none',
                    borderRight: i < strategy.length - 1 ? '1px solid var(--border)' : 'none',
                    color: activePhase === i ? '#fff' : 'var(--text-muted)',
                    fontFamily: 'Hanken Grotesk',
                    fontSize: '0.75rem',
                    fontWeight: 600,
                    letterSpacing: '0.05em',
                    transition: 'background 0.2s, color 0.2s',
                  }}
                >
                  Phase {s.phase}
                </button>
              ))}
            </div>

            {/* active phase card */}
            <div style={{ position: 'relative', overflow: 'hidden' }}>
              {strategy.map((s, i) => (
                <motion.div
                  key={s.phase}
                  initial={false}
                  animate={{ opacity: activePhase === i ? 1 : 0, y: activePhase === i ? 0 : 8 }}
                  transition={{ duration: 0.3, ease: [0.25, 0.1, 0.25, 1] }}
                  style={{
                    position: activePhase === i ? 'relative' : 'absolute',
                    top: 0, left: 0, right: 0,
                    pointerEvents: activePhase === i ? 'auto' : 'none',
                  }}
                >
                  <div style={{
                    padding: '1.75rem',
                    border: `1px solid ${s.color}`,
                    borderRadius: '8px',
                    background: 'var(--surface)',
                    position: 'relative',
                    overflow: 'hidden',
                  }}>
                    {/* accent corner */}
                    <div style={{
                      position: 'absolute',
                      top: 0, right: 0,
                      width: '80px', height: '80px',
                      background: `radial-gradient(circle at top right, ${s.color}20, transparent 70%)`,
                      pointerEvents: 'none',
                    }} />
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1rem' }}>
                      <span style={{
                        fontFamily: 'Instrument Serif',
                        fontSize: '2.5rem',
                        fontWeight: 300,
                        color: s.color,
                        opacity: 0.4,
                        lineHeight: 1,
                      }}>{s.phase}</span>
                      <div>
                        <p style={{ fontFamily: 'Hanken Grotesk', fontSize: '1rem', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '0.15rem' }}>
                          {s.label}
                        </p>
                        <p className="text-subheading" style={{ color: s.color }}>{s.epochs}</p>
                      </div>
                    </div>
                    <p className="text-body" style={{ fontSize: '0.95rem', lineHeight: 1.8 }}>
                      {s.detail}
                    </p>
                  </div>
                </motion.div>
              ))}
            </div>

            {/* training config table */}
            <motion.div {...fadeUp(0.25)} style={{ marginTop: '1.5rem', border: '1px solid var(--border)', borderRadius: '8px', overflow: 'hidden' }}>
              {[
                { label: 'Backbone', value: 'ResNet18' },
                { label: 'Pretrained', value: 'ImageNet' },
                { label: 'Learning rate', value: '1e-4 (head) · 1e-5 (backbone)' },
                { label: 'Scheduler', value: 'Cosine annealing' },
                { label: 'Dropout', value: '0.3' },
                { label: 'Batch size', value: '16' },
                { label: 'Weight decay', value: '1e-4' },
              ].map((row, i, arr) => (
                <div key={row.label} style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  padding: '0.7rem 1.1rem',
                  borderBottom: i < arr.length - 1 ? '1px solid var(--border)' : 'none',
                  background: i % 2 === 0 ? 'transparent' : 'var(--surface)',
                }}>
                  <p className="text-subheading">{row.label}</p>
                  <p style={{ fontFamily: 'Hanken Grotesk', fontSize: '0.82rem', fontWeight: 700, color: 'var(--text-primary)' }}>{row.value}</p>
                </div>
              ))}
            </motion.div>
          </div>
        </div>
      </section>

      <div className="divider" />

      {/* result + honest framing — two equal columns */}
      <section className="section">
        <div className="result-grid" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '4rem', alignItems: 'start' }}>

          {/* left: result */}
          <div>
            <motion.div {...fadeUp(0)}>
              <p className="text-subheading" style={{ marginBottom: '0.5rem' }}>Result</p>
            </motion.div>
            <motion.div {...fadeUp(0.06)}>
              <h2 style={{ marginBottom: '1.5rem' }}>
                <span className="text-heading" style={{ color: 'var(--text-primary)' }}>85.08% </span>
                <span className="text-heading font-display" style={{ fontStyle: 'italic', color: 'var(--text-primary)' }}>accuracy.</span>
              </h2>
            </motion.div>

            {/* large animated accuracy bar */}
            <motion.div {...fadeUp(0.1)}>
              <div style={{ marginBottom: '0.6rem', display: 'flex', justifyContent: 'space-between' }}>
                <p className="text-subheading">validation accuracy</p>
                <p style={{ fontFamily: 'Hanken Grotesk', fontSize: '0.875rem', fontWeight: 800, color: 'var(--accent-light)' }}>85.08%</p>
              </div>
              <div style={{ height: '8px', background: 'var(--border)', borderRadius: '4px', overflow: 'hidden', marginBottom: '2rem' }}>
                <motion.div
                  initial={{ width: 0 }}
                  whileInView={{ width: '85.08%' }}
                  viewport={{ once: true, amount: 0.1 }}
                  transition={{ duration: 1.4, ease: [0.25, 0.1, 0.25, 1] as [number,number,number,number] }}
                  style={{ height: '100%', background: 'var(--gradient-accent)', borderRadius: '4px' }}
                />
              </div>

              {/* class breakdown bars */}
              {[
                { label: 'Normal (majority class)', pct: 77.5, color: 'var(--accent-light)' },
                { label: 'Pneumonia (minority class)', pct: 22.5, color: '#9B6DCC' },
              ].map((bar) => (
                <div key={bar.label} style={{ marginBottom: '1rem' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.35rem' }}>
                    <p className="text-subheading">{bar.label}</p>
                    <p className="text-subheading">{bar.pct}%</p>
                  </div>
                  <div style={{ height: '4px', background: 'var(--border)', borderRadius: '2px', overflow: 'hidden' }}>
                    <motion.div
                      initial={{ width: 0 }}
                      whileInView={{ width: `${bar.pct}%` }}
                      viewport={{ once: true, amount: 0.1 }}
                      transition={{ duration: 1.2, ease: [0.25, 0.1, 0.25, 1] as [number,number,number,number] }}
                      style={{ height: '100%', background: bar.color, borderRadius: '2px' }}
                    />
                  </div>
                </div>
              ))}
            </motion.div>

            {/* checkpoint info */}
            <motion.div {...fadeUp(0.18)}>
              <div style={{ padding: '1.25rem', border: '1px solid var(--border)', borderRadius: '8px', background: 'var(--surface)', marginTop: '1rem' }}>
                <p className="text-subheading" style={{ marginBottom: '0.6rem' }}>Checkpoint saved</p>
                <p style={{ fontFamily: 'monospace', fontSize: '0.8rem', color: 'var(--accent-light)', letterSpacing: '0.04em' }}>
                  outputs/checkpoints/best_resnet18.pth
                </p>
                <p className="text-body" style={{ fontSize: '0.85rem', marginTop: '0.5rem' }}>
                  Epoch 11 — best validation accuracy. Used for all downstream inference.
                </p>
              </div>
            </motion.div>
          </div>

          {/* right: honest framing */}
          <div>
            <motion.div {...fadeUp(0)}>
              <p className="text-subheading" style={{ marginBottom: '0.5rem' }}>Honest Framing</p>
            </motion.div>
            <motion.div {...fadeUp(0.06)}>
              <h2 style={{ marginBottom: '1.5rem' }}>
                <span className="text-heading" style={{ color: 'var(--text-primary)' }}>What this </span>
                <span className="text-heading font-display" style={{ fontStyle: 'italic', color: 'var(--text-primary)' }}>actually means.</span>
              </h2>
            </motion.div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '1px', border: '1px solid var(--border)', borderRadius: '8px', overflow: 'hidden' }}>
              {[
                {
                  icon: 'check',
                  heading: '85% on an imbalanced dataset is real.',
                  body: 'With 77.5% normal samples, a model that always predicts "normal" would score 77.5% accuracy. At 85.08%, this model is genuinely learning pneumonia features, not just exploiting class imbalance.',
                },
                {
                  icon: 'check',
                  heading: 'No confusion matrix available.',
                  body: 'Confusion matrix was not exported at evaluation time. The honest metric is val accuracy at epoch 11. A future run should log per-class precision, recall, and AUC for a complete picture.',
                },
                {
                  icon: 'check',
                  heading: 'Transfer learning is the deciding factor.',
                  body: 'Training from scratch on 21,347 images would likely converge much lower. ImageNet weights provide edge detectors and texture features that directly apply to radiograph structure, even across domains.',
                },
                {
                  icon: 'check',
                  heading: 'Cosine LR decay prevents plateauing.',
                  body: 'The learning rate follows a cosine curve from 1e-4 to near-zero over 20 epochs. This smooth decay lets the model settle into sharp minima rather than bouncing around them.',
                },
              ].map((item, i, arr) => (
                <motion.div
                  key={item.heading}
                  initial={{ opacity: 0, y: 10 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true, amount: 0.1 }}
                  transition={{ duration: 0.45, delay: i * 0.07 }}
                >
                  <div style={{
                    padding: '1.25rem',
                    borderBottom: i < arr.length - 1 ? '1px solid var(--border)' : 'none',
                    background: 'transparent',
                    display: 'flex',
                    gap: '1rem',
                    alignItems: 'flex-start',
                  }}>
                    <div style={{ paddingTop: '2px', flexShrink: 0 }}>
                      <CheckCircle size={16} style={{ color: 'var(--accent-light)', opacity: 0.8 }} weight="fill" />
                    </div>
                    <div>
                      <p style={{ fontFamily: 'Hanken Grotesk', fontSize: '0.875rem', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '0.4rem', lineHeight: 1.4 }}>
                        {item.heading}
                      </p>
                      <p className="text-body" style={{ fontSize: '0.85rem', lineHeight: 1.7 }}>
                        {item.body}
                      </p>
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
          <Link href="/preprocessing" style={{ textDecoration: 'none' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--text-muted)', fontFamily: 'Hanken Grotesk', fontSize: '0.8rem', fontWeight: 600, letterSpacing: '0.06em' }}>
              <ArrowLeft size={13} weight="bold" /> preprocessing
            </div>
          </Link>
          <Link href="/vae" style={{ textDecoration: 'none' }}>
            <button style={{
              display: 'flex', alignItems: 'center', gap: '0.5rem',
              padding: '0.75rem 1.6rem',
              background: 'linear-gradient(135deg, #3A015C, #7B2FBE)',
              border: 'none', borderRadius: '3px', color: '#fff',
              fontFamily: 'Hanken Grotesk', fontSize: '0.8rem', fontWeight: 600,
              letterSpacing: '0.05em',
              boxShadow: '0 4px 24px rgba(123,47,190,0.3)',
            }}>
              next: vae <ArrowRight size={13} weight="bold" />
            </button>
          </Link>
        </div>
      </section>

      <style jsx>{`
        .layer-row:hover {
          background: var(--surface) !important;
        }
        @media (max-width: 900px) {
          .arch-strategy-grid {
            grid-template-columns: 1fr !important;
          }
          .result-grid {
            grid-template-columns: 1fr !important;
          }
          .metrics-grid {
            grid-template-columns: repeat(2, 1fr) !important;
          }
        }
        @media (max-width: 640px) {
          .metrics-grid {
            grid-template-columns: repeat(2, 1fr) !important;
          }
        }
      `}</style>
    </div>
  )
}