'use client'

import Link from 'next/link'
import { motion, useInView, animate } from 'framer-motion'
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

const layers = [
  { label: 'Input', detail: '224 × 224 × 1', type: 'io' },
  { label: 'ResNet18 Backbone', detail: '11.7M params · ImageNet pretrained', type: 'backbone' },
  { label: 'Dropout', detail: 'p = 0.3', type: 'reg' },
  { label: 'Linear 512 → 256', detail: 'ReLU activation', type: 'head' },
  { label: 'Dropout (2)', detail: 'p = 0.15', type: 'reg' },
  { label: 'Linear 256 → 2', detail: 'Softmax · Normal / Pneumonia', type: 'head' },
  { label: 'Output', detail: 'Binary classification', type: 'io' },
]

const layerColor: Record<string, string> = {
  io: 'var(--text-muted)',
  backbone: 'var(--accent-light)',
  reg: '#9B6DCC',
  head: '#C084FC',
}

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

const metrics = [
  { value: 85.08, label: 'Val Accuracy', sub: 'Best checkpoint', suffix: '%', decimals: 2 },
  { value: 0.8804, label: 'AUC-ROC', sub: 'Val split', decimals: 4 },
  { value: 11, label: 'Best Epoch', sub: 'Out of 20 scheduled', decimals: 0 },
  { value: 21347, label: 'Training Images', sub: '80% split', decimals: 0 },
]

// real confusion matrix data
const cm = { tn: 1928, fp: 139, fn: 259, tp: 342 }

// real ROC curve data
const rocFpr = [0.0,0.001,0.0029,0.0039,0.0053,0.0063,0.0077,0.0087,0.0116,0.0131,0.0145,0.0155,0.0174,0.0194,0.0208,0.0218,0.0232,0.0256,0.0276,0.0295,0.0314,0.0334,0.0358,0.0382,0.0397,0.0406,0.0431,0.044,0.0493,0.0503,0.0522,0.0537,0.0561,0.059,0.061,0.0624,0.0639,0.0668,0.0687,0.0711,0.0755,0.0793,0.0876,0.09,0.0948,0.0992,0.1026,0.104,0.1118,0.1147,0.1171,0.1205,0.1277,0.1316,0.1403,0.1413,0.1466,0.1505,0.1529,0.1548,0.1587,0.1601,0.164,0.1684,0.1708,0.1717,0.1882,0.1921,0.1969,0.1984,0.2013,0.2027,0.209,0.2129,0.2167,0.2211,0.2288,0.2313,0.2395,0.2429,0.2443,0.2511,0.2608,0.2733,0.2835,0.2874,0.2922,0.3038,0.3106,0.3203,0.3333,0.3362,0.3537,0.3662,0.3861,0.3885,0.3967,0.3986,0.4045,0.417,0.4243,0.4378,0.4412,0.4538,0.4615,0.4649,0.4683,0.4828,0.4896,0.5051,0.5317,0.5409,0.5622,0.5946,0.6715,0.8834,1.0]
const rocTpr = [0.0,0.02,0.0399,0.0865,0.1248,0.1597,0.1647,0.1847,0.2047,0.218,0.2429,0.2729,0.2845,0.2928,0.2995,0.3178,0.3278,0.3394,0.3428,0.371,0.376,0.3977,0.4143,0.4226,0.4326,0.4426,0.4493,0.4692,0.4792,0.4925,0.4975,0.5092,0.5141,0.5225,0.5374,0.5591,0.5624,0.5691,0.5724,0.5824,0.589,0.594,0.6023,0.614,0.6173,0.6223,0.6273,0.6389,0.6456,0.6572,0.6672,0.6772,0.6822,0.6889,0.6955,0.7038,0.7088,0.7155,0.7205,0.7271,0.7304,0.7371,0.7421,0.7537,0.7571,0.7654,0.7704,0.777,0.782,0.787,0.792,0.797,0.8053,0.8103,0.8153,0.8203,0.8236,0.8286,0.8319,0.8386,0.8453,0.8502,0.8536,0.8602,0.8636,0.8686,0.8735,0.8785,0.8819,0.8869,0.8902,0.8952,0.8985,0.9035,0.9068,0.9118,0.9151,0.9201,0.9235,0.9285,0.9318,0.9368,0.9401,0.9451,0.9484,0.9534,0.9567,0.9617,0.9667,0.9717,0.975,0.98,0.9834,0.9884,0.9917,0.9967,1.0]

function RocCurve() {
  const ref = useRef<SVGSVGElement>(null)
  const inView = useInView(ref as React.RefObject<Element>, { once: true, amount: 0.3 })
  const [drawn, setDrawn] = useState(false)

  useEffect(() => {
    if (inView) setTimeout(() => setDrawn(true), 0)
  }, [inView])

  const W = 400
  const H = 300
  const PAD = { top: 16, right: 16, bottom: 40, left: 44 }
  const pw = W - PAD.left - PAD.right
  const ph = H - PAD.top - PAD.bottom

  const pts = rocFpr.map((x, i) => ({
    sx: PAD.left + x * pw,
    sy: PAD.top + (1 - rocTpr[i]) * ph,
  }))

  const pathD = pts.map((p, i) => `${i === 0 ? 'M' : 'L'}${p.sx.toFixed(1)},${p.sy.toFixed(1)}`).join(' ')
  const fillD = pathD + ` L${PAD.left + pw},${PAD.top + ph} L${PAD.left},${PAD.top + ph} Z`

  // diagonal reference
  const diagD = `M${PAD.left},${PAD.top + ph} L${PAD.left + pw},${PAD.top}`

  const xTicks = [0, 0.25, 0.5, 0.75, 1.0]
  const yTicks = [0, 0.25, 0.5, 0.75, 1.0]

  return (
    <svg ref={ref} viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', overflow: 'visible' }}>
      <defs>
        <linearGradient id="roc-fill" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#7B2FBE" stopOpacity="0.35" />
          <stop offset="100%" stopColor="#7B2FBE" stopOpacity="0.04" />
        </linearGradient>
        <clipPath id="roc-clip">
          <rect x={PAD.left} y={PAD.top} width={pw} height={ph} />
        </clipPath>
      </defs>

      {/* grid lines */}
      {xTicks.map(t => (
        <line key={t} x1={PAD.left + t * pw} y1={PAD.top} x2={PAD.left + t * pw} y2={PAD.top + ph}
          stroke="var(--border)" strokeWidth="1" />
      ))}
      {yTicks.map(t => (
        <line key={t} x1={PAD.left} y1={PAD.top + (1 - t) * ph} x2={PAD.left + pw} y2={PAD.top + (1 - t) * ph}
          stroke="var(--border)" strokeWidth="1" />
      ))}

      {/* diagonal */}
      <path d={diagD} stroke="var(--text-muted)" strokeWidth="1" strokeDasharray="4 4" opacity="0.4" />

      {/* fill */}
      <path d={fillD} fill="url(#roc-fill)" clipPath="url(#roc-clip)" />

      {/* curve */}
      <path
        d={pathD}
        fill="none"
        stroke="var(--accent-light)"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
        clipPath="url(#roc-clip)"
        strokeDasharray={drawn ? 'none' : '2000'}
        strokeDashoffset={drawn ? '0' : '2000'}
        style={{ transition: drawn ? 'stroke-dashoffset 1.8s cubic-bezier(0.25,0.1,0.25,1)' : 'none' }}
      />

      {/* x axis labels */}
      {xTicks.map(t => (
        <text key={t} x={PAD.left + t * pw} y={PAD.top + ph + 16} textAnchor="middle"
          fontSize="9" fill="var(--text-muted)" fontFamily="Hanken Grotesk">
          {t}
        </text>
      ))}
      {/* y axis labels */}
      {yTicks.map(t => (
        <text key={t} x={PAD.left - 8} y={PAD.top + (1 - t) * ph + 3} textAnchor="end"
          fontSize="9" fill="var(--text-muted)" fontFamily="Hanken Grotesk">
          {t}
        </text>
      ))}

      {/* axis labels */}
      <text x={PAD.left + pw / 2} y={H - 4} textAnchor="middle"
        fontSize="8" fill="var(--text-muted)" fontFamily="Hanken Grotesk" letterSpacing="0.1em">
        FALSE POSITIVE RATE
      </text>
      <text x={12} y={PAD.top + ph / 2} textAnchor="middle"
        fontSize="8" fill="var(--text-muted)" fontFamily="Hanken Grotesk" letterSpacing="0.1em"
        transform={`rotate(-90, 12, ${PAD.top + ph / 2})`}>
        TRUE POSITIVE RATE
      </text>

      {/* AUC label */}
      <text x={PAD.left + pw - 4} y={PAD.top + 14} textAnchor="end"
        fontSize="10" fill="var(--accent-light)" fontFamily="Hanken Grotesk" fontWeight="700">
        AUC = 0.8804
      </text>
    </svg>
  )
}

export default function ClassifierPage() {
  const [activePhase, setActivePhase] = useState(0)
  const [activeLayer, setActiveLayer] = useState<number | null>(null)

  const total = cm.tn + cm.fp + cm.fn + cm.tp

  return (
    <div>

      {/* header */}
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
            progressive layer unfreezing reaches 85.08% validation accuracy and AUC 0.88
            on the RSNA dataset — without any domain-specific pretraining.
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

      {/* architecture + training strategy */}
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

            <div style={{ display: 'flex', gap: '0', border: '1px solid var(--border)', borderRadius: '6px', overflow: 'hidden', marginBottom: '1.5rem' }}>
              {strategy.map((s, i) => (
                <button key={s.phase} onClick={() => setActivePhase(i)} style={{
                  flex: 1, padding: '0.7rem 0.5rem',
                  background: activePhase === i ? 'linear-gradient(135deg, #3A015C, #7B2FBE)' : 'transparent',
                  border: 'none', borderRight: i < strategy.length - 1 ? '1px solid var(--border)' : 'none',
                  color: activePhase === i ? '#fff' : 'var(--text-muted)',
                  fontFamily: 'Hanken Grotesk', fontSize: '0.75rem', fontWeight: 600, letterSpacing: '0.05em',
                  transition: 'background 0.2s, color 0.2s',
                }}>Phase {s.phase}</button>
              ))}
            </div>

            <div style={{ position: 'relative', overflow: 'hidden' }}>
              {strategy.map((s, i) => (
                <motion.div key={s.phase} initial={false}
                  animate={{ opacity: activePhase === i ? 1 : 0, y: activePhase === i ? 0 : 8 }}
                  transition={{ duration: 0.3, ease: [0.25, 0.1, 0.25, 1] }}
                  style={{ position: activePhase === i ? 'relative' : 'absolute', top: 0, left: 0, right: 0, pointerEvents: activePhase === i ? 'auto' : 'none' }}
                >
                  <div style={{ padding: '1.75rem', border: `1px solid ${s.color}`, borderRadius: '8px', background: 'var(--surface)', position: 'relative', overflow: 'hidden' }}>
                    <div style={{ position: 'absolute', top: 0, right: 0, width: '80px', height: '80px', background: `radial-gradient(circle at top right, ${s.color}20, transparent 70%)`, pointerEvents: 'none' }} />
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1rem' }}>
                      <span style={{ fontFamily: 'Instrument Serif', fontSize: '2.5rem', fontWeight: 300, color: s.color, opacity: 0.4, lineHeight: 1 }}>{s.phase}</span>
                      <div>
                        <p style={{ fontFamily: 'Hanken Grotesk', fontSize: '1rem', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '0.15rem' }}>{s.label}</p>
                        <p className="text-subheading" style={{ color: s.color }}>{s.epochs}</p>
                      </div>
                    </div>
                    <p className="text-body" style={{ fontSize: '0.95rem', lineHeight: 1.8 }}>{s.detail}</p>
                  </div>
                </motion.div>
              ))}
            </div>

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
                <div key={row.label} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '0.7rem 1.1rem', borderBottom: i < arr.length - 1 ? '1px solid var(--border)' : 'none', background: i % 2 === 0 ? 'transparent' : 'var(--surface)' }}>
                  <p className="text-subheading">{row.label}</p>
                  <p style={{ fontFamily: 'Hanken Grotesk', fontSize: '0.82rem', fontWeight: 700, color: 'var(--text-primary)' }}>{row.value}</p>
                </div>
              ))}
            </motion.div>
          </div>
        </div>
      </section>

      <div className="divider" />

      {/* confusion matrix + ROC curve */}
      <section className="section">
        <div className="eval-grid" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '4rem', alignItems: 'start' }}>

          {/* left: confusion matrix */}
          <div>
            <motion.div {...fadeUp(0)}>
              <p className="text-subheading" style={{ marginBottom: '0.5rem' }}>Evaluation</p>
            </motion.div>
            <motion.div {...fadeUp(0.06)}>
              <h2 style={{ marginBottom: '1.5rem' }}>
                <span className="text-heading" style={{ color: 'var(--text-primary)' }}>Confusion </span>
                <span className="text-heading font-display" style={{ fontStyle: 'italic', color: 'var(--text-primary)' }}>matrix.</span>
              </h2>
            </motion.div>

            {/* 2x2 confusion matrix grid */}
            <motion.div {...fadeUp(0.1)}>
              {/* col headers */}
              <div style={{ display: 'grid', gridTemplateColumns: '80px 1fr 1fr', gap: '4px', marginBottom: '4px' }}>
                <div />
                <p className="text-subheading" style={{ textAlign: 'center' }}>Pred: Normal</p>
                <p className="text-subheading" style={{ textAlign: 'center' }}>Pred: Pneumonia</p>
              </div>

              {/* row 1: actual normal */}
              <div style={{ display: 'grid', gridTemplateColumns: '80px 1fr 1fr', gap: '4px', marginBottom: '4px' }}>
                <div style={{ display: 'flex', alignItems: 'center' }}>
                  <p className="text-subheading" style={{ writingMode: 'vertical-rl', transform: 'rotate(180deg)', textAlign: 'center' }}>Act: Normal</p>
                </div>
                {/* TN */}
                <motion.div
                  initial={{ opacity: 0, scale: 0.9 }}
                  whileInView={{ opacity: 1, scale: 1 }}
                  viewport={{ once: true, amount: 0.1 }}
                  transition={{ duration: 0.5, delay: 0.1 }}
                  style={{ padding: '1.5rem', borderRadius: '6px', background: 'rgba(123,47,190,0.15)', border: '1px solid rgba(123,47,190,0.3)', textAlign: 'center' }}
                >
                  <p style={{ fontFamily: 'Hanken Grotesk', fontSize: '1.8rem', fontWeight: 800, color: 'var(--accent-light)', letterSpacing: '-0.03em', lineHeight: 1, marginBottom: '0.25rem' }}>
                    {cm.tn.toLocaleString()}
                  </p>
                  <p className="text-subheading" style={{ color: 'var(--accent-light)' }}>TN</p>
                  <p className="text-subheading" style={{ marginTop: '0.2rem' }}>{((cm.tn / total) * 100).toFixed(1)}%</p>
                </motion.div>
                {/* FP */}
                <motion.div
                  initial={{ opacity: 0, scale: 0.9 }}
                  whileInView={{ opacity: 1, scale: 1 }}
                  viewport={{ once: true, amount: 0.1 }}
                  transition={{ duration: 0.5, delay: 0.18 }}
                  style={{ padding: '1.5rem', borderRadius: '6px', background: 'rgba(255,237,223,0.04)', border: '1px solid var(--border)', textAlign: 'center' }}
                >
                  <p style={{ fontFamily: 'Hanken Grotesk', fontSize: '1.8rem', fontWeight: 800, color: 'var(--text-muted)', letterSpacing: '-0.03em', lineHeight: 1, marginBottom: '0.25rem' }}>
                    {cm.fp.toLocaleString()}
                  </p>
                  <p className="text-subheading">FP</p>
                  <p className="text-subheading" style={{ marginTop: '0.2rem' }}>{((cm.fp / total) * 100).toFixed(1)}%</p>
                </motion.div>
              </div>

              {/* row 2: actual pneumonia */}
              <div style={{ display: 'grid', gridTemplateColumns: '80px 1fr 1fr', gap: '4px' }}>
                <div style={{ display: 'flex', alignItems: 'center' }}>
                  <p className="text-subheading" style={{ writingMode: 'vertical-rl', transform: 'rotate(180deg)', textAlign: 'center' }}>Act: Pneumonia</p>
                </div>
                {/* FN */}
                <motion.div
                  initial={{ opacity: 0, scale: 0.9 }}
                  whileInView={{ opacity: 1, scale: 1 }}
                  viewport={{ once: true, amount: 0.1 }}
                  transition={{ duration: 0.5, delay: 0.26 }}
                  style={{ padding: '1.5rem', borderRadius: '6px', background: 'rgba(255,237,223,0.04)', border: '1px solid var(--border)', textAlign: 'center' }}
                >
                  <p style={{ fontFamily: 'Hanken Grotesk', fontSize: '1.8rem', fontWeight: 800, color: 'var(--text-muted)', letterSpacing: '-0.03em', lineHeight: 1, marginBottom: '0.25rem' }}>
                    {cm.fn.toLocaleString()}
                  </p>
                  <p className="text-subheading">FN</p>
                  <p className="text-subheading" style={{ marginTop: '0.2rem' }}>{((cm.fn / total) * 100).toFixed(1)}%</p>
                </motion.div>
                {/* TP */}
                <motion.div
                  initial={{ opacity: 0, scale: 0.9 }}
                  whileInView={{ opacity: 1, scale: 1 }}
                  viewport={{ once: true, amount: 0.1 }}
                  transition={{ duration: 0.5, delay: 0.34 }}
                  style={{ padding: '1.5rem', borderRadius: '6px', background: 'rgba(123,47,190,0.15)', border: '1px solid rgba(123,47,190,0.3)', textAlign: 'center' }}
                >
                  <p style={{ fontFamily: 'Hanken Grotesk', fontSize: '1.8rem', fontWeight: 800, color: 'var(--accent-light)', letterSpacing: '-0.03em', lineHeight: 1, marginBottom: '0.25rem' }}>
                    {cm.tp.toLocaleString()}
                  </p>
                  <p className="text-subheading" style={{ color: 'var(--accent-light)' }}>TP</p>
                  <p className="text-subheading" style={{ marginTop: '0.2rem' }}>{((cm.tp / total) * 100).toFixed(1)}%</p>
                </motion.div>
              </div>
            </motion.div>

            {/* per-class metrics */}
            <motion.div {...fadeUp(0.3)} style={{ marginTop: '1.5rem', border: '1px solid var(--border)', borderRadius: '8px', overflow: 'hidden' }}>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr repeat(3, 80px)', padding: '0.6rem 1rem', background: 'var(--surface)', borderBottom: '1px solid var(--border)', gap: '0.5rem' }}>
                <p className="text-subheading">Class</p>
                <p className="text-subheading" style={{ textAlign: 'right' }}>Prec.</p>
                <p className="text-subheading" style={{ textAlign: 'right' }}>Recall</p>
                <p className="text-subheading" style={{ textAlign: 'right' }}>F1</p>
              </div>
              {[
                { label: 'Normal', color: 'var(--accent-light)', precision: 0.8816, recall: 0.9328, f1: 0.9064 },
                { label: 'Pneumonia', color: '#9B6DCC', precision: 0.711, recall: 0.5691, f1: 0.6322 },
              ].map((row, i) => (
                <div key={row.label} style={{ display: 'grid', gridTemplateColumns: '1fr repeat(3, 80px)', padding: '0.8rem 1rem', borderBottom: i === 0 ? '1px solid var(--border)' : 'none', gap: '0.5rem', alignItems: 'center' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <div style={{ width: '6px', height: '6px', borderRadius: '50%', background: row.color, flexShrink: 0 }} />
                    <p style={{ fontFamily: 'Hanken Grotesk', fontSize: '0.875rem', fontWeight: 600, color: 'var(--text-primary)' }}>{row.label}</p>
                  </div>
                  <p style={{ fontFamily: 'Hanken Grotesk', fontSize: '0.875rem', fontWeight: 700, color: 'var(--text-primary)', textAlign: 'right' }}>{row.precision.toFixed(2)}</p>
                  <p style={{ fontFamily: 'Hanken Grotesk', fontSize: '0.875rem', fontWeight: 700, color: row.recall < 0.65 ? '#9B6DCC' : 'var(--text-primary)', textAlign: 'right' }}>{row.recall.toFixed(2)}</p>
                  <p style={{ fontFamily: 'Hanken Grotesk', fontSize: '0.875rem', fontWeight: 700, color: 'var(--text-primary)', textAlign: 'right' }}>{row.f1.toFixed(2)}</p>
                </div>
              ))}
            </motion.div>
          </div>

          {/* right: ROC curve */}
          <div>
            <motion.div {...fadeUp(0)}>
              <p className="text-subheading" style={{ marginBottom: '0.5rem' }}>ROC Curve</p>
            </motion.div>
            <motion.div {...fadeUp(0.06)}>
              <h2 style={{ marginBottom: '1.5rem' }}>
                <span className="text-heading" style={{ color: 'var(--text-primary)' }}>AUC </span>
                <span className="text-heading font-display" style={{ fontStyle: 'italic', color: 'var(--text-primary)' }}>0.8804.</span>
              </h2>
            </motion.div>

            <motion.div {...fadeUp(0.1)} style={{ border: '1px solid var(--border)', borderRadius: '8px', padding: '1.25rem', background: 'var(--surface)' }}>
              <RocCurve />
            </motion.div>

            <motion.div {...fadeUp(0.2)} style={{ marginTop: '1.5rem', display: 'flex', flexDirection: 'column', gap: '1px', border: '1px solid var(--border)', borderRadius: '8px', overflow: 'hidden' }}>
              {[
                { heading: 'Recall gap is the key clinical concern.', body: 'Normal recall 0.93 vs pneumonia recall 0.57 — the model misses 43% of pneumonia cases (FN=259). In clinical deployment this matters more than accuracy: a false negative means a missed diagnosis.' },
                { heading: 'AUC 0.88 shows real discriminative ability.', body: 'At every decision threshold, the model correctly ranks a pneumonia case above a normal case 88% of the time. This is robust performance for a zero-domain-pretraining baseline.' },
                { heading: 'Class imbalance explains the recall gap.', body: '77.5% of training data is normal. The model implicitly learns a prior toward normal predictions. Weighted loss or oversampling pneumonia cases would close this gap.' },
              ].map((item, i, arr) => (
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
            </motion.div>
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
            <button style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', padding: '0.75rem 1.6rem', background: 'linear-gradient(135deg, #3A015C, #7B2FBE)', border: 'none', borderRadius: '3px', color: '#fff', fontFamily: 'Hanken Grotesk', fontSize: '0.8rem', fontWeight: 600, letterSpacing: '0.05em', boxShadow: '0 4px 24px rgba(123,47,190,0.3)' }}>
              next: vae <ArrowRight size={13} weight="bold" />
            </button>
          </Link>
        </div>
      </section>

      <style jsx>{`
        .layer-row:hover { background: var(--surface) !important; }
        @media (max-width: 900px) {
          .arch-strategy-grid { grid-template-columns: 1fr !important; }
          .eval-grid { grid-template-columns: 1fr !important; }
          .result-grid { grid-template-columns: 1fr !important; }
          .metrics-grid { grid-template-columns: repeat(2, 1fr) !important; }
        }
      `}</style>
    </div>
  )
}