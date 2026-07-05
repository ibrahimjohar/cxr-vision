'use client'

import Image from 'next/image'
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

const metrics = [
  { value: 0.5093, label: 'CLIP AUC (plain)', sub: 'ViT-B/32 — zero domain pretraining', decimals: 4 },
  { value: 0.8386, label: 'BiomedCLIP AUC', sub: 'PubMedBERT-256 ViT-B/16, medical prompt', decimals: 4 },
  { value: 0.3293, label: 'Δ AUC', sub: 'Isolates domain pretraining as the lever', decimals: 4 },
  { value: 0.44, label: 'Clinical Prompt AUC', sub: '77.4% accuracy on the same run — see below', decimals: 2 },
]

const mechanism = [
  { label: 'Image Encoder', detail: 'ViT-B/32 (CLIP) or ViT-B/16 (BiomedCLIP)', type: 'backbone' },
  { label: 'Text Encoder', detail: 'Encodes each class prompt, e.g. "a chest x-ray showing pneumonia"', type: 'backbone' },
  { label: 'Cosine Similarity', detail: 'Image embedding compared against every prompt embedding', type: 'reg' },
  { label: 'Zero-Shot Prediction', detail: 'Highest-similarity prompt wins — no classifier head, no fine-tuning', type: 'head' },
]

const layerColor: Record<string, string> = {
  io: 'var(--text-muted)',
  backbone: 'var(--accent-light)',
  reg: '#9B6DCC',
  head: '#C084FC',
}

const prompts = [
  {
    id: 'plain',
    label: 'Plain',
    tag: '00',
    color: 'rgba(255,237,223,0.5)',
    example: '"a photo of a normal chest x-ray" / "a photo of a chest x-ray with pneumonia"',
    auc: 0.5093,
    accuracy: 0.2252,
    note: 'Generic photo-caption phrasing. Chance-level AUC — CLIP has no notion of what these words mean radiologically.',
  },
  {
    id: 'clinical',
    label: 'Clinical',
    tag: '01',
    color: 'var(--accent-light)',
    example: '"no acute cardiopulmonary abnormality" / "findings consistent with pneumonia"',
    auc: 0.44,
    accuracy: 0.7744,
    note: 'Sounds like a radiology report. Accuracy jumps to 77% — but AUC drops below chance. This is the trap.',
  },
  {
    id: 'descriptive',
    label: 'Descriptive',
    tag: '02',
    color: '#9B6DCC',
    example: '"clear lung fields, no infiltrate" / "patchy opacity in the lung parenchyma"',
    auc: 0.3854,
    accuracy: 0.2349,
    note: 'More visually descriptive wording. Worst AUC of the three — wording alone can\'t substitute for domain pretraining.',
  },
]

const verdicts = [
  {
    heading: 'Domain pretraining is the deciding factor.',
    body: 'A +0.33 AUC gap separates generic CLIP from BiomedCLIP — and both are zero-shot, with zero RSNA training. The only difference is what each model was pretrained on.',
  },
  {
    heading: 'The clinical prompt is a trap, not a win.',
    body: '77.4% accuracy sounds strong until AUC reveals it: the model predicted "normal" for nearly every image and still scored well, because 77.5% of the dataset is normal. AUC exposes the majority-class default; accuracy hides it.',
  },
  {
    heading: 'Prompt wording can\'t fix a missing domain prior.',
    body: 'Plain and descriptive prompts score similarly poorly (AUC 0.51 and 0.39) on the same CLIP backbone. No amount of prompt engineering recovers what BiomedCLIP gets for free from its pretraining data.',
  },
]

export default function ClipPage() {
  const [activePrompt, setActivePrompt] = useState(1) // default: clinical
  const [activeStep, setActiveStep] = useState<number | null>(null)
  const current = prompts[activePrompt]

  return (
    <div>

      {/* header */}
      <section className="section" style={{ paddingBottom: '2rem' }}>
        <motion.div {...fadeUp(0)}>
          <p className="text-subheading" style={{ marginBottom: '0.75rem' }}>Stage 05 · Zero-Shot Multimodal</p>
        </motion.div>
        <motion.div {...fadeUp(0.06)}>
          <h1 style={{ marginBottom: '1.5rem', lineHeight: 0.95 }}>
            <span className="text-display" style={{ color: 'var(--text-primary)', display: 'block', fontSize: 'clamp(2.5rem, 6vw, 5rem)' }}>
              CLIP
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
              Zero-Shot.
            </span>
          </h1>
        </motion.div>
        <motion.div {...fadeUp(0.12)}>
          <p className="text-body" style={{ maxWidth: '620px', fontSize: '1.05rem', lineHeight: 1.8 }}>
            No RSNA training at all — just a pretrained vision-language model matching X-rays to text
            prompts by embedding similarity. Generic CLIP performs at chance. Domain-pretrained
            BiomedCLIP nearly matches the supervised classifier. And one prompt formulation shows
            exactly why accuracy alone can lie.
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

      {/* mechanism + comparison figure */}
      <section className="section">
        <div className="mech-compare-grid" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '4rem', alignItems: 'start' }}>

          {/* left: zero-shot mechanism */}
          <div>
            <motion.div {...fadeUp(0)}>
              <p className="text-subheading" style={{ marginBottom: '0.5rem' }}>How Zero-Shot Works</p>
            </motion.div>
            <motion.div {...fadeUp(0.06)}>
              <h2 style={{ marginBottom: '1.5rem' }}>
                <span className="text-heading" style={{ color: 'var(--text-primary)' }}>No training, </span>
                <span className="text-heading font-display" style={{ fontStyle: 'italic', color: 'var(--text-primary)' }}>just embeddings.</span>
              </h2>
            </motion.div>
            <motion.div {...fadeUp(0.1)}>
              <p className="text-body" style={{ fontSize: '0.95rem', lineHeight: 1.75, marginBottom: '1.5rem' }}>
                Hover a step. Both CLIP and BiomedCLIP follow this exact pipeline — the only difference
                between them is what data the two encoders were pretrained on.
              </p>
            </motion.div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '0', border: '1px solid var(--border)', borderRadius: '8px', overflow: 'hidden' }}>
              {mechanism.map((step, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, x: -12 }}
                  whileInView={{ opacity: 1, x: 0 }}
                  viewport={{ once: true, amount: 0.1 }}
                  transition={{ duration: 0.4, delay: i * 0.06 }}
                >
                  <div
                    className="layer-row"
                    onMouseEnter={() => setActiveStep(i)}
                    onMouseLeave={() => setActiveStep(null)}
                    style={{
                      display: 'flex', alignItems: 'center', gap: '1rem',
                      padding: '0.9rem 1.25rem',
                      borderBottom: i < mechanism.length - 1 ? '1px solid var(--border)' : 'none',
                      background: activeStep === i ? 'var(--surface)' : 'transparent',
                      transition: 'background 0.2s',
                    }}
                  >
                    <div style={{ width: '3px', height: '2rem', borderRadius: '2px', background: layerColor[step.type], flexShrink: 0, opacity: activeStep === i ? 1 : 0.5, transition: 'opacity 0.2s' }} />
                    <div style={{ flex: 1 }}>
                      <p style={{ fontFamily: 'Hanken Grotesk', fontSize: '0.875rem', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '0.15rem' }}>{step.label}</p>
                      <p className="text-subheading" style={{ fontSize: '0.62rem' }}>{step.detail}</p>
                    </div>
                    <span style={{ fontFamily: 'Hanken Grotesk', fontSize: '0.6rem', fontWeight: 700, letterSpacing: '0.1em', textTransform: 'uppercase', color: layerColor[step.type], opacity: 0.8, flexShrink: 0 }}>
                      {step.type}
                    </span>
                  </div>
                </motion.div>
              ))}
            </div>

            <motion.div {...fadeUp(0.3)} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginTop: '1rem' }}>
              <div style={{ width: '1px', height: '24px', background: 'var(--gradient-accent)' }} />
              <p className="text-subheading">CLIP: 400M pairs · BiomedCLIP: 15M biomedical pairs</p>
            </motion.div>
          </div>

          {/* right: comparison figure */}
          <div>
            <motion.div {...fadeUp(0)}>
              <p className="text-subheading" style={{ marginBottom: '0.5rem' }}>General vs Medical</p>
            </motion.div>
            <motion.div {...fadeUp(0.06)}>
              <h2 style={{ marginBottom: '1.5rem' }}>
                <span className="text-heading" style={{ color: 'var(--text-primary)' }}>Same pipeline, </span>
                <span className="text-heading font-display" style={{ fontStyle: 'italic', color: 'var(--text-primary)' }}>different priors.</span>
              </h2>
            </motion.div>
            <motion.div {...fadeUp(0.1)} style={{ border: '1px solid var(--border)', borderRadius: '10px', overflow: 'hidden', background: '#fff' }}>
              <div style={{ position: 'relative', width: '100%', aspectRatio: '889 / 490' }}>
                <Image
                  src="/results/images/clip_comparison.png"
                  alt="OpenAI CLIP vs BiomedCLIP AUC and accuracy comparison"
                  fill
                  sizes="700px"
                  style={{ objectFit: 'contain' }}
                />
              </div>
            </motion.div>
            <motion.div {...fadeUp(0.16)}>
              <p className="text-body" style={{ fontSize: '0.875rem', lineHeight: 1.75, marginTop: '1.25rem' }}>
                Both bars are zero-shot. Neither model has seen a single RSNA training image. The gap is
                entirely a function of what each encoder learned to associate during pretraining.
              </p>
            </motion.div>
          </div>
        </div>
      </section>

      <div className="divider" />

      {/* prompt engineering */}
      <section className="section">
        <motion.div {...fadeUp(0)}>
          <p className="text-subheading" style={{ marginBottom: '0.5rem' }}>Prompt Engineering</p>
        </motion.div>
        <motion.div {...fadeUp(0.06)}>
          <h2 style={{ marginBottom: '1.5rem' }}>
            <span className="text-heading" style={{ color: 'var(--text-primary)' }}>Same model, </span>
            <span className="text-heading font-display" style={{ fontStyle: 'italic', color: 'var(--text-primary)' }}>three prompts.</span>
          </h2>
        </motion.div>
        <motion.div {...fadeUp(0.1)}>
          <p className="text-body" style={{ maxWidth: '640px', fontSize: '0.95rem', lineHeight: 1.8, marginBottom: '2rem' }}>
            All three runs use the same generic OpenAI CLIP — only the wording of the two class prompts
            changes. Accuracy and AUC don&rsquo;t move together.
          </p>
        </motion.div>

        <div className="prompt-main-grid" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '4rem', alignItems: 'start' }}>

          {/* left: prompt tabs + detail */}
          <div>
            <div style={{ display: 'flex', gap: '0', marginBottom: '1.5rem', border: '1px solid var(--border)', borderRadius: '6px', overflow: 'hidden', width: 'fit-content' }}>
              {prompts.map((p, i) => (
                <button
                  key={p.id}
                  onClick={() => setActivePrompt(i)}
                  style={{
                    padding: '0.6rem 1.4rem',
                    background: activePrompt === i ? 'linear-gradient(135deg, #3A015C, #7B2FBE)' : 'transparent',
                    border: 'none',
                    borderRight: i < prompts.length - 1 ? '1px solid var(--border)' : 'none',
                    color: activePrompt === i ? '#fff' : 'var(--text-muted)',
                    fontFamily: 'Hanken Grotesk, sans-serif',
                    fontSize: '0.78rem',
                    fontWeight: 600,
                    letterSpacing: '0.06em',
                    transition: 'background 0.2s, color 0.2s',
                  }}
                >
                  {p.label}
                </button>
              ))}
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1.25rem' }}>
              <div style={{ width: '2px', height: '2rem', background: current.color, borderRadius: '1px' }} />
              <p className="text-subheading" style={{ color: current.color }}>{current.tag} · {current.label}</p>
            </div>

            <p style={{
              fontFamily: 'Hanken Grotesk, sans-serif',
              fontSize: '0.85rem',
              fontStyle: 'italic',
              color: 'var(--text-secondary)',
              lineHeight: 1.7,
              marginBottom: '1.5rem',
              padding: '1rem 1.25rem',
              border: '1px solid var(--border)',
              borderRadius: '6px',
              background: 'var(--surface)',
            }}>
              {current.example}
            </p>

            <div style={{ display: 'flex', gap: '2rem', marginBottom: '1.5rem' }}>
              <div>
                <p style={{ fontFamily: 'Hanken Grotesk', fontSize: '1.6rem', fontWeight: 800, color: current.color, letterSpacing: '-0.03em' }}>
                  {current.auc.toFixed(4)}
                </p>
                <p className="text-subheading">AUC</p>
              </div>
              <div>
                <p style={{ fontFamily: 'Hanken Grotesk', fontSize: '1.6rem', fontWeight: 800, color: 'var(--text-muted)', letterSpacing: '-0.03em' }}>
                  {(current.accuracy * 100).toFixed(1)}%
                </p>
                <p className="text-subheading">Accuracy</p>
              </div>
            </div>

            <p className="text-body" style={{ fontSize: '0.9rem', lineHeight: 1.8 }}>{current.note}</p>
          </div>

          {/* right: prompt figure */}
          <div>
            <motion.div {...fadeUp(0.1)} style={{ border: '1px solid var(--border)', borderRadius: '10px', overflow: 'hidden', background: '#fff' }}>
              <div style={{ position: 'relative', width: '100%', aspectRatio: '790 / 490' }}>
                <Image
                  src="/results/images/clip_zeroshot_prompts.png"
                  alt="CLIP zero-shot performance by prompt set"
                  fill
                  sizes="700px"
                  style={{ objectFit: 'contain' }}
                />
              </div>
            </motion.div>
            <motion.div {...fadeUp(0.16)} style={{
              marginTop: '1.25rem',
              padding: '1.25rem',
              border: '1px solid var(--border-accent)',
              borderRadius: '8px',
              background: 'var(--surface)',
            }}>
              <p style={{ fontFamily: 'Hanken Grotesk', fontSize: '0.875rem', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '0.5rem' }}>
                The accuracy/AUC trap
              </p>
              <p className="text-body" style={{ fontSize: '0.85rem', lineHeight: 1.75 }}>
                Clinical prompts give the highest accuracy (77.4%) but the lowest-but-one AUC (0.44) —
                below chance. Look only at accuracy and you&rsquo;d ship this prompt. AUC shows it&rsquo;s
                defaulting to &rsquo;normal&rsquo; almost every time, which happens to match the dataset&rsquo;s 77.5%
                class imbalance.
              </p>
            </motion.div>
          </div>
        </div>
      </section>

      <div className="divider" />

      {/* honest verdict */}
      <section className="section">
        <motion.div {...fadeUp(0)}>
          <p className="text-subheading" style={{ marginBottom: '0.5rem' }}>What This Means</p>
        </motion.div>
        <motion.div {...fadeUp(0.06)}>
          <h2 style={{ marginBottom: '2rem' }}>
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
      </section>

      <div className="divider" />

      {/* page nav */}
      <section className="section" style={{ paddingTop: 'clamp(2rem, 5vw, 4rem)', paddingBottom: 'clamp(2rem, 5vw, 4rem)' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Link href="/unet" style={{ textDecoration: 'none' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--text-muted)', fontFamily: 'Hanken Grotesk', fontSize: '0.8rem', fontWeight: 600, letterSpacing: '0.06em' }}>
              <ArrowLeft size={13} weight="bold" /> u-net
            </div>
          </Link>
          <Link href="/generative" style={{ textDecoration: 'none' }}>
            <button style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', padding: '0.75rem 1.6rem', background: 'linear-gradient(135deg, #3A015C, #7B2FBE)', border: 'none', borderRadius: '3px', color: '#fff', fontFamily: 'Hanken Grotesk', fontSize: '0.8rem', fontWeight: 600, letterSpacing: '0.05em', boxShadow: '0 4px 24px rgba(123,47,190,0.3)' }}>
              next: generative <ArrowRight size={13} weight="bold" />
            </button>
          </Link>
        </div>
      </section>

      <style jsx>{`
        .layer-row:hover { background: var(--surface) !important; }
        @media (max-width: 900px) {
          .mech-compare-grid { grid-template-columns: 1fr !important; }
          .prompt-main-grid { grid-template-columns: 1fr !important; }
          .metrics-grid { grid-template-columns: repeat(2, 1fr) !important; }
        }
      `}</style>
    </div>
  )
}