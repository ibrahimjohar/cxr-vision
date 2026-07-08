'use client'

import Image from 'next/image'
import Link from 'next/link'
import { motion, AnimatePresence, useInView, animate } from 'framer-motion'
import { useRef, useState, useEffect, useCallback } from 'react'
import { ArrowLeft, ArrowRight, CheckCircle, Sparkle, Lightning, CircleNotch, Warning } from '@phosphor-icons/react'

//deployed modal endpoint. update this if the app is ever renamed or redeployed under a new url.
const API_BASE = 'https://ibrahimjoharfarooqi--cxr-vision-inference-fastapi-app.modal.run'

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
  { value: 89.56, label: 'DCGAN FID', sub: '224×224 · epoch 10 checkpoint', decimals: 2 },
  { value: 64.09, label: 'DDPM FID', sub: '128×128 · DDIM 50-step sampling', decimals: 2 },
  { value: 25.47, label: 'FID Gap', sub: 'Diffusion vs GAN — different resolutions', decimals: 2 },
  { value: 500, label: 'FID Sample Size', sub: 'Inception features, relative comparison only', decimals: 0 },
]

const ganSpecs = [
  { label: 'Architecture', value: 'DCGAN' },
  { label: 'Generator params', value: '5,320,640' },
  { label: 'Discriminator params', value: '693,632' },
  { label: 'Latent dim', value: '100' },
  { label: 'Resolution', value: '224 × 224' },
  { label: 'Epochs', value: '20 (all, no early stop)' },
  { label: 'Learning rate', value: '2e-4 (G and D) · β1 = 0.5' },
  { label: 'Best checkpoint', value: 'Epoch 10 — D/G equilibrium' },
]

const diffusionSpecs = [
  { label: 'Architecture', value: 'DDPM U-Net (ResNet blocks)' },
  { label: 'Parameters', value: '6,613,761' },
  { label: 'Timesteps', value: '1,000 · linear β: 1e-4 → 0.02' },
  { label: 'Resolution', value: '128 × 128' },
  { label: 'Epochs', value: '10' },
  { label: 'Learning rate', value: '2e-4 · batch 8' },
  { label: 'Sampling', value: 'DDIM, 50 steps' },
  { label: 'Loss', value: '0.0216 → 0.0118 (epoch 1 → 10)' },
]

const ganSamples = [
  { epoch: 1, path: '/results/images/gan_samples_epoch001.png' },
  { epoch: 5, path: '/results/images/gan_samples_epoch005.png' },
  { epoch: 10, path: '/results/images/gan_samples_epoch010.png' },
  { epoch: 15, path: '/results/images/gan_samples_epoch015.png' },
  { epoch: 20, path: '/results/images/gan_samples_epoch020.png' },
]

const diffusionSamples = [
  { epoch: 1, path: '/results/images/diffusion_samples_epoch001.png' },
  { epoch: 5, path: '/results/images/diffusion_samples_epoch005.png' },
  { epoch: 10, path: '/results/images/diffusion_samples_epoch010.png' },
]

const verdicts = [
  {
    heading: 'Diffusion wins on FID — but not on a level playing field.',
    body: 'DDPM (64.09) beats DCGAN (89.56) by 25 points, but DDPM trained at 128×128 for 10 epochs while DCGAN trained at 224×224 for 20. Lower resolution tends to lower FID somewhat on its own, so this gap overstates the architectural advantage.',
  },
  {
    heading: 'FID here is relative, not an absolute realism score.',
    body: 'Both scores come from an ImageNet-pretrained Inception network extracting features from chest X-rays — a domain mismatch the figure itself flags. Read these as "diffusion ranks better than GAN," not as calibrated scores comparable to FID values reported on natural images.',
  },
  {
    heading: 'The visual gap holds up independent of the caveats.',
    body: 'Even accounting for resolution and epoch differences, DDPM samples at epoch 10 show cleaner rib structure and fewer checkerboard artifacts than DCGAN\'s. The qualitative difference isn\'t just an artifact of the unfair comparison.',
  },
]

export default function GenerativePage() {
  const [activeGanEpoch, setActiveGanEpoch] = useState(4) // epoch 20
  const [activeDiffEpoch, setActiveDiffEpoch] = useState(2) // epoch 10

  const [ganGenerating, setGanGenerating] = useState(false)
  const [ganGenError, setGanGenError] = useState<string | null>(null)
  const [ganGenImage, setGanGenImage] = useState<string | null>(null)

  const [diffGenerating, setDiffGenerating] = useState(false)
  const [diffGenError, setDiffGenError] = useState<string | null>(null)
  const [diffGenImage, setDiffGenImage] = useState<string | null>(null)

  const currentGan = ganSamples[activeGanEpoch]
  const currentDiff = diffusionSamples[activeDiffEpoch]

  const handleGenerateGan = useCallback(async () => {
    setGanGenerating(true)
    setGanGenError(null)
    setGanGenImage(null)
    try {
      const res = await fetch(`${API_BASE}/generate/gan`, { method: 'POST' })
      if (!res.ok) {
        throw new Error(res.status === 429 ? 'rate limit reached, try again in a minute' : 'generation request failed')
      }
      const data = await res.json()
      setGanGenImage(data.image_png_base64)
    } catch (err) {
      setGanGenError(err instanceof Error ? err.message : 'something went wrong')
    } finally {
      setGanGenerating(false)
    }
  }, [])

  const handleGenerateDiffusion = useCallback(async () => {
    setDiffGenerating(true)
    setDiffGenError(null)
    setDiffGenImage(null)
    try {
      const res = await fetch(`${API_BASE}/generate/diffusion`, { method: 'POST' })
      if (!res.ok) {
        throw new Error(res.status === 429 ? 'rate limit reached, try again in a minute' : 'generation request failed')
      }
      const data = await res.json()
      setDiffGenImage(data.image_png_base64)
    } catch (err) {
      setDiffGenError(err instanceof Error ? err.message : 'something went wrong')
    } finally {
      setDiffGenerating(false)
    }
  }, [])

  return (
    <div>

      {/* header */}
      <section className="section" style={{ paddingBottom: '2rem' }}>
        <motion.div {...fadeUp(0)}>
          <p className="text-subheading" style={{ marginBottom: '0.75rem' }}>Stage 06 · Generative Models</p>
        </motion.div>
        <motion.div {...fadeUp(0.06)}>
          <h1 style={{ marginBottom: '1.5rem', lineHeight: 0.95 }}>
            <span className="text-display" style={{ color: 'var(--text-primary)', display: 'block', fontSize: 'clamp(2.5rem, 6vw, 5rem)' }}>
              GAN vs
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
              Diffusion.
            </span>
          </h1>
        </motion.div>
        <motion.div {...fadeUp(0.12)}>
          <p className="text-body" style={{ maxWidth: '620px', fontSize: '1.05rem', lineHeight: 1.8 }}>
            Two ways to generate a chest X-ray from noise: adversarial training and iterative denoising.
            DDPM scores 25 FID points better than DCGAN — though not on a fully matched comparison,
            as the caveats below explain.
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

      {/* spec sheets */}
      <section className="section">
        <div className="spec-grid" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '4rem', alignItems: 'start' }}>

          {/* left: gan specs */}
          <div>
            <motion.div {...fadeUp(0)}>
              <p className="text-subheading" style={{ marginBottom: '0.5rem' }}>Adversarial</p>
            </motion.div>
            <motion.div {...fadeUp(0.06)}>
              <h2 style={{ marginBottom: '1.5rem' }}>
                <span className="text-heading" style={{ color: 'var(--text-primary)' }}>DCGAN </span>
                <span className="text-heading font-display" style={{ fontStyle: 'italic', color: 'var(--text-primary)' }}>setup.</span>
              </h2>
            </motion.div>
            <motion.div {...fadeUp(0.1)}>
              <p className="text-body" style={{ fontSize: '0.95rem', lineHeight: 1.75, marginBottom: '1.5rem' }}>
                Generator and discriminator trained in tandem. D(x) stabilized at 0.69–0.75 and D(G(z))
                at 0.25–0.31 — a healthy equilibrium, not degenerate collapse.
              </p>
            </motion.div>
            <motion.div {...fadeUp(0.15)} style={{ border: '1px solid var(--border)', borderRadius: '8px', overflow: 'hidden' }}>
              {ganSpecs.map((row, i, arr) => (
                <div key={row.label} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '0.7rem 1.1rem', borderBottom: i < arr.length - 1 ? '1px solid var(--border)' : 'none', background: i % 2 === 0 ? 'transparent' : 'var(--surface)' }}>
                  <p className="text-subheading">{row.label}</p>
                  <p style={{ fontFamily: 'Hanken Grotesk', fontSize: '0.82rem', fontWeight: 700, color: 'var(--text-primary)', textAlign: 'right' }}>{row.value}</p>
                </div>
              ))}
            </motion.div>
          </div>

          {/* right: diffusion specs */}
          <div>
            <motion.div {...fadeUp(0)}>
              <p className="text-subheading" style={{ marginBottom: '0.5rem' }}>Score-Based</p>
            </motion.div>
            <motion.div {...fadeUp(0.06)}>
              <h2 style={{ marginBottom: '1.5rem' }}>
                <span className="text-heading" style={{ color: 'var(--text-primary)' }}>DDPM </span>
                <span className="text-heading font-display" style={{ fontStyle: 'italic', color: 'var(--text-primary)' }}>setup.</span>
              </h2>
            </motion.div>
            <motion.div {...fadeUp(0.1)}>
              <p className="text-body" style={{ fontSize: '0.95rem', lineHeight: 1.75, marginBottom: '1.5rem' }}>
                No adversarial pair to balance — just a denoising network trained to reverse a fixed
                noising process, one small step at a time.
              </p>
            </motion.div>
            <motion.div {...fadeUp(0.15)} style={{ border: '1px solid var(--border)', borderRadius: '8px', overflow: 'hidden' }}>
              {diffusionSpecs.map((row, i, arr) => (
                <div key={row.label} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '0.7rem 1.1rem', borderBottom: i < arr.length - 1 ? '1px solid var(--border)' : 'none', background: i % 2 === 0 ? 'transparent' : 'var(--surface)' }}>
                  <p className="text-subheading">{row.label}</p>
                  <p style={{ fontFamily: 'Hanken Grotesk', fontSize: '0.82rem', fontWeight: 700, color: 'var(--text-primary)', textAlign: 'right' }}>{row.value}</p>
                </div>
              ))}
            </motion.div>
          </div>
        </div>
      </section>

      <div className="divider" />

      {/* sample evolution */}
      <section className="section">
        <motion.div {...fadeUp(0)}>
          <p className="text-subheading" style={{ marginBottom: '0.5rem' }}>Sample Evolution</p>
        </motion.div>
        <motion.div {...fadeUp(0.06)}>
          <h2 style={{ marginBottom: '1.5rem' }}>
            <span className="text-heading" style={{ color: 'var(--text-primary)' }}>Noise to </span>
            <span className="text-heading font-display" style={{ fontStyle: 'italic', color: 'var(--text-primary)' }}>anatomy.</span>
          </h2>
        </motion.div>
        <motion.div {...fadeUp(0.1)}>
          <p className="text-body" style={{ maxWidth: '640px', fontSize: '0.95rem', lineHeight: 1.8, marginBottom: '2rem' }}>
            16 generated samples per checkpoint, no real images involved. GAN starts from checkerboard
            noise; diffusion starts already anatomically plausible but blurry — the two failure modes
            look nothing alike.
          </p>
        </motion.div>

        <div className="sample-evo-grid" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '3rem', alignItems: 'start' }}>

          {/* left: gan viewer */}
          <div>
            <p className="text-subheading" style={{ marginBottom: '0.75rem', color: 'var(--accent-light)' }}>DCGAN</p>
            <div style={{ display: 'flex', gap: '0', marginBottom: '1rem', border: '1px solid var(--border)', borderRadius: '6px', overflow: 'hidden', width: 'fit-content' }}>
              {ganSamples.map((s, i) => (
                <button
                  key={s.epoch}
                  onClick={() => setActiveGanEpoch(i)}
                  style={{
                    padding: '0.5rem 1rem',
                    background: activeGanEpoch === i ? 'linear-gradient(135deg, #3A015C, #7B2FBE)' : 'transparent',
                    border: 'none',
                    borderRight: i < ganSamples.length - 1 ? '1px solid var(--border)' : 'none',
                    color: activeGanEpoch === i ? '#fff' : 'var(--text-muted)',
                    fontFamily: 'Hanken Grotesk, sans-serif',
                    fontSize: '0.72rem',
                    fontWeight: 600,
                    letterSpacing: '0.05em',
                    transition: 'background 0.2s, color 0.2s',
                  }}
                >
                  {s.epoch}
                </button>
              ))}
            </div>
            <AnimatePresence mode="wait">
              <motion.div
                key={currentGan.epoch}
                initial={{ opacity: 0, scale: 0.98 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.98 }}
                transition={{ duration: 0.3 }}
              >
                <div style={{ position: 'relative', width: '100%', aspectRatio: '781 / 789', borderRadius: '10px', overflow: 'hidden', border: '1px solid var(--border)', background: '#fff' }}>
                  <Image src={currentGan.path} alt={`DCGAN samples — epoch ${currentGan.epoch}`} fill sizes="600px" style={{ objectFit: 'contain' }} />
                </div>
              </motion.div>
            </AnimatePresence>
            {activeGanEpoch === 2 && (
              <p className="text-subheading" style={{ marginTop: '0.75rem', color: 'var(--accent-light)' }}>← selected checkpoint (D/G equilibrium)</p>
            )}
          </div>

          {/* right: diffusion viewer */}
          <div>
            <p className="text-subheading" style={{ marginBottom: '0.75rem', color: '#9B6DCC' }}>DDPM Diffusion</p>
            <div style={{ display: 'flex', gap: '0', marginBottom: '1rem', border: '1px solid var(--border)', borderRadius: '6px', overflow: 'hidden', width: 'fit-content' }}>
              {diffusionSamples.map((s, i) => (
                <button
                  key={s.epoch}
                  onClick={() => setActiveDiffEpoch(i)}
                  style={{
                    padding: '0.5rem 1rem',
                    background: activeDiffEpoch === i ? 'linear-gradient(135deg, #3A015C, #7B2FBE)' : 'transparent',
                    border: 'none',
                    borderRight: i < diffusionSamples.length - 1 ? '1px solid var(--border)' : 'none',
                    color: activeDiffEpoch === i ? '#fff' : 'var(--text-muted)',
                    fontFamily: 'Hanken Grotesk, sans-serif',
                    fontSize: '0.72rem',
                    fontWeight: 600,
                    letterSpacing: '0.05em',
                    transition: 'background 0.2s, color 0.2s',
                  }}
                >
                  {s.epoch}
                </button>
              ))}
            </div>
            <AnimatePresence mode="wait">
              <motion.div
                key={currentDiff.epoch}
                initial={{ opacity: 0, scale: 0.98 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.98 }}
                transition={{ duration: 0.3 }}
              >
                <div style={{ position: 'relative', width: '100%', aspectRatio: '781 / 789', borderRadius: '10px', overflow: 'hidden', border: '1px solid var(--border)', background: '#fff' }}>
                  <Image src={currentDiff.path} alt={`Diffusion samples — epoch ${currentDiff.epoch}`} fill sizes="600px" style={{ objectFit: 'contain' }} />
                </div>
              </motion.div>
            </AnimatePresence>
            {activeDiffEpoch === 2 && (
              <p className="text-subheading" style={{ marginTop: '0.75rem', color: '#9B6DCC' }}>← final checkpoint</p>
            )}
          </div>
        </div>
      </section>

      <div className="divider" />

      {/* fid comparison + honest verdict */}
      <section className="section">
        <div className="fid-verdict-grid" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '4rem', alignItems: 'start' }}>

          {/* left: fid figure */}
          <div>
            <motion.div {...fadeUp(0)}>
              <p className="text-subheading" style={{ marginBottom: '0.5rem' }}>FID Comparison</p>
            </motion.div>
            <motion.div {...fadeUp(0.06)}>
              <h2 style={{ marginBottom: '1.5rem' }}>
                <span className="text-heading" style={{ color: 'var(--text-primary)' }}>Lower is </span>
                <span className="text-heading font-display" style={{ fontStyle: 'italic', color: 'var(--text-primary)' }}>better.</span>
              </h2>
            </motion.div>
            <motion.div {...fadeUp(0.1)} style={{ border: '1px solid var(--border)', borderRadius: '10px', overflow: 'hidden', background: '#fff' }}>
              <div style={{ position: 'relative', width: '100%', aspectRatio: '626 / 489' }}>
                <Image
                  src="/results/images/generative_fid_comparison.png"
                  alt="FID comparison — DCGAN vs DDPM diffusion"
                  fill
                  sizes="700px"
                  style={{ objectFit: 'contain' }}
                />
              </div>
            </motion.div>
          </div>

          {/* right: verdict cards */}
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

      {/* live generation — calls the deployed modal endpoints */}
      <section className="section">
        <motion.div {...fadeUp(0)}>
          <p className="text-subheading" style={{ marginBottom: '0.5rem' }}>Live Generation</p>
        </motion.div>
        <motion.div {...fadeUp(0.06)}>
          <h2 style={{ marginBottom: '1.5rem' }}>
            <span className="text-heading" style={{ color: 'var(--text-primary)' }}>Generate a </span>
            <span className="text-heading font-display" style={{ fontStyle: 'italic', color: 'var(--text-primary)' }}>sample, live.</span>
          </h2>
        </motion.div>
        <motion.div {...fadeUp(0.1)}>
          <p className="text-body" style={{ maxWidth: '620px', fontSize: '0.95rem', lineHeight: 1.8, marginBottom: '2rem' }}>
            Runs inference on a live Modal-hosted CPU endpoint — one forward pass for GAN,
            50 DDIM steps (~10-20s) for diffusion.
          </p>
        </motion.div>

        <div className="live-gen-grid" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '3rem', alignItems: 'start' }}>

          {/* gan generation */}
          <div>
            <p className="text-subheading" style={{ marginBottom: '0.75rem', color: 'var(--accent-light)' }}>DCGAN</p>
            <div style={{
              position: 'relative', width: '100%', aspectRatio: '1', borderRadius: '10px',
              border: ganGenImage ? '1px solid var(--border)' : '1px dashed var(--border-accent)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              background: ganGenImage ? '#fff' : 'var(--surface)',
              overflow: 'hidden', marginBottom: '1.25rem',
            }}>
              {ganGenerating ? (
                <motion.span animate={{ rotate: 360 }} transition={{ duration: 1, repeat: Infinity, ease: 'linear' }} style={{ display: 'flex' }}>
                  <CircleNotch size={32} style={{ color: 'var(--text-muted)' }} weight="bold" />
                </motion.span>
              ) : ganGenImage ? (
                // eslint-disable-next-line @next/next/no-img-element
                <img src={`data:image/png;base64,${ganGenImage}`} alt="generated gan sample" style={{ width: '100%', height: '100%', objectFit: 'contain' }} />
              ) : (
                <Sparkle size={36} style={{ color: 'var(--text-muted)', opacity: 0.4 }} />
              )}
            </div>

            <button
              onClick={handleGenerateGan}
              disabled={ganGenerating}
              style={{
                width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem',
                padding: '0.85rem',
                background: ganGenerating ? 'var(--glass-btn-bg)' : 'linear-gradient(135deg, #3A015C, #7B2FBE)',
                border: ganGenerating ? '1px solid var(--glass-btn-border)' : 'none',
                borderRadius: '3px',
                color: ganGenerating ? 'var(--text-muted)' : '#fff',
                fontFamily: 'Hanken Grotesk', fontSize: '0.8rem', fontWeight: 600, letterSpacing: '0.05em',
                cursor: ganGenerating ? 'not-allowed' : 'pointer',
                boxShadow: ganGenerating ? 'none' : '0 4px 24px rgba(123,47,190,0.3)',
              }}
            >
              {ganGenerating ? (
                <motion.span animate={{ rotate: 360 }} transition={{ duration: 1, repeat: Infinity, ease: 'linear' }} style={{ display: 'flex' }}>
                  <CircleNotch size={14} weight="bold" />
                </motion.span>
              ) : (
                <Lightning size={14} weight="bold" />
              )}
              {ganGenerating ? 'generating' : 'generate gan sample'}
            </button>

            <p className="text-subheading" style={{ textAlign: 'center', marginTop: '0.6rem' }}>
              {ganGenerating ? 'one forward pass, should be quick' : ganGenError ? '' : 'runs on a live modal-hosted cpu endpoint'}
            </p>
            {ganGenError && (
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', justifyContent: 'center', marginTop: '0.4rem' }}>
                <Warning size={12} style={{ color: '#C084FC' }} weight="bold" />
                <p className="text-subheading" style={{ color: '#C084FC' }}>{ganGenError}</p>
              </div>
            )}
          </div>

          {/* diffusion generation */}
          <div>
            <p className="text-subheading" style={{ marginBottom: '0.75rem', color: '#9B6DCC' }}>DDPM Diffusion</p>
            <div style={{
              position: 'relative', width: '100%', aspectRatio: '1', borderRadius: '10px',
              border: diffGenImage ? '1px solid var(--border)' : '1px dashed var(--border-accent)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              background: diffGenImage ? '#fff' : 'var(--surface)',
              overflow: 'hidden', marginBottom: '1.25rem',
            }}>
              {diffGenerating ? (
                <motion.span animate={{ rotate: 360 }} transition={{ duration: 1, repeat: Infinity, ease: 'linear' }} style={{ display: 'flex' }}>
                  <CircleNotch size={32} style={{ color: 'var(--text-muted)' }} weight="bold" />
                </motion.span>
              ) : diffGenImage ? (
                // eslint-disable-next-line @next/next/no-img-element
                <img src={`data:image/png;base64,${diffGenImage}`} alt="generated diffusion sample" style={{ width: '100%', height: '100%', objectFit: 'contain' }} />
              ) : (
                <Sparkle size={36} style={{ color: 'var(--text-muted)', opacity: 0.4 }} />
              )}
            </div>

            <button
              onClick={handleGenerateDiffusion}
              disabled={diffGenerating}
              style={{
                width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem',
                padding: '0.85rem',
                background: diffGenerating ? 'var(--glass-btn-bg)' : 'linear-gradient(135deg, #3A015C, #7B2FBE)',
                border: diffGenerating ? '1px solid var(--glass-btn-border)' : 'none',
                borderRadius: '3px',
                color: diffGenerating ? 'var(--text-muted)' : '#fff',
                fontFamily: 'Hanken Grotesk', fontSize: '0.8rem', fontWeight: 600, letterSpacing: '0.05em',
                cursor: diffGenerating ? 'not-allowed' : 'pointer',
                boxShadow: diffGenerating ? 'none' : '0 4px 24px rgba(123,47,190,0.3)',
              }}
            >
              {diffGenerating ? (
                <motion.span animate={{ rotate: 360 }} transition={{ duration: 1, repeat: Infinity, ease: 'linear' }} style={{ display: 'flex' }}>
                  <CircleNotch size={14} weight="bold" />
                </motion.span>
              ) : (
                <Lightning size={14} weight="bold" />
              )}
              {diffGenerating ? 'generating' : 'generate diffusion sample'}
            </button>

            <p className="text-subheading" style={{ textAlign: 'center', marginTop: '0.6rem' }}>
              {diffGenerating ? '50 ddim steps, roughly 10-20s on cpu' : diffGenError ? '' : 'runs on a live modal-hosted cpu endpoint'}
            </p>
            {diffGenError && (
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', justifyContent: 'center', marginTop: '0.4rem' }}>
                <Warning size={12} style={{ color: '#C084FC' }} weight="bold" />
                <p className="text-subheading" style={{ color: '#C084FC' }}>{diffGenError}</p>
              </div>
            )}
          </div>
        </div>
      </section>

      <div className="divider" />

      {/* page nav */}
      <section className="section" style={{ paddingTop: 'clamp(2rem, 5vw, 4rem)', paddingBottom: 'clamp(2rem, 5vw, 4rem)' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Link href="/clip" style={{ textDecoration: 'none' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--text-muted)', fontFamily: 'Hanken Grotesk', fontSize: '0.8rem', fontWeight: 600, letterSpacing: '0.06em' }}>
              <ArrowLeft size={13} weight="bold" /> clip
            </div>
          </Link>
          <Link href="/inference" style={{ textDecoration: 'none' }}>
            <button style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', padding: '0.75rem 1.6rem', background: 'linear-gradient(135deg, #3A015C, #7B2FBE)', border: 'none', borderRadius: '3px', color: '#fff', fontFamily: 'Hanken Grotesk', fontSize: '0.8rem', fontWeight: 600, letterSpacing: '0.05em', boxShadow: '0 4px 24px rgba(123,47,190,0.3)' }}>
              next: inference <ArrowRight size={13} weight="bold" />
            </button>
          </Link>
        </div>
      </section>

      <style jsx>{`
        @media (max-width: 900px) {
          .spec-grid { grid-template-columns: 1fr !important; }
          .sample-evo-grid { grid-template-columns: 1fr !important; }
          .fid-verdict-grid { grid-template-columns: 1fr !important; }
          .live-gen-grid { grid-template-columns: 1fr !important; }
          .metrics-grid { grid-template-columns: repeat(2, 1fr) !important; }
        }
      `}</style>
    </div>
  )
}