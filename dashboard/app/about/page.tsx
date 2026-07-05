'use client'

import Link from 'next/link'
import { motion } from 'framer-motion'
import { ArrowLeft, ArrowRight, GithubLogo } from '@phosphor-icons/react'

const fadeUp = (delay = 0) => ({
  initial: { opacity: 0, y: 18 },
  whileInView: { opacity: 1, y: 0 },
  viewport: { once: true, amount: 0.1 } as const,
  transition: { duration: 0.55, delay },
})

const stack = [
  { label: 'Next.js 16', group: 'Frontend' },
  { label: 'React 19', group: 'Frontend' },
  { label: 'Framer Motion', group: 'Frontend' },
  { label: 'Recharts', group: 'Frontend' },
  { label: 'PyTorch 2.7', group: 'Modeling' },
  { label: 'FastAPI', group: 'Backend' },
  { label: 'Modal', group: 'Backend' },
  { label: 'Docker', group: 'Deployment' },
  { label: 'GitHub Actions', group: 'Deployment' },
  { label: 'Vercel', group: 'Deployment' },
]

const models = [
  'ResNet18 Classifier',
  'Variational Autoencoder',
  'DCGAN',
  'Attention U-Net',
  'DDPM Diffusion',
  'CLIP · BiomedCLIP',
]

export default function AboutPage() {
  return (
    <div>

      {/* header */}
      <section className="section" style={{ paddingBottom: '2rem' }}>
        <motion.div {...fadeUp(0)}>
          <p className="text-subheading" style={{ marginBottom: '0.75rem' }}>About</p>
        </motion.div>
        <motion.div {...fadeUp(0.06)}>
          <h1 style={{ marginBottom: '1.5rem', lineHeight: 0.95 }}>
            <span className="text-display" style={{ color: 'var(--text-primary)', display: 'block', fontSize: 'clamp(2.5rem, 6vw, 5rem)' }}>
              This
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
              Project.
            </span>
          </h1>
        </motion.div>
        <motion.div {...fadeUp(0.12)}>
          <p className="text-body" style={{ maxWidth: '620px', fontSize: '1.05rem', lineHeight: 1.8 }}>
            cxr-vision is a chest X-ray pathology detection pipeline built for Deep Learning for
            Perception (CS4045) at FAST NUCES Karachi, and doubling as a portfolio piece — six models
            spanning classical CV, generative modeling, segmentation, and multimodal zero-shot
            learning, all evaluated on the same 26,684-image RSNA dataset.
          </p>
        </motion.div>
      </section>

      <div className="divider" />

      {/* deployment story */}
      <section className="section" style={{ paddingTop: 'clamp(2.5rem, 5vw, 4rem)', paddingBottom: 'clamp(2.5rem, 5vw, 4rem)' }}>
        <motion.div {...fadeUp(0)}>
          <p className="text-subheading" style={{ marginBottom: '0.5rem' }}>Deployment Story</p>
        </motion.div>
        <motion.div {...fadeUp(0.06)}>
          <h2 style={{ marginBottom: '1.5rem' }}>
            <span className="text-heading" style={{ color: 'var(--text-primary)' }}>From a 4GB GPU </span>
            <span className="text-heading font-display" style={{ fontStyle: 'italic', color: 'var(--text-primary)' }}>to the web.</span>
          </h2>
        </motion.div>
        <motion.div {...fadeUp(0.1)}>
          <div className="accent-line">
            <p className="text-body" style={{ fontSize: '1.05rem', lineHeight: 1.9, maxWidth: '640px' }}>
              Every model here was trained locally on a GTX 1650 — 4GB of VRAM, which shaped nearly every
              architecture decision: reduced U-Net feature widths, 128×128 diffusion, CPU-friendly
              batch sizes. Training happens locally; inference is dockerized, shipped through GitHub
              Actions CI/CD, and served serverless on Modal&rsquo;s free CPU tier. This dashboard, built in
              Next.js, is the layer that ties it together.
            </p>
          </div>
        </motion.div>
      </section>

      <div className="divider" />

      {/* models trained */}
      <section className="section">
        <motion.div {...fadeUp(0)}>
          <p className="text-subheading" style={{ marginBottom: '0.5rem' }}>Models Trained</p>
        </motion.div>
        <motion.div {...fadeUp(0.06)}>
          <h2 style={{ marginBottom: '2rem' }}>
            <span className="text-heading" style={{ color: 'var(--text-primary)' }}>Six approaches, </span>
            <span className="text-heading font-display" style={{ fontStyle: 'italic', color: 'var(--text-primary)' }}>one dataset.</span>
          </h2>
        </motion.div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1px', background: 'var(--border)', borderRadius: '8px', overflow: 'hidden' }} className="models-grid">
          {models.map((m, i) => (
            <motion.div key={m} {...fadeUp(i * 0.05)}>
              <div style={{ padding: '1.5rem', background: 'var(--bg)', height: '100%' }}>
                <p style={{ fontFamily: 'Instrument Serif, serif', fontSize: '1.3rem', color: 'var(--accent-light)', opacity: 0.35, lineHeight: 1, marginBottom: '0.75rem' }}>
                  {String(i + 1).padStart(2, '0')}
                </p>
                <p style={{ fontFamily: 'Hanken Grotesk', fontSize: '0.9rem', fontWeight: 600, color: 'var(--text-primary)' }}>{m}</p>
              </div>
            </motion.div>
          ))}
        </div>
      </section>

      <div className="divider" />

      {/* tech stack */}
      <section className="section">
        <motion.div {...fadeUp(0)}>
          <p className="text-subheading" style={{ marginBottom: '0.5rem' }}>Stack</p>
        </motion.div>
        <motion.div {...fadeUp(0.06)}>
          <h2 style={{ marginBottom: '2rem' }}>
            <span className="text-heading" style={{ color: 'var(--text-primary)' }}>Tools </span>
            <span className="text-heading font-display" style={{ fontStyle: 'italic', color: 'var(--text-primary)' }}>used.</span>
          </h2>
        </motion.div>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.75rem' }}>
          {stack.map((s, i) => (
            <motion.div key={s.label} {...fadeUp(i * 0.03)}>
              <div style={{
                display: 'flex', alignItems: 'center', gap: '0.6rem',
                padding: '0.6rem 1.1rem',
                border: '1px solid var(--border)',
                borderRadius: '20px',
                background: 'var(--surface)',
              }}>
                <span style={{ fontFamily: 'Hanken Grotesk', fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-primary)' }}>{s.label}</span>
                <span className="text-subheading" style={{ fontSize: '0.58rem' }}>{s.group}</span>
              </div>
            </motion.div>
          ))}
        </div>
      </section>

      <div className="divider" />

      {/* philosophy */}
      <section className="section">
        <motion.div {...fadeUp(0)}>
          <p className="text-subheading" style={{ marginBottom: '0.5rem' }}>Philosophy</p>
        </motion.div>
        <motion.div {...fadeUp(0.06)}>
          <h2 style={{ marginBottom: '1.5rem' }}>
            <span className="text-heading" style={{ color: 'var(--text-primary)' }}>Report what </span>
            <span className="text-heading font-display" style={{ fontStyle: 'italic', color: 'var(--text-primary)' }}>happened.</span>
          </h2>
        </motion.div>
        <motion.div {...fadeUp(0.1)} style={{ padding: '2rem', border: '1px solid var(--border-accent)', borderRadius: '10px', background: 'var(--surface)', maxWidth: '680px' }}>
          <p className="text-body" style={{ fontSize: '1rem', lineHeight: 1.9 }}>
            Every result on this dashboard is reported as it came out of evaluation — including the VAE&rsquo;s
            chance-level AUC, the U-Net&rsquo;s inflated overall Dice, and the CLIP prompt that scores 77%
            accuracy while sitting at an AUC below chance. A negative result explained honestly is more
            useful, and more interesting, than a good number left unexamined.
          </p>
        </motion.div>
      </section>

      <div className="divider" />

      {/* links */}
      <section className="section" style={{ paddingTop: 'clamp(2.5rem, 5vw, 4rem)', paddingBottom: 'clamp(2.5rem, 5vw, 4rem)' }}>
        <motion.div {...fadeUp(0)}>
          <a
            href="https://github.com/ibrahimjohar/cxr-vision"
            target="_blank"
            rel="noopener noreferrer"
            style={{ textDecoration: 'none' }}
          >
            <div style={{
              display: 'inline-flex', alignItems: 'center', gap: '0.75rem',
              padding: '1rem 1.75rem',
              border: '1px solid var(--border)',
              borderRadius: '6px',
              background: 'var(--surface)',
            }}>
              <GithubLogo size={20} style={{ color: 'var(--text-primary)' }} weight="fill" />
              <div>
                <p style={{ fontFamily: 'Hanken Grotesk', fontSize: '0.9rem', fontWeight: 600, color: 'var(--text-primary)' }}>ibrahimjohar / cxr-vision</p>
                <p className="text-subheading">Full source on GitHub</p>
              </div>
            </div>
          </a>
        </motion.div>
      </section>

      <div className="divider" />

      {/* page nav */}
      <section className="section" style={{ paddingTop: 'clamp(2rem, 5vw, 4rem)', paddingBottom: 'clamp(2rem, 5vw, 4rem)' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Link href="/inference" style={{ textDecoration: 'none' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--text-muted)', fontFamily: 'Hanken Grotesk', fontSize: '0.8rem', fontWeight: 600, letterSpacing: '0.06em' }}>
              <ArrowLeft size={13} weight="bold" /> inference
            </div>
          </Link>
          <Link href="/overview" style={{ textDecoration: 'none' }}>
            <button style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', padding: '0.75rem 1.6rem', background: 'linear-gradient(135deg, #3A015C, #7B2FBE)', border: 'none', borderRadius: '3px', color: '#fff', fontFamily: 'Hanken Grotesk', fontSize: '0.8rem', fontWeight: 600, letterSpacing: '0.05em', boxShadow: '0 4px 24px rgba(123,47,190,0.3)' }}>
              back to overview <ArrowRight size={13} weight="bold" />
            </button>
          </Link>
        </div>
      </section>

      <style jsx>{`
        @media (max-width: 900px) {
          .models-grid { grid-template-columns: repeat(2, 1fr) !important; }
        }
        @media (max-width: 600px) {
          .models-grid { grid-template-columns: 1fr !important; }
        }
      `}</style>
    </div>
  )
}