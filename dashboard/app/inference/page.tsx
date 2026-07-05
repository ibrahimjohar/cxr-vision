'use client'

import Link from 'next/link'
import { motion } from 'framer-motion'
import { useRef, useState, useCallback } from 'react'
import {
  ArrowLeft, ArrowRight, UploadSimple, ChartBar,
  Selection, ArrowsClockwise, Sparkle, Lightning, X,
} from '@phosphor-icons/react'

const fadeUp = (delay = 0) => ({
  initial: { opacity: 0, y: 18 },
  whileInView: { opacity: 1, y: 0 },
  viewport: { once: true, amount: 0.1 } as const,
  transition: { duration: 0.55, delay },
})

const pipeline = [
  { step: '01', label: 'ResNet18 Classifier', endpoint: 'POST /predict', detail: 'Normal vs Pneumonia probability', type: 'backbone' },
  { step: '02', label: 'Attention U-Net', endpoint: 'POST /predict', detail: 'Opacity localization heatmap', type: 'backbone' },
  { step: '03', label: 'VAE', endpoint: 'POST /predict', detail: 'Reconstruction + anomaly score', type: 'reg' },
  { step: '04', label: 'CLIP + BiomedCLIP', endpoint: 'POST /clip', detail: 'Zero-shot similarity scores', type: 'head' },
]

const layerColor: Record<string, string> = {
  io: 'var(--text-muted)',
  backbone: 'var(--accent-light)',
  reg: '#9B6DCC',
  head: '#C084FC',
}

const resultCards = [
  { id: 'classifier', label: 'Classifier', icon: ChartBar, detail: 'Normal / Pneumonia probability' },
  { id: 'unet', label: 'U-Net Heatmap', icon: Selection, detail: 'Opacity localization overlay' },
  { id: 'vae', label: 'VAE Reconstruction', icon: ArrowsClockwise, detail: 'Recon error anomaly score' },
  { id: 'clip', label: 'CLIP Scores', icon: Sparkle, detail: 'Zero-shot vs BiomedCLIP' },
]

export default function InferencePage() {
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [fileName, setFileName] = useState<string | null>(null)
  const [dragActive, setDragActive] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  const handleFile = useCallback((file: File | undefined) => {
    if (!file || !file.type.startsWith('image/')) return
    setPreviewUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev)
      return URL.createObjectURL(file)
    })
    setFileName(file.name)
  }, [])

  const onDrop = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    setDragActive(false)
    handleFile(e.dataTransfer.files?.[0])
  }, [handleFile])

  const clearFile = useCallback(() => {
    setPreviewUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev)
      return null
    })
    setFileName(null)
    if (inputRef.current) inputRef.current.value = ''
  }, [])

  return (
    <div>

      {/* header */}
      <section className="section" style={{ paddingBottom: '2rem' }}>
        <motion.div {...fadeUp(0)}>
          <p className="text-subheading" style={{ marginBottom: '0.75rem' }}>Stage 07 · Live Inference</p>
        </motion.div>
        <motion.div {...fadeUp(0.06)}>
          <h1 style={{ marginBottom: '1.5rem', lineHeight: 0.95 }}>
            <span className="text-display" style={{ color: 'var(--text-primary)', display: 'block', fontSize: 'clamp(2.5rem, 6vw, 5rem)' }}>
              Upload.
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
              Inspect.
            </span>
          </h1>
        </motion.div>
        <motion.div {...fadeUp(0.12)}>
          <p className="text-body" style={{ maxWidth: '620px', fontSize: '1.05rem', lineHeight: 1.8 }}>
            One chest X-ray, four models. Upload an image and it runs through the classifier, the
            attention U-Net, the VAE, and both CLIP variants — all four results side by side.
            Inference isn&apos;t live yet; this is the interface waiting on the Modal backend.
          </p>
        </motion.div>
      </section>

      <div className="divider" />

      {/* upload + pipeline */}
      <section className="section" style={{ paddingTop: 'clamp(2.5rem, 5vw, 4rem)' }}>
        <div className="upload-pipeline-grid" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '4rem', alignItems: 'start' }}>

          {/* left: upload dropzone */}
          <div>
            <motion.div {...fadeUp(0)}>
              <p className="text-subheading" style={{ marginBottom: '0.5rem' }}>Input</p>
            </motion.div>
            <motion.div {...fadeUp(0.06)}>
              <h2 style={{ marginBottom: '1.5rem' }}>
                <span className="text-heading" style={{ color: 'var(--text-primary)' }}>Drop an </span>
                <span className="text-heading font-display" style={{ fontStyle: 'italic', color: 'var(--text-primary)' }}>X-ray.</span>
              </h2>
            </motion.div>

            <motion.div {...fadeUp(0.1)}>
              <input
                ref={inputRef}
                type="file"
                accept="image/*"
                onChange={(e) => handleFile(e.target.files?.[0])}
                style={{ display: 'none' }}
              />

              <div
                onClick={() => !previewUrl && inputRef.current?.click()}
                onDragOver={(e) => { e.preventDefault(); setDragActive(true) }}
                onDragLeave={() => setDragActive(false)}
                onDrop={onDrop}
                style={{
                  position: 'relative',
                  width: '100%',
                  aspectRatio: '1',
                  borderRadius: '10px',
                  border: `1px dashed ${dragActive ? 'var(--accent-light)' : 'var(--border-accent)'}`,
                  background: dragActive ? 'var(--surface-hover)' : 'var(--surface)',
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: '1rem',
                  cursor: previewUrl ? 'default' : 'pointer',
                  overflow: 'hidden',
                  transition: 'background 0.2s, border-color 0.2s',
                }}
              >
                {previewUrl ? (
                  <>
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img
                      src={previewUrl}
                      alt="Uploaded chest X-ray preview"
                      style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', objectFit: 'cover' }}
                    />
                    <div style={{ position: 'absolute', inset: 0, background: 'linear-gradient(to top, rgba(8,5,16,0.65) 0%, transparent 45%)' }} />
                    <button
                      onClick={(e) => { e.stopPropagation(); clearFile() }}
                      style={{
                        position: 'absolute', top: '0.75rem', right: '0.75rem',
                        width: '28px', height: '28px', borderRadius: '50%',
                        background: 'rgba(8,5,16,0.7)', border: '1px solid rgba(255,237,223,0.2)',
                        display: 'flex', alignItems: 'center', justifyContent: 'center',
                        color: '#fff',
                      }}
                    >
                      <X size={13} weight="bold" />
                    </button>
                    <div style={{ position: 'absolute', bottom: '1rem', left: '1rem', right: '1rem' }}>
                      <span style={{ fontFamily: 'monospace', fontSize: '0.65rem', color: 'rgba(200,170,255,0.9)', letterSpacing: '0.05em' }}>
                        {fileName}
                      </span>
                    </div>
                  </>
                ) : (
                  <>
                    <UploadSimple size={28} style={{ color: 'var(--text-muted)' }} />
                    <div style={{ textAlign: 'center' }}>
                      <p style={{ fontFamily: 'Hanken Grotesk', fontSize: '0.875rem', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '0.3rem' }}>
                        Click or drop an image
                      </p>
                      <p className="text-subheading">PNG, JPG · frontal chest X-ray</p>
                    </div>
                  </>
                )}
              </div>
            </motion.div>

            <motion.div {...fadeUp(0.16)} style={{ marginTop: '1.25rem' }}>
              <button
                disabled
                style={{
                  width: '100%',
                  display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem',
                  padding: '0.85rem',
                  background: 'var(--glass-btn-bg)',
                  border: '1px solid var(--glass-btn-border)',
                  borderRadius: '3px',
                  color: 'var(--text-muted)',
                  fontFamily: 'Hanken Grotesk',
                  fontSize: '0.85rem',
                  fontWeight: 600,
                  letterSpacing: '0.05em',
                  cursor: 'not-allowed',
                  opacity: previewUrl ? 0.7 : 0.5,
                }}
              >
                <Lightning size={14} weight="bold" />
                run inference
              </button>
              <p className="text-subheading" style={{ textAlign: 'center', marginTop: '0.6rem' }}>
                coming soon — pending Modal deployment
              </p>
            </motion.div>
          </div>

          {/* right: pipeline */}
          <div>
            <motion.div {...fadeUp(0)}>
              <p className="text-subheading" style={{ marginBottom: '0.5rem' }}>What Runs</p>
            </motion.div>
            <motion.div {...fadeUp(0.06)}>
              <h2 style={{ marginBottom: '1.5rem' }}>
                <span className="text-heading" style={{ color: 'var(--text-primary)' }}>Four models, </span>
                <span className="text-heading font-display" style={{ fontStyle: 'italic', color: 'var(--text-primary)' }}>one pass.</span>
              </h2>
            </motion.div>
            <motion.div {...fadeUp(0.1)}>
              <p className="text-body" style={{ fontSize: '0.95rem', lineHeight: 1.75, marginBottom: '1.5rem' }}>
                Two Modal endpoints handle all four models — one shared call for the classifier, U-Net,
                and VAE, and a second for both CLIP variants.
              </p>
            </motion.div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '0', border: '1px solid var(--border)', borderRadius: '8px', overflow: 'hidden' }}>
              {pipeline.map((p, i) => (
                <motion.div
                  key={p.step}
                  initial={{ opacity: 0, x: -12 }}
                  whileInView={{ opacity: 1, x: 0 }}
                  viewport={{ once: true, amount: 0.1 }}
                  transition={{ duration: 0.4, delay: i * 0.06 }}
                >
                  <div style={{
                    display: 'flex', alignItems: 'center', gap: '1rem',
                    padding: '0.9rem 1.25rem',
                    borderBottom: i < pipeline.length - 1 ? '1px solid var(--border)' : 'none',
                  }}>
                    <div style={{ width: '3px', height: '2rem', borderRadius: '2px', background: layerColor[p.type], flexShrink: 0, opacity: 0.7 }} />
                    <div style={{ flex: 1 }}>
                      <p style={{ fontFamily: 'Hanken Grotesk', fontSize: '0.875rem', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '0.15rem' }}>{p.label}</p>
                      <p className="text-subheading" style={{ fontSize: '0.62rem' }}>{p.detail}</p>
                    </div>
                    <span style={{ fontFamily: 'monospace', fontSize: '0.62rem', color: layerColor[p.type], opacity: 0.8, flexShrink: 0 }}>
                      {p.endpoint}
                    </span>
                  </div>
                </motion.div>
              ))}
            </div>
          </div>
        </div>
      </section>

      <div className="divider" />

      {/* result cards — placeholders until backend is live */}
      <section className="section">
        <motion.div {...fadeUp(0)}>
          <p className="text-subheading" style={{ marginBottom: '0.5rem' }}>Results</p>
        </motion.div>
        <motion.div {...fadeUp(0.06)}>
          <h2 style={{ marginBottom: '2rem' }}>
            <span className="text-heading" style={{ color: 'var(--text-primary)' }}>Four outputs, </span>
            <span className="text-heading font-display" style={{ fontStyle: 'italic', color: 'var(--text-primary)' }}>side by side.</span>
          </h2>
        </motion.div>

        <div className="result-cards-grid" style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1rem' }}>
          {resultCards.map((card, i) => {
            const Icon = card.icon
            return (
              <motion.div key={card.id} {...fadeUp(i * 0.06)}>
                <div style={{
                  padding: '1.75rem 1.25rem',
                  border: '1px dashed var(--border-accent)',
                  borderRadius: '8px',
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  textAlign: 'center',
                  gap: '0.75rem',
                  minHeight: '180px',
                  justifyContent: 'center',
                }}>
                  <Icon size={22} style={{ color: 'var(--text-muted)', opacity: 0.5 }} />
                  <p style={{ fontFamily: 'Hanken Grotesk', fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-primary)' }}>{card.label}</p>
                  <p className="text-subheading" style={{ fontSize: '0.62rem' }}>{card.detail}</p>
                  <p className="text-subheading" style={{ fontSize: '0.58rem', color: 'var(--text-muted)', opacity: 0.7 }}>renders after upload</p>
                </div>
              </motion.div>
            )
          })}
        </div>
      </section>

      <div className="divider" />

      {/* page nav */}
      <section className="section" style={{ paddingTop: 'clamp(2rem, 5vw, 4rem)', paddingBottom: 'clamp(2rem, 5vw, 4rem)' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Link href="/generative" style={{ textDecoration: 'none' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--text-muted)', fontFamily: 'Hanken Grotesk', fontSize: '0.8rem', fontWeight: 600, letterSpacing: '0.06em' }}>
              <ArrowLeft size={13} weight="bold" /> generative
            </div>
          </Link>
          <Link href="/about" style={{ textDecoration: 'none' }}>
            <button style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', padding: '0.75rem 1.6rem', background: 'linear-gradient(135deg, #3A015C, #7B2FBE)', border: 'none', borderRadius: '3px', color: '#fff', fontFamily: 'Hanken Grotesk', fontSize: '0.8rem', fontWeight: 600, letterSpacing: '0.05em', boxShadow: '0 4px 24px rgba(123,47,190,0.3)' }}>
              next: about <ArrowRight size={13} weight="bold" />
            </button>
          </Link>
        </div>
      </section>

      <style jsx>{`
        @media (max-width: 900px) {
          .upload-pipeline-grid { grid-template-columns: 1fr !important; }
        }
        @media (max-width: 700px) {
          .result-cards-grid { grid-template-columns: repeat(2, 1fr) !important; }
        }
      `}</style>
    </div>
  )
}