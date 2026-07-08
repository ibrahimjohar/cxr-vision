'use client'

import Link from 'next/link'
import { motion, AnimatePresence } from 'framer-motion'
import { useRef, useState, useCallback, useEffect } from 'react'
import {
  ArrowLeft, ArrowRight, UploadSimple, ChartBar,
  Selection, ArrowsClockwise, Sparkle, Lightning, X,
  CircleNotch, Warning,
} from '@phosphor-icons/react'

//deployed modal endpoint. update this if the app is ever renamed or redeployed under a new url.
const API_BASE = 'https://ibrahimjoharfarooqi--cxr-vision-inference-fastapi-app.modal.run'

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

interface PredictResponse {
  classifier: { normal: number; pneumonia: number }
  unet: { mask_png_base64: string }
  vae: { reconstruction_png_base64: string; reconstruction_mse: number }
}

interface ClipResponse {
  openai_clip: { normal: number; pneumonia: number }
  biomedclip: { normal: number; pneumonia: number }
}

export default function InferencePage() {
  const [file, setFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [fileName, setFileName] = useState<string | null>(null)
  const [dragActive, setDragActive] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [predictResult, setPredictResult] = useState<PredictResponse | null>(null)
  const [clipResult, setClipResult] = useState<ClipResponse | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)
  const resultsRef = useRef<HTMLDivElement>(null)

  const handleFile = useCallback((f: File | undefined) => {
    if (!f || !f.type.startsWith('image/')) return
    setPreviewUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev)
      return URL.createObjectURL(f)
    })
    setFile(f)
    setFileName(f.name)
    setPredictResult(null)
    setClipResult(null)
    setError(null)
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
    setFile(null)
    setFileName(null)
    setPredictResult(null)
    setClipResult(null)
    setError(null)
    if (inputRef.current) inputRef.current.value = ''
  }, [])

  const runInference = useCallback(async () => {
    if (!file) return
    setLoading(true)
    setError(null)
    setPredictResult(null)
    setClipResult(null)

    try {
      const predictForm = new FormData()
      predictForm.append('file', file)
      const clipForm = new FormData()
      clipForm.append('file', file)

      const [predictRes, clipRes] = await Promise.all([
        fetch(`${API_BASE}/predict`, { method: 'POST', body: predictForm }),
        fetch(`${API_BASE}/clip`, { method: 'POST', body: clipForm }),
      ])

      if (!predictRes.ok) {
        throw new Error(predictRes.status === 429 ? 'rate limit reached, try again in a minute' : 'predict request failed')
      }
      if (!clipRes.ok) {
        throw new Error(clipRes.status === 429 ? 'rate limit reached, try again in a minute' : 'clip request failed')
      }

      const predictData: PredictResponse = await predictRes.json()
      const clipData: ClipResponse = await clipRes.json()

      setPredictResult(predictData)
      setClipResult(clipData)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'something went wrong')
    } finally {
      setLoading(false)
    }
  }, [file])

  const hasResults = predictResult !== null && clipResult !== null

  useEffect(() => {
    if (hasResults && resultsRef.current) {
      resultsRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' })
    }
  }, [hasResults])

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
            attention U-Net, the VAE, and both CLIP variants — all four results side by side, live
            on a Modal-hosted CPU endpoint.
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
                onClick={runInference}
                disabled={!file || loading}
                style={{
                  width: '100%',
                  display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem',
                  padding: '0.85rem',
                  background: (!file || loading) ? 'var(--glass-btn-bg)' : 'linear-gradient(135deg, #3A015C, #7B2FBE)',
                  border: (!file || loading) ? '1px solid var(--glass-btn-border)' : 'none',
                  borderRadius: '3px',
                  color: (!file || loading) ? 'var(--text-muted)' : '#fff',
                  fontFamily: 'Hanken Grotesk',
                  fontSize: '0.85rem',
                  fontWeight: 600,
                  letterSpacing: '0.05em',
                  cursor: (!file || loading) ? 'not-allowed' : 'pointer',
                  boxShadow: (!file || loading) ? 'none' : '0 4px 24px rgba(123,47,190,0.3)',
                }}
              >
                {loading ? (
                  <motion.span
                    animate={{ rotate: 360 }}
                    transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
                    style={{ display: 'flex' }}
                  >
                    <CircleNotch size={14} weight="bold" />
                  </motion.span>
                ) : (
                  <Lightning size={14} weight="bold" />
                )}
                {loading ? 'running inference' : 'run inference'}
              </button>
              <p className="text-subheading" style={{ textAlign: 'center', marginTop: '0.6rem' }}>
                {loading
                  ? 'cold start can take 10-30s on the first request'
                  : error
                  ? ''
                  : 'runs on a live modal-hosted cpu endpoint'}
              </p>
              {error && (
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', justifyContent: 'center', marginTop: '0.5rem' }}>
                  <Warning size={12} style={{ color: '#C084FC' }} weight="bold" />
                  <p className="text-subheading" style={{ color: '#C084FC' }}>{error}</p>
                </div>
              )}
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

      {/* result cards */}
      <section className="section" ref={resultsRef}>
        <motion.div {...fadeUp(0)}>
          <p className="text-subheading" style={{ marginBottom: '0.5rem' }}>Results</p>
        </motion.div>
        <motion.div {...fadeUp(0.06)}>
          <h2 style={{ marginBottom: '2rem' }}>
            <span className="text-heading" style={{ color: 'var(--text-primary)' }}>Four outputs, </span>
            <span className="text-heading font-display" style={{ fontStyle: 'italic', color: 'var(--text-primary)' }}>side by side.</span>
          </h2>
        </motion.div>

        <AnimatePresence mode="wait">
          {hasResults ? (
            <motion.div key="results" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
              <div
                className="result-cards-grid"
                style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1rem' }}
              >
              {/* classifier */}
              <div style={{ padding: '1.5rem', border: '1px solid var(--border)', borderRadius: '8px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', marginBottom: '1.25rem' }}>
                  <ChartBar size={18} style={{ color: 'var(--accent-light)' }} />
                  <p style={{ fontFamily: 'Hanken Grotesk', fontSize: '0.95rem', fontWeight: 600, color: 'var(--text-primary)' }}>Classifier</p>
                </div>
                {[
                  { label: 'Normal', value: predictResult!.classifier.normal, color: 'var(--accent-light)' },
                  { label: 'Pneumonia', value: predictResult!.classifier.pneumonia, color: '#9B6DCC' },
                ].map((row) => (
                  <div key={row.label} style={{ marginBottom: '0.9rem' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.35rem' }}>
                      <p className="text-subheading" style={{ fontSize: '0.75rem' }}>{row.label}</p>
                      <p style={{ fontFamily: 'Hanken Grotesk', fontSize: '0.9rem', fontWeight: 700, color: 'var(--text-primary)' }}>{(row.value * 100).toFixed(1)}%</p>
                    </div>
                    <div style={{ height: '5px', background: 'var(--border)', borderRadius: '2px', overflow: 'hidden' }}>
                      <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: `${row.value * 100}%` }}
                        transition={{ duration: 0.6 }}
                        style={{ height: '100%', background: row.color }}
                      />
                    </div>
                  </div>
                ))}
              </div>

              {/* unet */}
              <div style={{ padding: '1.5rem', border: '1px solid var(--border)', borderRadius: '8px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', marginBottom: '1.25rem' }}>
                  <Selection size={18} style={{ color: 'var(--accent-light)' }} />
                  <p style={{ fontFamily: 'Hanken Grotesk', fontSize: '0.95rem', fontWeight: 600, color: 'var(--text-primary)' }}>U-Net Heatmap</p>
                </div>
                <div style={{ position: 'relative', width: '100%', aspectRatio: '1', borderRadius: '6px', overflow: 'hidden', background: '#000' }}>
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={`data:image/png;base64,${predictResult!.unet.mask_png_base64}`}
                    alt="U-Net predicted opacity mask"
                    style={{ width: '100%', height: '100%', objectFit: 'contain' }}
                  />
                </div>
                <p className="text-subheading" style={{ fontSize: '0.72rem', marginTop: '0.65rem', textAlign: 'center' }}>
                  bright regions = predicted opacity
                </p>
              </div>

              {/* vae */}
              <div style={{ padding: '1.5rem', border: '1px solid var(--border)', borderRadius: '8px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', marginBottom: '1.25rem' }}>
                  <ArrowsClockwise size={18} style={{ color: 'var(--accent-light)' }} />
                  <p style={{ fontFamily: 'Hanken Grotesk', fontSize: '0.95rem', fontWeight: 600, color: 'var(--text-primary)' }}>VAE Reconstruction</p>
                </div>
                <div style={{ position: 'relative', width: '100%', aspectRatio: '1', borderRadius: '6px', overflow: 'hidden', background: '#000' }}>
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={`data:image/png;base64,${predictResult!.vae.reconstruction_png_base64}`}
                    alt="VAE reconstruction"
                    style={{ width: '100%', height: '100%', objectFit: 'contain' }}
                  />
                </div>
                <p className="text-subheading" style={{ fontSize: '0.72rem', marginTop: '0.65rem', textAlign: 'center' }}>
                  recon mse: {predictResult!.vae.reconstruction_mse.toFixed(4)}
                </p>
              </div>

              {/* clip */}
              <div style={{ padding: '1.5rem', border: '1px solid var(--border)', borderRadius: '8px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', marginBottom: '1.25rem' }}>
                  <Sparkle size={18} style={{ color: 'var(--accent-light)' }} />
                  <p style={{ fontFamily: 'Hanken Grotesk', fontSize: '0.95rem', fontWeight: 600, color: 'var(--text-primary)' }}>CLIP Scores</p>
                </div>
                {[
                  { label: 'OpenAI CLIP', value: clipResult!.openai_clip.pneumonia },
                  { label: 'BiomedCLIP', value: clipResult!.biomedclip.pneumonia },
                ].map((row) => (
                  <div key={row.label} style={{ marginBottom: '0.9rem' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.35rem' }}>
                      <p className="text-subheading" style={{ fontSize: '0.75rem' }}>{row.label}</p>
                      <p style={{ fontFamily: 'Hanken Grotesk', fontSize: '0.9rem', fontWeight: 700, color: 'var(--text-primary)' }}>{(row.value * 100).toFixed(1)}% pneumonia</p>
                    </div>
                    <div style={{ height: '5px', background: 'var(--border)', borderRadius: '2px', overflow: 'hidden' }}>
                      <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: `${row.value * 100}%` }}
                        transition={{ duration: 0.6 }}
                        style={{ height: '100%', background: '#9B6DCC' }}
                      />
                    </div>
                  </div>
                ))}
              </div>
              </div>
            </motion.div>
          ) : (
            <motion.div key="placeholders" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
              <div
                className="result-cards-grid"
                style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1rem' }}
              >
              {[
                { id: 'classifier', label: 'Classifier', icon: ChartBar, detail: 'Normal / Pneumonia probability' },
                { id: 'unet', label: 'U-Net Heatmap', icon: Selection, detail: 'Opacity localization overlay' },
                { id: 'vae', label: 'VAE Reconstruction', icon: ArrowsClockwise, detail: 'Recon error anomaly score' },
                { id: 'clip', label: 'CLIP Scores', icon: Sparkle, detail: 'Zero-shot vs BiomedCLIP' },
              ].map((card, i) => {
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
            </motion.div>
          )}
        </AnimatePresence>
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