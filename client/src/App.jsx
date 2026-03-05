import { useState, useEffect, useRef } from 'react'
import './App.css'

const API = 'http://localhost:4010'

// ── Voice characteristic options ────────────────────────────────────

const GENDER = ['Male', 'Female', 'Androgynous']
const AGE = ['Young', '30s', '40s', 'Mature', 'Senior']
const ACCENT = ['American', 'British', 'Australian', 'Indian', 'Chinese', 'Japanese', 'German', 'French']
const REGISTER = ['Bass', 'Baritone', 'Tenor', 'Alto', 'Soprano']
const QUALITY = ['Warm', 'Clear', 'Raspy', 'Smooth', 'Bright', 'Crisp']
const ENERGY = ['Calm', 'Conversational', 'Energetic', 'Authoritative', 'Intimate']
const PACE = ['Slow', 'Medium', 'Fast']

function buildPrompt({ gender, age, accent, register, quality, energy, pace }) {
  const parts = []

  // Open with age + gender
  if (age || gender) {
    const a = age ? (age === '30s' ? 'in their 30s' : age === '40s' ? 'in their 40s' : age.toLowerCase()) : ''
    const g = gender ? gender.toLowerCase() : ''
    if (a && g) parts.push(`A ${a} ${g} voice`)
    else if (g) parts.push(`A ${g} voice`)
    else if (a) parts.push(`A ${a} voice`)
  } else {
    parts.push('A voice')
  }

  // Register
  if (register) {
    parts[0] += ` with a ${register.toLowerCase()} register`
  }

  // Quality
  if (quality) {
    parts.push(`${quality} tone`)
  }

  // Accent
  if (accent) {
    const accentMap = {
      'American': 'American English accent',
      'British': 'British English accent',
      'Australian': 'Australian English accent',
      'Indian': 'Indian English accent',
      'Chinese': 'Chinese-accented English',
      'Japanese': 'Japanese-accented English',
      'German': 'German-accented English',
      'French': 'French-accented English',
    }
    parts.push(accentMap[accent] || `${accent} accent`)
  }

  // Energy + pace
  if (energy) parts.push(`${energy.toLowerCase()} delivery`)
  if (pace) parts.push(`${pace.toLowerCase()} speaking pace`)

  // Join naturally
  if (parts.length <= 1) return parts[0] || ''
  const first = parts[0]
  const rest = parts.slice(1).join(', ')
  return `${first}. ${rest.charAt(0).toUpperCase() + rest.slice(1)}.`
}

// ── Pill selector component ────────────────────────────────────────

function PillSelector({ label, options, value, onChange, wide }) {
  return (
    <div className={`selector-row${wide ? ' wide' : ''}`}>
      <div className="selector-label">{label}</div>
      <div className="pills">
        {options.map(opt => (
          <button
            key={opt}
            className={`pill${value === opt ? ' active' : ''}`}
            onClick={() => onChange(value === opt ? null : opt)}
          >
            {opt}
          </button>
        ))}
      </div>
    </div>
  )
}

// ── Audio player for base64 samples ────────────────────────────────

function AudioPlayer({ audioB64 }) {
  const [url, setUrl] = useState(null)

  useEffect(() => {
    if (audioB64) {
      const bytes = Uint8Array.from(atob(audioB64), c => c.charCodeAt(0))
      const blob = new Blob([bytes], { type: 'audio/wav' })
      const blobUrl = URL.createObjectURL(blob)
      setUrl(blobUrl)
      return () => URL.revokeObjectURL(blobUrl)
    }
  }, [audioB64])

  return url ? <audio controls src={url} /> : null
}

// ── Save dialog ────────────────────────────────────────────────────

function SaveDialog({ open, onSave, onCancel }) {
  const [name, setName] = useState('')
  const [validationError, setValidationError] = useState(null)
  const inputRef = useRef(null)

  useEffect(() => {
    if (open) {
      setName('')
      setValidationError(null)
      setTimeout(() => inputRef.current?.focus(), 50)
    }
  }, [open])

  function handleSubmit(e) {
    e.preventDefault()
    const trimmed = name.trim()
    if (!trimmed) return
    if (!/^[a-zA-Z0-9_-]+$/.test(trimmed)) {
      setValidationError('Use only letters, numbers, hyphens, underscores.')
      return
    }
    onSave(trimmed)
  }

  if (!open) return null

  return (
    <>
      <div className="dialog-overlay" onClick={onCancel} />
      <div className="dialog">
        <div className="dialog-title">Save Voice</div>
        <form onSubmit={handleSubmit}>
          <input
            ref={inputRef}
            className="dialog-input"
            type="text"
            value={name}
            onChange={e => { setName(e.target.value); setValidationError(null) }}
            placeholder="e.g. radio-host-1"
            maxLength={64}
          />
          {validationError && <div className="dialog-error">{validationError}</div>}
          <div className="dialog-actions">
            <button type="button" className="btn-dialog-cancel" onClick={onCancel}>Cancel</button>
            <button type="submit" className="btn-dialog-save" disabled={!name.trim()}>Save</button>
          </div>
        </form>
      </div>
    </>
  )
}

// ── Main App ────────────────────────────────────────────────────────

function App() {
  const [theme, setTheme] = useState(() => {
    const saved = localStorage.getItem('vd-theme')
    return saved || 'system'
  })

  // Voice characteristics
  const [gender, setGender] = useState(null)
  const [age, setAge] = useState(null)
  const [accent, setAccent] = useState(null)
  const [register, setRegister] = useState(null)
  const [quality, setQuality] = useState(null)
  const [energy, setEnergy] = useState(null)
  const [pace, setPace] = useState(null)

  // Prompt (auto-generated from pills, editable)
  const [instruct, setInstruct] = useState('')
  const [promptEdited, setPromptEdited] = useState(false)

  const [text, setText] = useState('Hello there. I hope you\'re having a wonderful day. Let me tell you about something interesting I discovered recently.')
  const [samples, setSamples] = useState([])
  const [generating, setGenerating] = useState(false)
  const [savedVoices, setSavedVoices] = useState([])
  const [savingId, setSavingId] = useState(null)
  const [savingSample, setSavingSample] = useState(null) // sample pending name dialog
  const [error, setError] = useState(null)
  const [toast, setToast] = useState(null) // { message, type: 'success' | 'error' }

  // Apply theme
  useEffect(() => {
    if (theme === 'system') {
      document.documentElement.removeAttribute('data-theme')
    } else {
      document.documentElement.setAttribute('data-theme', theme)
    }
    localStorage.setItem('vd-theme', theme)
  }, [theme])

  // Auto-generate prompt when pills change (unless user has manually edited)
  useEffect(() => {
    if (!promptEdited) {
      setInstruct(buildPrompt({ gender, age, accent, register, quality, energy, pace }))
    }
  }, [gender, age, accent, register, quality, energy, pace, promptEdited])

  useEffect(() => { loadVoices() }, [])

  function cycleTheme() {
    setTheme(t => t === 'system' ? 'light' : t === 'light' ? 'dark' : 'system')
  }

  const themeIcon = theme === 'light' ? '☀' : theme === 'dark' ? '☾' : '◐'

  async function loadVoices() {
    try {
      const res = await fetch(`${API}/voices`)
      const data = await res.json()
      setSavedVoices(data.voices || [])
    } catch {}
  }

  async function generate() {
    if (!instruct.trim() || !text.trim()) return
    setGenerating(true)
    setError(null)
    setSamples([])

    const promises = Array.from({ length: 3 }, (_, i) =>
      fetch(`${API}/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ instruct: instruct.trim(), text: text.trim() }),
      })
        .then(res => {
          if (!res.ok) throw new Error(`Generation failed (${res.status})`)
          return res.json()
        })
        .then(data => ({ ...data, index: i }))
        .catch(err => ({ error: err.message, index: i }))
    )

    const results = await Promise.all(promises)
    setSamples(results)
    setGenerating(false)
  }

  async function saveVoice(name) {
    const sample = savingSample
    setSavingSample(null)
    if (!sample || !name) return

    setSavingId(sample.id)
    setError(null)
    try {
      const res = await fetch(`${API}/save`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, audio_b64: sample.audio_b64 }),
      })
      if (!res.ok) {
        const data = await res.json().catch(() => ({}))
        throw new Error(data.detail || `Save failed (${res.status})`)
      }
      await loadVoices()
    } catch (e) {
      setError(e.message)
    }
    setSavingId(null)
  }

  async function deleteVoice(name) {
    if (!confirm(`Delete voice "${name}"?`)) return
    try {
      await fetch(`${API}/voices/${name}`, { method: 'DELETE' })
      await loadVoices()
    } catch (e) {
      setError(e.message)
    }
  }

  function showToast(message, type = 'success') {
    setToast({ message, type })
    setTimeout(() => setToast(null), 4000)
  }

  async function reloadTTS() {
    try {
      const res = await fetch('http://localhost:4003/v1/voices/reload', { method: 'POST' })
      if (!res.ok) throw new Error(`Server returned ${res.status}`)
      const data = await res.json()
      showToast(`Reloaded: ${data.voice_count} voices (${data.custom_count} custom)`)
    } catch (e) {
      showToast(`TTS server reload failed: ${e.message}`, 'error')
    }
  }

  return (
    <div className="app">
      {/* Header */}
      <div className="header">
        <div>
          <h1>Voice Designer</h1>
          <div className="header-subtitle">Qwen3-TTS custom voice workshop</div>
        </div>
        <button className="theme-toggle" onClick={cycleTheme} title={`Theme: ${theme}`}>
          {themeIcon}
        </button>
      </div>

      <SaveDialog
        open={savingSample !== null}
        onSave={saveVoice}
        onCancel={() => setSavingSample(null)}
      />

      {toast && <div className={`toast toast-${toast.type}`}>{toast.message}</div>}

      {error && <div className="error-banner">{error}</div>}

      {/* Voice characteristics */}
      <div className="section">
        <div className="section-label">VOICE CHARACTERISTICS</div>
        <div className="selector-grid">
          <PillSelector label="Gender" options={GENDER} value={gender} onChange={setGender} />
          <PillSelector label="Age" options={AGE} value={age} onChange={setAge} />
          <PillSelector label="Register" options={REGISTER} value={register} onChange={setRegister} />
          <PillSelector label="Quality" options={QUALITY} value={quality} onChange={setQuality} />
          <PillSelector label="Energy" options={ENERGY} value={energy} onChange={setEnergy} />
          <PillSelector label="Pace" options={PACE} value={pace} onChange={setPace} />
          <PillSelector label="Accent" options={ACCENT} value={accent} onChange={setAccent} wide />
        </div>
      </div>

      {/* Prompt */}
      <div className="section">
        <div className="prompt-field">
          <div className="prompt-label-row">
            <div className="prompt-label">Voice description prompt</div>
            {promptEdited && (
              <button
                className="pill"
                style={{ fontSize: '0.625rem', padding: '2px 8px' }}
                onClick={() => { setPromptEdited(false) }}
              >
                Reset to auto
              </button>
            )}
          </div>
          <textarea
            value={instruct}
            onChange={e => { setInstruct(e.target.value); setPromptEdited(true) }}
            placeholder="Describe the voice you want to create..."
            rows={3}
          />
        </div>

        <div className="sample-text-field">
          <div className="prompt-label">Sample text</div>
          <textarea
            value={text}
            onChange={e => setText(e.target.value)}
            rows={2}
            style={{ marginTop: 6 }}
          />
        </div>

        <button
          className="btn-generate"
          onClick={generate}
          disabled={generating || !instruct.trim() || !text.trim()}
        >
          {generating ? 'GENERATING 3 SAMPLES...' : 'GENERATE 3 SAMPLES'}
        </button>
      </div>

      {/* Samples */}
      {samples.length > 0 && (
        <div className="section">
          <div className="section-label">SAMPLES</div>
          <div className="sample-grid">
            {samples.map((s, i) => (
              <div key={i} className="sample-card">
                <div className="sample-card-header">SAMPLE {i + 1}</div>
                {s.error ? (
                  <p className="sample-error">{s.error}</p>
                ) : (
                  <>
                    <AudioPlayer audioB64={s.audio_b64} />
                    <button
                      className="btn-save"
                      onClick={() => setSavingSample(s)}
                      disabled={savingId === s.id}
                    >
                      {savingId === s.id ? 'SAVING...' : 'SAVE VOICE'}
                    </button>
                  </>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Saved voices */}
      <hr className="divider" />
      <div className="section">
        <div className="saved-header">
          <div className="section-label">SAVED VOICES</div>
          <button className="btn-reload" onClick={reloadTTS}>
            Reload TTS Server
          </button>
        </div>
        {savedVoices.length === 0 ? (
          <p className="empty-state">No saved voices yet. Generate samples and save the ones you like.</p>
        ) : (
          <div className="voice-list">
            {savedVoices.map(v => (
              <div key={v.name} className="voice-item">
                <span className="voice-name">{v.name}</span>
                <audio controls src={`${API}/voices/${v.name}/audio`} />
                <button className="btn-delete" onClick={() => deleteVoice(v.name)}>Delete</button>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

export default App
