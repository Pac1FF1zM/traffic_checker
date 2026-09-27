import { useEffect, useMemo, useRef, useState } from 'react'
import './App.css'
import member1 from './assets/team/member1.jpg'
import member2 from './assets/team/member2.jpg'
import member3 from './assets/team/member3.jpg'

const API_BASE = import.meta.env.VITE_API_URL || ''

const metrics = [
  ['84.55%', 'Test AUROC'],
  ['79.46%', 'Average Precision'],
  ['79.33%', 'Accuracy'],
  ['64.96%', 'F1 score'],
]

const eventNames = {
  accident: 'Accident risk', near_miss: 'Near miss', red_light: 'Red light',
  wrong_way: 'Wrong way', stopped_vehicle: 'Stopped vehicle', jaywalking: 'Jaywalking',
  failure_to_yield: 'Failure to yield', solid_line_crossing: 'Solid-line crossing', congestion: 'Congestion',
}

const pct = (value = 0) => `${Math.round(value * 100)}%`
const seconds = (value = 0) => `${Number(value).toFixed(1)} s`

function RiskChart({ timeline = [], duration = 0 }) {
  const points = useMemo(() => {
    if (!timeline.length || !duration) return ''
    return timeline.map((point) => `${(point.time / duration) * 100},${100 - point.risk * 100}`).join(' ')
  }, [timeline, duration])

  return (
    <div className="chart-shell">
      <div className="chart-scale"><span>100%</span><span>50%</span><span>0%</span></div>
      <svg className="risk-chart" viewBox="0 0 100 100" preserveAspectRatio="none" aria-label="Risk score over time">
        <defs><linearGradient id="riskFill" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#19c984" stopOpacity="0.34" /><stop offset="100%" stopColor="#19c984" stopOpacity="0" /></linearGradient></defs>
        <line x1="0" y1="50" x2="100" y2="50" className="chart-grid" />
        {points && <polygon points={`0,100 ${points} 100,100`} fill="url(#riskFill)" />}
        {points && <polyline points={points} className="chart-line" />}
      </svg>
      <div className="chart-time"><span>0:00</span><span>{seconds(duration)}</span></div>
    </div>
  )
}

function DemoSection() {
  const inputRef = useRef(null)
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState('')
  const [status, setStatus] = useState('idle')
  const [backend, setBackend] = useState(null)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')

  useEffect(() => {
    fetch(`${API_BASE}/api/health`).then((response) => response.json()).then(setBackend).catch(() => setBackend({ status: 'offline' }))
  }, [])
  useEffect(() => () => preview && URL.revokeObjectURL(preview), [preview])

  const chooseFile = (nextFile) => {
    if (!nextFile) return
    if (preview) URL.revokeObjectURL(preview)
    setFile(nextFile); setPreview(URL.createObjectURL(nextFile)); setResult(null); setError(''); setStatus('idle')
  }

  const analyze = async () => {
    if (!file) return
    setStatus('running'); setError('')
    const body = new FormData(); body.append('file', file)
    try {
      const response = await fetch(`${API_BASE}/api/analyze`, { method: 'POST', body })
      const payload = await response.json()
      if (!response.ok) throw new Error(payload.detail || 'Analysis failed')
      setResult(payload); setStatus('done')
    } catch (requestError) { setError(requestError.message); setStatus('error') }
  }

  const ready = backend?.status === 'ready'
  return (
    <section id="demo" className="section demo-section">
      <div className="section-heading"><div className="eyebrow">03 — LIVE DEMO</div><h2>Upload a clip. <span>See the risk.</span></h2><p>The final causal model sees only current and previous frames. Your video is processed locally and deleted immediately after analysis.</p></div>
      <div className="demo-status-row"><span className={`backend-status ${ready ? 'online' : 'offline'}`}><i />{ready ? `GPU ready · ${backend.device}` : 'Local GPU backend offline'}</span><span>Fixed-camera calibration · causal</span></div>
      <div className="demo-grid">
        <div className="upload-card">
          <div className={`drop-zone ${file ? 'has-file' : ''}`} onClick={() => inputRef.current?.click()} onDragOver={(event) => event.preventDefault()} onDrop={(event) => { event.preventDefault(); chooseFile(event.dataTransfer.files[0]) }} role="button" tabIndex={0} onKeyDown={(event) => event.key === 'Enter' && inputRef.current?.click()}>
            <input ref={inputRef} type="file" accept="video/*" hidden onChange={(event) => chooseFile(event.target.files[0])} />
            {preview ? <video src={preview} controls /> : <div className="upload-empty"><span>↑</span><strong>Drop a traffic video here</strong><small>MP4, MOV, AVI, MKV or WebM · up to 300 MB</small></div>}
          </div>
          <div className="file-row"><div><strong>{file?.name || 'No video selected'}</strong><span>{file ? `${(file.size / 1024 / 1024).toFixed(1)} MB` : 'Choose a short clip for the fastest demo'}</span></div><button className="primary-button" disabled={!file || !ready || status === 'running'} onClick={analyze}>{status === 'running' ? 'Analyzing…' : 'Analyze video'}</button></div>
          {status === 'running' && <div className="analysis-progress"><i /><span>Running detection, tracking and temporal risk inference on the GPU…</span></div>}
          {error && <div className="demo-error">{error}</div>}
        </div>
        <div className={`analysis-card ${result ? 'has-result' : ''}`}>
          {!result ? <div className="analysis-empty"><span>⌁</span><h3>Analysis will appear here</h3><p>Risk timeline, detected events and processing performance are generated from the uploaded video.</p></div> : <>
            <div className="analysis-header"><div><span>MAXIMUM RISK</span><strong>{pct(result.summary.max_risk)}</strong></div><div><span>AT</span><strong>{seconds(result.summary.max_risk_time)}</strong></div><div><span>EVENTS</span><strong>{result.summary.event_count}</strong></div></div>
            <RiskChart timeline={result.timeline} duration={result.video.duration} />
            <div className="event-list">{result.events.length ? result.events.map((event, index) => <div className="event-row" key={`${event.label}-${index}`}><i /><strong>{eventNames[event.label] || event.label}</strong><span>{seconds(event.start)} — {seconds(event.end)}</span></div>) : <div className="no-events">No thresholded traffic events detected.</div>}</div>
            <div className="runtime-row"><span>{result.video.frames.toLocaleString()} frames</span><span>{seconds(result.summary.processing_seconds)} processing</span><span>{result.summary.realtime_factor.toFixed(2)}× video duration</span>{result.summary.max_ttc_risk !== undefined && <span>max TTC {pct(result.summary.max_ttc_risk)}</span>}{result.summary.max_visual_change !== undefined && <span>max visual Δ {pct(result.summary.max_visual_change)}</span>}</div>
          </>}
        </div>
      </div>
    </section>
  )
}

const scrollToSection = (id) => document.getElementById(id)?.scrollIntoView({ behavior: 'smooth' })

function App() {
  return <div className="site">
    <header className="navbar"><div className="nav-inner"><button className="brand" onClick={() => scrollToSection('home')}><span className="brand-dot" />WIUT<span>CV</span></button><nav className="nav-links"><button onClick={() => scrollToSection('solution')}>Solution</button><button onClick={() => scrollToSection('demo')}>Live demo</button><button onClick={() => scrollToSection('results')}>Results</button><button onClick={() => scrollToSection('team')}>Team</button></nav><a href="https://github.com/Pac1FF1zM/traffic_checker" className="github-button" target="_blank" rel="noopener noreferrer">GitHub ↗</a></div></header>
    <main>
      <section id="home" className="hero section"><div className="hero-grid"><div className="hero-content"><div className="eyebrow">WIUT HACKATHON 2026</div><h1>Intelligent<span>Traffic Vision</span></h1><p className="hero-description">A causal computer-vision system that detects traffic events and anticipates dangerous situations from a fixed road camera.</p><div className="hero-buttons"><button className="primary-button" onClick={() => scrollToSection('demo')}>Try live demo <span>→</span></button><button className="secondary-button" onClick={() => scrollToSection('results')}>View verified results</button></div><div className="stats"><div className="stat"><strong>84.55%</strong><span>DADA held-out AUROC</span></div><div className="stat"><strong>64.96%</strong><span>DADA held-out F1</span></div><div className="stat"><strong>227</strong><span>DADA test clips</span></div></div></div><div className="camera-wrapper"><div className="camera-card"><div className="camera-header"><span>CAUSAL ANALYSIS</span><span className="live"><i /> LIVE CAPABLE</span></div><div className="road"><div className="road-line line-one" /><div className="road-line line-two" /><div className="car car-one"><span /></div><div className="car car-two"><span /></div><div className="car car-three"><span /></div><div className="detection detection-one"><label>VEHICLE</label></div><div className="detection detection-risk"><label>RISK</label></div></div><div className="camera-footer"><span>VideoMAE-S + YOLO11n/TTC</span><strong>Past frames only</strong></div></div></div></div></section>
      <section id="solution" className="section light-section"><div className="section-heading"><div className="eyebrow">01 — SOLUTION</div><h2>From video to <span>traffic intelligence.</span></h2><p>A compact pipeline combines object detection, deterministic tracking and a fine-tuned temporal transformer.</p></div><div className="feature-grid"><article className="feature-card"><div className="number">01</div><div className="feature-icon">◉</div><h3>Road-user detection</h3><p>YOLO11n identifies vehicles and pedestrians while a lightweight tracker preserves motion history.</p></article><article className="feature-card"><div className="number">02</div><div className="feature-icon">⌁</div><h3>Temporal understanding</h3><p>VideoMAE-S evaluates causal 16-frame windows. A fixed-camera calibration suppresses steady scene and traffic-density bias.</p></article><article className="feature-card"><div className="number">03</div><div className="feature-icon">△</div><h3>Explainable output</h3><p>Risk scores, event intervals and processing speed are returned as structured results.</p></article></div></section>
      <section className="section architecture-section"><div className="section-heading"><div className="eyebrow">02 — ARCHITECTURE</div><h2>One causal <span>pipeline.</span></h2></div><div className="pipeline">{[['01','Input video','Fixed CCTV or MP4'],['02','Detection','Road users'],['03','Temporal model','Past 16 frames'],['04','Risk output','Timeline & events']].map((item, index) => <div className="pipeline-fragment" key={item[0]}><div className="pipeline-step"><div className="pipeline-number">{item[0]}</div><h3>{item[1]}</h3><p>{item[2]}</p></div>{index < 3 && <div className="pipeline-line" />}</div>)}</div></section>
      <DemoSection />
      <section id="results" className="section light-section results-section"><div className="section-heading"><div className="eyebrow">04 — VERIFIED RESULTS</div><h2>Measured once. <span>Reported honestly.</span></h2><p>The temporal model was selected on validation AUROC, then evaluated once on 227 untouched DADA-2000 dashcam clips comprising 52,255 windows. Fixed-camera demo clips are cross-domain and are not included in these metrics.</p></div><div className="metric-grid">{metrics.map(([value,label]) => <article className="metric-card" key={label}><strong>{value}</strong><span>{label}</span></article>)}</div><div className="evidence-panel"><div><span>VALIDATION IMPROVEMENT</span><strong>+4.58 pp F1</strong><p>54.82% → 59.40% after full fine-tuning</p></div><div><span>DADA TEST PRECISION</span><strong>84.24%</strong><p>Measured at the fixed 0.5 threshold</p></div><div><span>TEST PROTOCOL</span><strong>Zero overlap</strong><p>Source-disjoint train, validation and held-out test</p></div></div></section>
      <section id="team" className="section team-section"><div className="section-heading"><div className="eyebrow">05 — TEAM</div><h2>The people behind <span>the system.</span></h2><p>Machine learning, systems integration and product presentation in one team.</p></div><div className="team-grid">{[[member1,'ML Developer','Model training, evaluation and video inference.','Machine Learning'],[member2,'System / Integration','Pipeline integration, testing and deployment.','Engineering'],[member3,'Website & Presentation','Interactive demo, visual narrative and delivery.','Product']].map(([photo,title,description,role], index) => <article className="team-card" key={title}><div className="team-number">0{index+1}</div><img src={photo} className="team-photo" alt={title} /><h3>{title}</h3><p>{description}</p><span className="team-role">{role}</span></article>)}</div><div className="final-card"><div><div className="eyebrow">WIUT HACKATHON 2026</div><h2>Turning road video into <span>actionable intelligence.</span></h2><p>Computer Vision Track · Traffic Event Detection & Accident Anticipation</p></div><button className="primary-button final-button" onClick={() => scrollToSection('demo')}>Open live demo →</button></div></section>
    </main>
    <footer><div><strong><span className="brand-dot" />WIUTCV</strong></div><p>WIUT Hackathon 2026 · Computer Vision Track</p></footer>
  </div>
}

export default App
