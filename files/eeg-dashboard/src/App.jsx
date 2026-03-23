import { useState, useEffect, useCallback, useRef } from "react"
import { useWebSocket } from "./hooks/useWebSocket"
import { useDemoData } from "./hooks/useDemoData"
import Header from "./components/Header"
import MetricCards from "./components/MetricCards"
import ControlBar from "./components/ControlBar"
import WaveformPanel from "./components/WaveformPanel"
import AnalysisPanel from "./components/AnalysisPanel"

export default function App() {
  const [isRecording, setIsRecording] = useState(false)
  const [timeWindow, setTimeWindow] = useState(10)
  const [channelFilter, setChannelFilter] = useState("All Channels")
  const [predictions, setPredictions] = useState([])
  const [alerts, setAlerts] = useState([])
  const [waveformHistory, setWaveformHistory] = useState({})
  const lastRisk = useRef("normal")

  // WebSocket connection (auto-reconnects)
  const ws = useWebSocket()

  // Demo data fallback when server isn't connected
  const demo = useDemoData(isRecording && !ws.connected)

  // Current data source
  const currentPrediction = ws.connected ? ws.latestPrediction : demo.prediction
  const currentWaveform = ws.connected ? ws.latestWaveform : demo.waveform

  // Track prediction history and alerts
  useEffect(() => {
    if (!currentPrediction) return
    setPredictions((prev) => {
      const next = [...prev, currentPrediction]
      if (next.length > 600) return next.slice(-600)
      return next
    })

    // Log risk level changes as alerts
    const newRisk = currentPrediction.risk_level
    if (newRisk !== lastRisk.current) {
      setAlerts((prev) => [
        ...prev.slice(-50),
        {
          timestamp: currentPrediction.timestamp,
          risk: newRisk,
          probability: currentPrediction.probability,
        },
      ])
      lastRisk.current = newRisk
    }
  }, [currentPrediction])

  // Track waveform history
  useEffect(() => {
    if (!currentWaveform || !currentWaveform.channels) return
    setWaveformHistory((prev) => {
      const next = { ...prev }
      const maxSamples = timeWindow * 250

      Object.entries(currentWaveform.channels).forEach(([ch, data]) => {
        const existing = next[ch] || []
        const combined = [...existing, ...data]
        next[ch] = combined.slice(-maxSamples)
      })
      return next
    })
  }, [currentWaveform, timeWindow])

  const handleStartRecording = useCallback(() => {
    setIsRecording(true)
    setPredictions([])
    setAlerts([])
    setWaveformHistory({})
    lastRisk.current = "normal"
    if (ws.connected) ws.send("start_recording")
  }, [ws])

  const handleStopRecording = useCallback(() => {
    setIsRecording(false)
    if (ws.connected) ws.send("stop_recording")
  }, [ws])

  const handleReset = useCallback(() => {
    setIsRecording(false)
    setPredictions([])
    setAlerts([])
    setWaveformHistory({})
    lastRisk.current = "normal"
    if (ws.connected) ws.send("stop_recording")
  }, [ws])

  const handleExport = useCallback(() => {
    if (predictions.length === 0) return

    const ch = ["Fp1", "Fp2", "C3", "C4", "T7", "T8"]
    const bands = ["delta", "theta", "alpha", "beta", "gamma"]

    // Session metadata header
    const meta = [
      `# EEG Seizure Prediction Report`,
      `# Generated: ${new Date().toISOString()}`,
      `# Model: CNN-LSTM v3 (CHB-MIT trained)`,
      `# Channels: ${ch.join(", ")} (10-20 system)`,
      `# Sample Rate: 250 Hz | Window: 4 sec | Prediction Threshold: 0.7`,
      `# Risk Levels: normal (<0.3) | elevated (0.3-0.6) | warning (0.6-0.8) | critical (>0.8)`,
      `# Duration: ${Math.round(predictions.length)} seconds (${(predictions.length / 60).toFixed(1)} min)`,
      `# Total Warnings: ${predictions.filter((p) => p.probability > 0.7).length}`,
      `# Max Probability: ${Math.max(...predictions.map((p) => p.probability)).toFixed(4)}`,
      `#`,
    ].join("\n")

    // Column headers
    const chStatCols = ch.flatMap((c) => [`${c}_mean_uv`, `${c}_std_uv`, `${c}_peak_uv`])
    const bandCols = ch.flatMap((c) => bands.map((b) => `${c}_${b}`))
    const header = [
      "timestamp", "window", "probability", "risk_level", "latency_ms",
      ...chStatCols,
      ...bandCols,
    ].join(",")

    // Data rows
    const rows = predictions.map((p) => {
      const ts = new Date(p.timestamp * 1000).toISOString()
      const base = [ts, p.window_index, p.probability, p.risk_level, p.latency_ms || ""]

      const stats = ch.map((c) => {
        const s = p.channels?.[c]
        return s ? [s.mean_uv ?? "", s.std_uv ?? "", s.peak_uv ?? ""] : ["", "", ""]
      }).flat()

      const bp = ch.map((c) => {
        const b = p.band_powers?.[c]
        return b ? bands.map((bn) => b[bn] ?? "") : bands.map(() => "")
      }).flat()

      return [...base, ...stats, ...bp].join(",")
    }).join("\n")

    const csv = meta + "\n" + header + "\n" + rows
    const blob = new Blob([csv], { type: "text/csv" })
    const url = URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    a.download = `eeg_session_${new Date().toISOString().slice(0, 19).replace(/:/g, "-")}.csv`
    a.click()
    URL.revokeObjectURL(url)
  }, [predictions])

  // Compute metrics
  const currentRisk = currentPrediction?.risk_level || "normal"
  const currentProb = currentPrediction?.probability ?? null
  const currentLatency = currentPrediction?.latency_ms
    ? Math.round(currentPrediction.latency_ms)
    : null

  const confidence = currentPrediction
    ? Math.round(
        (1 - Math.abs(currentPrediction.probability - (currentPrediction.probability >= 0.5 ? 1 : 0))) * 100
      )
    : 97

  const riskLabel = {
    normal: "Low",
    elevated: "Moderate",
    warning: "High",
    critical: "Critical",
  }[currentRisk] || "Low"

  // How long since last non-normal alert
  const stableMinutes = predictions.length > 0
    ? Math.round((predictions.length * 1) / 60 * 10) / 10
    : 0

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-[1440px] mx-auto space-y-4">
        <Header
          connected={ws.connected}
          isRecording={isRecording}
          currentRisk={currentRisk}
        />

        <MetricCards
          riskLevel={riskLabel}
          stableTime={stableMinutes}
          probability={currentProb}
          confidence={confidence}
          latency={currentLatency}
        />

        <ControlBar
          isRecording={isRecording}
          onStart={handleStartRecording}
          onStop={handleStopRecording}
          onReset={handleReset}
          onExport={handleExport}
          timeWindow={timeWindow}
          onTimeWindowChange={setTimeWindow}
          channelFilter={channelFilter}
          onChannelFilterChange={setChannelFilter}
          windowCount={predictions.length}
        />

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          <div className="lg:col-span-2">
            <WaveformPanel
              waveformHistory={waveformHistory}
              channelFilter={channelFilter}
              timeWindow={timeWindow}
              isRecording={isRecording}
            />
          </div>
          <div className="lg:col-span-1">
            <AnalysisPanel
              predictions={predictions}
              alerts={alerts}
              isRecording={isRecording}
            />
          </div>
        </div>
      </div>
    </div>
  )
}
