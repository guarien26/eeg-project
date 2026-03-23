import { useState, useEffect, useRef } from "react"

const SFREQ = 250
const WINDOW_SAMPLES = 1000
const CHANNEL_NAMES = ["Fp1", "Fp2", "C3", "C4", "T7", "T8"]

function generateFlatWaveform() {
  const channels = {}
  CHANNEL_NAMES.forEach((name) => {
    channels[name] = new Array(WINDOW_SAMPLES).fill(0)
  })
  return {
    timestamp: Date.now() / 1000,
    channels,
    sfreq: SFREQ,
  }
}

function generateNormalWaveform(t) {
  const channels = {}
  CHANNEL_NAMES.forEach((name, idx) => {
    const data = []
    const phase = idx * 0.7
    for (let i = 0; i < WINDOW_SAMPLES; i++) {
      const s = (t + i) / SFREQ
      const alpha = 15 * Math.sin(2 * Math.PI * 10 * s + phase)
      const theta = 8 * Math.sin(2 * Math.PI * 6 * s + phase * 1.3)
      const beta = 5 * Math.sin(2 * Math.PI * 20 * s + phase * 0.8)
      const noise = (Math.random() - 0.5) * 6
      data.push(alpha + theta + beta + noise)
    }
    channels[name] = data
  })
  return {
    timestamp: Date.now() / 1000,
    channels,
    sfreq: SFREQ,
  }
}

export function useDemoData(isRecording) {
  const [waveform, setWaveform] = useState(generateFlatWaveform())
  const [prediction, setPrediction] = useState(null)
  const timeRef = useRef(0)
  const windowCount = useRef(0)

  useEffect(() => {
    if (!isRecording) {
      setWaveform(generateFlatWaveform())
      setPrediction(null)
      return
    }

    const interval = setInterval(() => {
      timeRef.current += SFREQ
      windowCount.current += 1
      const wf = generateNormalWaveform(timeRef.current)
      setWaveform(wf)

      const prob = Math.random() * 0.12
      let risk = "normal"
      if (prob > 0.08) risk = "elevated"

      setPrediction({
        timestamp: Date.now() / 1000,
        probability: Math.round(prob * 10000) / 10000,
        risk_level: risk,
        threshold: 0.7,
        window_index: windowCount.current,
        latency_ms: Math.round(40 + Math.random() * 30),
      })
    }, 1000)

    return () => clearInterval(interval)
  }, [isRecording])

  return { waveform, prediction }
}
