import { useState, useMemo } from "react"
import { LineChart, Line, XAxis, YAxis, ResponsiveContainer, ReferenceLine } from "recharts"

const CHANNEL_CONFIG = {
  Fp1: { label: "Fp1 (Frontal Left)", color: "#2563eb", group: "Frontal" },
  Fp2: { label: "Fp2 (Frontal Right)", color: "#3b82f6", group: "Frontal" },
  C3:  { label: "C3 (Central Left)", color: "#059669", group: "Central" },
  C4:  { label: "C4 (Central Right)", color: "#10b981", group: "Central" },
  T7:  { label: "T7 (Temporal Left)", color: "#d97706", group: "Temporal" },
  T8:  { label: "T8 (Temporal Right)", color: "#f59e0b", group: "Temporal" },
}

const TABS = ["All Channels", "Frontal", "Central", "Temporal", "Compare"]

function downsample(data, maxPoints = 500) {
  if (data.length <= maxPoints) return data
  const step = Math.ceil(data.length / maxPoints)
  const result = []
  for (let i = 0; i < data.length; i += step) {
    result.push(data[i])
  }
  return result
}

function ChannelChart({ name, data, color, label, timeWindow }) {
  const chartData = useMemo(() => {
    if (!data || data.length === 0) {
      return Array.from({ length: 100 }, (_, i) => ({
        t: ((i / 100) * timeWindow).toFixed(1),
        v: 0,
      }))
    }

    const ds = downsample(data, 500)
    return ds.map((v, i) => ({
      t: ((i / ds.length) * timeWindow).toFixed(1),
      v: Math.round(v * 100) / 100,
    }))
  }, [data, timeWindow])

  const yDomain = useMemo(() => {
    if (!data || data.length === 0) return [-50, 50]
    const vals = data.filter((v) => !isNaN(v))
    if (vals.length === 0) return [-50, 50]
    const max = Math.max(...vals.map(Math.abs))
    const bound = Math.max(Math.ceil(max / 10) * 10, 20)
    return [-bound, bound]
  }, [data])

  return (
    <div className="py-3">
      <p className="text-xs font-medium text-gray-600 mb-2 px-2">{label}</p>
      <ResponsiveContainer width="100%" height={120}>
        <LineChart data={chartData} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
          <XAxis
            dataKey="t"
            tick={{ fontSize: 10, fill: "#9ca3af" }}
            tickLine={false}
            axisLine={{ stroke: "#e5e7eb" }}
            interval="preserveStartEnd"
          />
          <YAxis
            domain={yDomain}
            tick={{ fontSize: 10, fill: "#9ca3af" }}
            tickLine={false}
            axisLine={false}
            width={35}
          />
          <ReferenceLine y={0} stroke="#e5e7eb" strokeDasharray="3 3" />
          <Line
            type="monotone"
            dataKey="v"
            stroke={color}
            strokeWidth={1.2}
            dot={false}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}

export default function WaveformPanel({
  waveformHistory,
  channelFilter,
  timeWindow,
  isRecording,
}) {
  const [activeTab, setActiveTab] = useState("All Channels")

  const visibleChannels = useMemo(() => {
    const tab = activeTab === "All Channels" ? channelFilter : activeTab
    if (tab === "All Channels") return Object.keys(CHANNEL_CONFIG)
    return Object.entries(CHANNEL_CONFIG)
      .filter(([_, cfg]) => cfg.group === tab)
      .map(([name]) => name)
  }, [activeTab, channelFilter])

  return (
    <div className="bg-white border border-gray-200 rounded-xl p-5">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-base font-semibold text-gray-900">
          EEG Channel Waveforms
        </h2>
        {isRecording && (
          <span className="flex items-center gap-1.5 text-xs text-green-600">
            <span className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse" />
            Live
          </span>
        )}
      </div>

      {/* Tabs */}
      <div className="flex gap-1 mb-2 border-b border-gray-100 pb-2">
        {TABS.map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors ${
              activeTab === tab
                ? "bg-gray-900 text-white"
                : "text-gray-500 hover:text-gray-700 hover:bg-gray-100"
            }`}
          >
            {tab}
          </button>
        ))}
      </div>

      {/* Channel Charts */}
      <div className="divide-y divide-gray-100">
        {activeTab === "Compare" ? (
          <CompareView waveformHistory={waveformHistory} timeWindow={timeWindow} />
        ) : (
          visibleChannels.map((ch) => (
            <ChannelChart
              key={ch}
              name={ch}
              data={waveformHistory[ch] || []}
              color={CHANNEL_CONFIG[ch].color}
              label={CHANNEL_CONFIG[ch].label}
              timeWindow={timeWindow}
            />
          ))
        )}
      </div>

      {!isRecording && Object.keys(waveformHistory).length === 0 && (
        <div className="text-center py-12 text-gray-400 text-sm">
          Press "Start Recording" to begin EEG monitoring
        </div>
      )}
    </div>
  )
}

function CompareView({ waveformHistory, timeWindow }) {
  const chartData = useMemo(() => {
    const maxLen = Math.max(
      ...Object.values(waveformHistory).map((d) => d?.length || 0),
      100
    )
    const ds = downsample(
      Array.from({ length: maxLen }, (_, i) => i),
      500
    )

    return ds.map((idx) => {
      const point = { t: ((idx / maxLen) * timeWindow).toFixed(1) }
      Object.keys(CHANNEL_CONFIG).forEach((ch) => {
        const data = waveformHistory[ch] || []
        point[ch] = data[idx] !== undefined ? Math.round(data[idx] * 100) / 100 : 0
      })
      return point
    })
  }, [waveformHistory, timeWindow])

  return (
    <div className="py-3">
      <p className="text-xs font-medium text-gray-600 mb-2 px-2">
        All channels overlaid
      </p>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={chartData} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
          <XAxis
            dataKey="t"
            tick={{ fontSize: 10, fill: "#9ca3af" }}
            tickLine={false}
            axisLine={{ stroke: "#e5e7eb" }}
            interval="preserveStartEnd"
          />
          <YAxis
            tick={{ fontSize: 10, fill: "#9ca3af" }}
            tickLine={false}
            axisLine={false}
            width={35}
          />
          <ReferenceLine y={0} stroke="#e5e7eb" strokeDasharray="3 3" />
          {Object.entries(CHANNEL_CONFIG).map(([ch, cfg]) => (
            <Line
              key={ch}
              type="monotone"
              dataKey={ch}
              stroke={cfg.color}
              strokeWidth={1}
              dot={false}
              isAnimationActive={false}
              opacity={0.7}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
