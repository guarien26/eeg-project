import { useMemo } from "react"
import {
  AreaChart, Area, XAxis, YAxis, ResponsiveContainer, ReferenceLine,
} from "recharts"

function ProbabilityTimeline({ predictions }) {
  const chartData = useMemo(() => {
    if (predictions.length === 0) return []
    const recent = predictions.slice(-60)
    return recent.map((p, i) => ({
      t: i,
      prob: Math.round(p.probability * 100),
      fill:
        p.probability >= 0.7 ? "#ef4444" :
        p.probability >= 0.3 ? "#f59e0b" :
        "#22c55e",
    }))
  }, [predictions])

  if (chartData.length === 0) {
    return (
      <div className="h-[140px] bg-gray-50 rounded-lg flex items-center justify-center">
        <p className="text-xs text-gray-400">Awaiting prediction data...</p>
      </div>
    )
  }

  return (
    <ResponsiveContainer width="100%" height={140}>
      <AreaChart data={chartData} margin={{ top: 5, right: 5, left: -20, bottom: 5 }}>
        <defs>
          <linearGradient id="probGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
            <stop offset="95%" stopColor="#3b82f6" stopOpacity={0.02} />
          </linearGradient>
        </defs>
        <XAxis
          dataKey="t"
          tick={false}
          axisLine={{ stroke: "#e5e7eb" }}
          tickLine={false}
        />
        <YAxis
          domain={[0, 100]}
          tick={{ fontSize: 10, fill: "#9ca3af" }}
          axisLine={false}
          tickLine={false}
          tickFormatter={(v) => `${v}%`}
        />
        <ReferenceLine
          y={70}
          stroke="#ef4444"
          strokeDasharray="4 4"
          strokeOpacity={0.5}
          label={{
            value: "Threshold",
            position: "right",
            fill: "#ef4444",
            fontSize: 9,
          }}
        />
        <Area
          type="monotone"
          dataKey="prob"
          stroke="#3b82f6"
          strokeWidth={1.5}
          fill="url(#probGrad)"
          isAnimationActive={false}
        />
      </AreaChart>
    </ResponsiveContainer>
  )
}

function AlertLog({ alerts }) {
  if (alerts.length === 0) {
    return (
      <div className="py-6 text-center">
        <p className="text-xs text-gray-400">No alerts yet</p>
      </div>
    )
  }

  const riskStyles = {
    critical: { bg: "bg-red-50", dot: "bg-red-500", text: "text-red-700" },
    warning: { bg: "bg-yellow-50", dot: "bg-yellow-500", text: "text-yellow-700" },
    elevated: { bg: "bg-amber-50", dot: "bg-amber-500", text: "text-amber-700" },
    normal: { bg: "bg-green-50", dot: "bg-green-500", text: "text-green-700" },
  }

  return (
    <div className="space-y-1.5 max-h-[200px] overflow-y-auto">
      {alerts.slice().reverse().slice(0, 20).map((alert, i) => {
        const style = riskStyles[alert.risk] || riskStyles.normal
        const time = new Date(alert.timestamp * 1000).toLocaleTimeString()
        return (
          <div
            key={i}
            className={`flex items-center gap-2 px-3 py-2 rounded-lg text-xs ${style.bg}`}
          >
            <span className={`w-1.5 h-1.5 rounded-full ${style.dot} flex-shrink-0`} />
            <span className={`font-medium ${style.text} w-16 flex-shrink-0`}>{time}</span>
            <span className={style.text}>
              Risk → {alert.risk.toUpperCase()} (prob: {(alert.probability * 100).toFixed(1)}%)
            </span>
          </div>
        )
      })}
    </div>
  )
}

export default function AnalysisPanel({
  predictions,
  alerts,
  isRecording,
}) {
  const hasData = predictions.length > 0
  const avgProb = hasData
    ? predictions.reduce((s, p) => s + p.probability, 0) / predictions.length
    : 0
  const avgLatency = hasData
    ? Math.round(
        predictions.reduce((s, p) => s + (p.latency_ms || 0), 0) / predictions.length
      )
    : 0
  const warningCount = predictions.filter((p) => p.probability > 0.7).length

  return (
    <div className="bg-white border border-gray-200 rounded-xl p-5 h-full flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-base font-semibold text-gray-900">
          Prediction Monitor
        </h2>
        <span className={`inline-flex items-center px-2.5 py-1 rounded-full text-[10px] font-semibold tracking-wide ${
          isRecording
            ? "bg-green-50 text-green-700 border border-green-200"
            : "bg-gray-50 text-gray-500 border border-gray-200"
        }`}>
          {isRecording ? "Live" : "Idle"}
        </span>
      </div>

      {/* Probability Timeline */}
      <div className="mb-4">
        <p className="text-xs font-medium text-gray-500 mb-2">
          Seizure probability over time
        </p>
        <ProbabilityTimeline predictions={predictions} />
        {hasData && (
          <div className="flex justify-between mt-1 text-[10px] text-gray-400">
            <span>{Math.max(0, predictions.length - 60)}s ago</span>
            <span>now</span>
          </div>
        )}
      </div>

      {/* Alert Log */}
      <div className="mb-4 flex-1">
        <p className="text-xs font-medium text-gray-500 mb-2">
          Alert log
        </p>
        <AlertLog alerts={alerts} />
      </div>

      {/* Session Stats */}
      <div className="pt-3 border-t border-gray-100">
        <p className="text-xs font-medium text-gray-500 mb-2">Session statistics</p>
        <div className="grid grid-cols-2 gap-2">
          <div className="bg-gray-50 rounded-lg px-3 py-2">
            <p className="text-[10px] text-gray-400">Windows analyzed</p>
            <p className="text-sm font-semibold text-gray-900">
              {predictions.length.toLocaleString()}
            </p>
          </div>
          <div className="bg-gray-50 rounded-lg px-3 py-2">
            <p className="text-[10px] text-gray-400">Avg probability</p>
            <p className="text-sm font-semibold text-gray-900">
              {hasData ? (avgProb * 100).toFixed(1) + "%" : "—"}
            </p>
          </div>
          <div className="bg-gray-50 rounded-lg px-3 py-2">
            <p className="text-[10px] text-gray-400">Avg latency</p>
            <p className="text-sm font-semibold text-gray-900">
              {hasData ? avgLatency + " ms" : "—"}
            </p>
          </div>
          <div className="bg-gray-50 rounded-lg px-3 py-2">
            <p className="text-[10px] text-gray-400">Warnings</p>
            <p className="text-sm font-semibold text-gray-900">{warningCount}</p>
          </div>
        </div>
      </div>
    </div>
  )
}
