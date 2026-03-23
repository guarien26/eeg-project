import { ShieldCheck, TrendingUp, Brain, Timer } from "lucide-react"

const riskColors = {
  Low: "text-green-600",
  Moderate: "text-yellow-600",
  High: "text-orange-600",
  Critical: "text-red-600",
}

export default function MetricCards({
  riskLevel,
  stableTime,
  probability,
  confidence,
  latency,
}) {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
      {/* Seizure Risk Level */}
      <div className="bg-white border border-gray-200 rounded-xl p-4 flex justify-between items-start">
        <div>
          <p className="text-xs text-gray-500 font-medium">Seizure Risk Level</p>
          <p className={`text-2xl font-bold mt-1 ${riskColors[riskLevel] || "text-green-600"}`}>
            {riskLevel}
          </p>
          <p className="text-xs text-gray-400 mt-1">
            {stableTime > 0 ? `Stable for ${stableTime} min` : "No data yet"}
          </p>
        </div>
        <ShieldCheck className={`w-5 h-5 ${riskColors[riskLevel] || "text-green-500"}`} />
      </div>

      {/* Seizure Probability */}
      <div className="bg-white border border-gray-200 rounded-xl p-4 flex justify-between items-start">
        <div>
          <p className="text-xs text-gray-500 font-medium">Seizure Probability</p>
          <p className="text-2xl font-bold mt-1 text-gray-900">
            {probability !== null ? (probability * 100).toFixed(1) : "—"}
            <span className="text-sm font-normal text-gray-500"> %</span>
          </p>
          <p className="text-xs text-gray-400 mt-1">Current window</p>
        </div>
        <TrendingUp className={`w-5 h-5 ${
          probability > 0.7 ? "text-red-500" :
          probability > 0.3 ? "text-yellow-500" :
          "text-green-500"
        }`} />
      </div>

      {/* AI Confidence */}
      <div className="bg-white border border-gray-200 rounded-xl p-4 flex justify-between items-start">
        <div>
          <p className="text-xs text-gray-500 font-medium">AI Confidence</p>
          <p className="text-2xl font-bold mt-1 text-gray-900">
            {confidence}<span className="text-sm font-normal text-gray-500"> %</span>
          </p>
          <p className="text-xs text-gray-400 mt-1">
            {confidence >= 90 ? "High confidence" : confidence >= 70 ? "Moderate" : "Low confidence"}
          </p>
        </div>
        <Brain className="w-5 h-5 text-blue-500" />
      </div>

      {/* Inference Latency */}
      <div className="bg-white border border-gray-200 rounded-xl p-4 flex justify-between items-start">
        <div>
          <p className="text-xs text-gray-500 font-medium">Inference Latency</p>
          <p className="text-2xl font-bold mt-1 text-gray-900">
            {latency !== null ? latency : "—"}
            <span className="text-sm font-normal text-gray-500"> ms</span>
          </p>
          <p className="text-xs text-gray-400 mt-1">Per 4-sec window</p>
        </div>
        <Timer className="w-5 h-5 text-gray-400" />
      </div>
    </div>
  )
}
