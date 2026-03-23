import { Activity, Wifi, WifiOff } from "lucide-react"

export default function Header({ connected, isRecording, currentRisk }) {
  const noSeizures = currentRisk === "normal" || currentRisk === "elevated"

  return (
    <div className="flex items-start justify-between">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">
          Seizure Detection & Epilepsy Monitoring System
        </h1>
        <p className="text-sm text-gray-500 mt-1">
          AI-powered real-time EEG analysis for early seizure onset detection
        </p>
      </div>

      <div className="flex items-center gap-3">
        {noSeizures ? (
          <span className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium bg-green-50 text-green-700 border border-green-200">
            <span className="w-2 h-2 rounded-full bg-green-500" />
            No Seizures Detected
          </span>
        ) : (
          <span className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium bg-red-50 text-red-700 border border-red-200">
            <span className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
            Seizure Warning
          </span>
        )}

        <span className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium ${
          isRecording
            ? "bg-green-50 text-green-700 border border-green-200"
            : "bg-gray-50 text-gray-600 border border-gray-200"
        }`}>
          <Activity className="w-3 h-3" />
          {isRecording ? "Active Monitoring" : "Standby"}
        </span>

        <span className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium ${
          connected
            ? "bg-blue-50 text-blue-700 border border-blue-200"
            : "bg-gray-50 text-gray-500 border border-gray-200"
        }`}>
          {connected ? <Wifi className="w-3 h-3" /> : <WifiOff className="w-3 h-3" />}
          {connected ? "Server Connected" : "Demo Mode"}
        </span>
      </div>
    </div>
  )
}
