import { Play, Square, RotateCcw, Download, ChevronDown } from "lucide-react"

export default function ControlBar({
  isRecording,
  onStart,
  onStop,
  onReset,
  onExport,
  timeWindow,
  onTimeWindowChange,
  channelFilter,
  onChannelFilterChange,
  windowCount,
}) {
  return (
    <div className="bg-white border border-gray-200 rounded-xl px-4 py-3 flex items-center justify-between">
      <div className="flex items-center gap-2">
        {!isRecording ? (
          <button
            onClick={onStart}
            className="inline-flex items-center gap-1.5 px-4 py-2 bg-gray-900 text-white text-sm font-medium rounded-lg hover:bg-gray-800 transition-colors"
          >
            <Play className="w-3.5 h-3.5" fill="currentColor" />
            Start Recording
          </button>
        ) : (
          <button
            onClick={onStop}
            className="inline-flex items-center gap-1.5 px-4 py-2 bg-red-600 text-white text-sm font-medium rounded-lg hover:bg-red-700 transition-colors"
          >
            <Square className="w-3.5 h-3.5" fill="currentColor" />
            Stop Recording
          </button>
        )}

        <button
          onClick={onReset}
          className="inline-flex items-center gap-1.5 px-3 py-2 text-sm text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded-lg transition-colors"
        >
          <RotateCcw className="w-3.5 h-3.5" />
          Reset
        </button>

        {windowCount > 0 && (
          <span className="text-xs text-gray-400 ml-2">
            {windowCount} windows analyzed
          </span>
        )}
      </div>

      <div className="flex items-center gap-3">
        {/* Time Window */}
        <div className="relative">
          <select
            value={timeWindow}
            onChange={(e) => onTimeWindowChange(Number(e.target.value))}
            className="appearance-none bg-gray-50 border border-gray-200 rounded-lg px-3 py-2 pr-8 text-sm text-gray-700 focus:outline-none focus:ring-2 focus:ring-blue-500 cursor-pointer"
          >
            <option value={5}>5 seconds</option>
            <option value={10}>10 seconds</option>
            <option value={30}>30 seconds</option>
            <option value={60}>60 seconds</option>
          </select>
          <ChevronDown className="w-3.5 h-3.5 absolute right-2.5 top-1/2 -translate-y-1/2 text-gray-400 pointer-events-none" />
        </div>

        {/* Channel Filter */}
        <div className="relative">
          <select
            value={channelFilter}
            onChange={(e) => onChannelFilterChange(e.target.value)}
            className="appearance-none bg-gray-50 border border-gray-200 rounded-lg px-3 py-2 pr-8 text-sm text-gray-700 focus:outline-none focus:ring-2 focus:ring-blue-500 cursor-pointer"
          >
            <option>All Channels</option>
            <option>Frontal</option>
            <option>Central</option>
            <option>Temporal</option>
          </select>
          <ChevronDown className="w-3.5 h-3.5 absolute right-2.5 top-1/2 -translate-y-1/2 text-gray-400 pointer-events-none" />
        </div>

        <div className="w-px h-6 bg-gray-200" />

        <button
          onClick={onExport}
          disabled={windowCount === 0}
          className="inline-flex items-center gap-1.5 px-3 py-2 text-sm text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded-lg transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
        >
          <Download className="w-3.5 h-3.5" />
          Export CSV
        </button>
      </div>
    </div>
  )
}
