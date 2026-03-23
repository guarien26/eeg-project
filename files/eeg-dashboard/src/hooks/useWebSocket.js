import { useState, useEffect, useRef, useCallback } from "react"

const WS_URL = "ws://localhost:8765"

export function useWebSocket() {
  const [connected, setConnected] = useState(false)
  const [serverInfo, setServerInfo] = useState(null)
  const [latestPrediction, setLatestPrediction] = useState(null)
  const [latestWaveform, setLatestWaveform] = useState(null)
  const wsRef = useRef(null)
  const reconnectTimer = useRef(null)

  const connect = useCallback(() => {
    try {
      const ws = new WebSocket(WS_URL)

      ws.onopen = () => {
        setConnected(true)
        console.log("[WS] Connected to inference server")
      }

      ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data)
          if (msg.type === "server_info") {
            setServerInfo(msg.data)
          } else if (msg.type === "prediction") {
            setLatestPrediction(msg.data)
          } else if (msg.type === "waveform") {
            setLatestWaveform(msg.data)
          }
        } catch (e) {
          console.error("[WS] Parse error:", e)
        }
      }

      ws.onclose = () => {
        setConnected(false)
        wsRef.current = null
        // Try to reconnect every 3 seconds
        reconnectTimer.current = setTimeout(connect, 3000)
      }

      ws.onerror = () => {
        ws.close()
      }

      wsRef.current = ws
    } catch (e) {
      // Server not running, retry
      reconnectTimer.current = setTimeout(connect, 3000)
    }
  }, [])

  useEffect(() => {
    connect()
    return () => {
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current)
      if (wsRef.current) wsRef.current.close()
    }
  }, [connect])

  const send = useCallback((type, data = {}) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type, ...data }))
    }
  }, [])

  return {
    connected,
    serverInfo,
    latestPrediction,
    latestWaveform,
    send,
  }
}
