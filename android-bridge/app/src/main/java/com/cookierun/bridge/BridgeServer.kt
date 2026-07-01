package com.cookierun.bridge

import java.io.BufferedOutputStream
import java.io.BufferedReader
import java.io.DataOutputStream
import java.io.InputStreamReader
import java.net.ServerSocket
import java.net.Socket

/**
 * Tiny line-based TCP server the PC brain connects to over Wi-Fi.
 *
 * Protocol (one request per line from the PC):
 *   FRAME            -> reply: 4-byte big-endian length + JPEG bytes (length 0 = no frame yet)
 *   TAP x y          -> reply: "OK\n"   (x,y are floats in captured-frame pixels)
 *   HOLD x y ms      -> reply: "OK\n"
 *   PING             -> reply: "PONG\n"
 */
class BridgeServer(
    private val port: Int,
    private val getJpeg: () -> ByteArray?,
    private val onTap: (Float, Float, Long) -> Unit,
) {
    @Volatile private var running = false
    private var thread: Thread? = null
    private var server: ServerSocket? = null

    fun start() {
        running = true
        thread = Thread({ serve() }, "bridge-server").apply { isDaemon = true; start() }
    }

    fun stop() {
        running = false
        try { server?.close() } catch (_: Exception) {}
        thread?.interrupt()
    }

    private fun serve() {
        val s = ServerSocket(port).also { it.reuseAddress = true }
        server = s
        while (running) {
            val sock = try {
                s.accept()
            } catch (e: Exception) {
                if (running) continue else break
            }
            try {
                handle(sock)
            } catch (_: Exception) {
            } finally {
                try { sock.close() } catch (_: Exception) {}
            }
        }
    }

    private fun handle(sock: Socket) {
        sock.tcpNoDelay = true
        val reader = BufferedReader(InputStreamReader(sock.getInputStream()))
        val out = DataOutputStream(BufferedOutputStream(sock.getOutputStream()))
        while (running) {
            val line = reader.readLine() ?: break
            val p = line.trim().split(" ")
            when (p.getOrNull(0)) {
                "FRAME" -> {
                    val jpeg = getJpeg()
                    if (jpeg == null) {
                        out.writeInt(0)
                    } else {
                        out.writeInt(jpeg.size)
                        out.write(jpeg)
                    }
                    out.flush()
                }
                "TAP" -> {
                    onTap(p[1].toFloat(), p[2].toFloat(), 30L)
                    out.writeBytes("OK\n"); out.flush()
                }
                "HOLD" -> {
                    onTap(p[1].toFloat(), p[2].toFloat(), p[3].toLong())
                    out.writeBytes("OK\n"); out.flush()
                }
                "PING" -> { out.writeBytes("PONG\n"); out.flush() }
                else -> { out.writeBytes("ERR\n"); out.flush() }
            }
        }
    }
}
