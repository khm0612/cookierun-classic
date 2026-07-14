package com.cookierun.bridge

import java.io.BufferedOutputStream
import java.io.BufferedReader
import java.io.DataOutputStream
import java.io.IOException
import java.io.InputStreamReader
import java.net.ServerSocket
import java.net.Socket
import java.security.MessageDigest

/**
 * Tiny line-based TCP server the PC brain connects to over Wi-Fi.
 *
 * Protocol (one request per line from the PC):
 *   AUTH token       -> first request; reply: "OK\n" or disconnect
 *   FRAME            -> reply: 4-byte big-endian length + JPEG bytes (length 0 = no frame yet)
 *   TAP x y          -> reply: "OK\n"   (x,y are floats in captured-frame pixels)
 *   HOLD x y ms      -> reply: "OK\n"
 *   PING             -> reply: "PONG\n"
 */
class BridgeServer(
    private val port: Int,
    private val token: String,
    private val getJpeg: () -> ByteArray?,
    private val getBounds: () -> Pair<Int, Int>,
    private val onTap: (Float, Float, Long) -> String,
    private val getInfo: () -> String,
    private val onGlobal: (String) -> Int,
    private val onProbe: (Int, Int) -> Unit,
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
        sock.soTimeout = CLIENT_TIMEOUT_MS
        val reader = BufferedReader(InputStreamReader(sock.getInputStream()))
        val out = DataOutputStream(BufferedOutputStream(sock.getOutputStream()))
        val auth = readCommand(reader)?.trim()?.split(Regex("\\s+")) ?: emptyList()
        val supplied = if (auth.size == 2 && auth[0] == "AUTH") auth[1] else ""
        if (!MessageDigest.isEqual(
                supplied.toByteArray(Charsets.UTF_8), token.toByteArray(Charsets.UTF_8))) {
            out.writeBytes("ERR auth\n")
            out.flush()
            return
        }
        out.writeBytes("OK\n")
        out.flush()
        while (running) {
            val line = readCommand(reader) ?: break
            val p = line.trim().split(Regex("\\s+"))
            when (p.getOrNull(0)) {
                "FRAME" -> {
                    if (p.size != 1) {
                        out.writeBytes("ERR bad_frame\n")
                        out.flush()
                        continue
                    }
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
                    val x = p.getOrNull(1)?.toFloatOrNull()
                    val y = p.getOrNull(2)?.toFloatOrNull()
                    val (width, height) = getBounds()
                    if (p.size != 3 || x == null || y == null || !x.isFinite() || !y.isFinite() ||
                        width <= 0 || height <= 0 || x < 0f || y < 0f ||
                        x >= width.toFloat() || y >= height.toFloat()) {
                        out.writeBytes("ERR bad_tap\n")
                    } else {
                        val r = onTap(x, y, 30L)
                        out.writeBytes("OK $r\n")
                    }
                    out.flush()
                }
                "HOLD" -> {
                    val x = p.getOrNull(1)?.toFloatOrNull()
                    val y = p.getOrNull(2)?.toFloatOrNull()
                    val ms = p.getOrNull(3)?.toLongOrNull()
                    val (width, height) = getBounds()
                    if (p.size != 4 || x == null || y == null || ms == null || !x.isFinite() ||
                        !y.isFinite() || width <= 0 || height <= 0 || x < 0f || y < 0f ||
                        x >= width.toFloat() || y >= height.toFloat() || ms !in 1..MAX_HOLD_MS) {
                        out.writeBytes("ERR bad_hold\n")
                    } else {
                        val r = onTap(x, y, ms)
                        out.writeBytes("OK $r\n")
                    }
                    out.flush()
                }
                "GLOBAL" -> {
                    val action = p.getOrNull(1)
                    if (p.size != 2 || action == null || action !in GLOBAL_ACTIONS) {
                        out.writeBytes("ERR bad_global\n")
                    } else {
                        out.writeBytes("OK ${onGlobal(action)}\n")
                    }
                    out.flush()
                }
                "PROBE" -> {
                    val x = p.getOrNull(1)?.toIntOrNull()
                    val y = p.getOrNull(2)?.toIntOrNull()
                    val (width, height) = getBounds()
                    if (p.size != 3 || x == null || y == null || width <= 0 || height <= 0 ||
                        x < 0 || y < 0 || x >= width || y >= height) {
                        out.writeBytes("ERR bad_probe\n")
                    } else {
                        onProbe(x, y)
                        out.writeBytes("OK\n")
                    }
                    out.flush()
                }
                "INFO" -> {
                    out.writeBytes(if (p.size == 1) getInfo() + "\n" else "ERR bad_info\n")
                    out.flush()
                }
                "PING" -> {
                    out.writeBytes(if (p.size == 1) "PONG\n" else "ERR bad_ping\n")
                    out.flush()
                }
                else -> { out.writeBytes("ERR\n"); out.flush() }
            }
        }
    }

    private fun readCommand(reader: BufferedReader): String? {
        val line = StringBuilder()
        val deadline = System.nanoTime() + CLIENT_TIMEOUT_MS * 1_000_000L
        while (true) {
            // Socket read timeouts reset after every byte. This absolute deadline also stops
            // a slow client from monopolising the server by dripping one byte at a time.
            if (System.nanoTime() > deadline) throw IOException("command timeout")
            val ch = reader.read()
            if (ch < 0) return if (line.isEmpty()) null else line.toString()
            if (ch == '\n'.code) return line.toString().trimEnd('\r')
            if (line.length >= MAX_COMMAND_CHARS) throw IOException("command too long")
            line.append(ch.toChar())
        }
    }

    companion object {
        private const val CLIENT_TIMEOUT_MS = 5_000
        private const val MAX_COMMAND_CHARS = 512
        private const val MAX_HOLD_MS = 10_000L
        private val GLOBAL_ACTIONS = setOf("BACK", "HOME", "SHADE")
    }
}
