package com.cookierun.bridge

import android.app.Activity
import android.content.Intent
import android.media.projection.MediaProjectionManager
import android.os.Bundle
import android.provider.Settings
import android.view.ViewGroup
import android.widget.Button
import android.widget.LinearLayout
import android.widget.TextView
import java.net.Inet4Address
import java.net.NetworkInterface

/** Minimal UI: grant screen capture, enable accessibility, show the phone's IP:port. */
class MainActivity : Activity() {

    private lateinit var status: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(48, 96, 48, 48)
        }
        val ip = TextView(this).apply {
            text = "Phone IP: ${localIp()}   Port: ${CaptureService.PORT}"
            textSize = 18f
        }
        val btnCap = Button(this).apply {
            text = "1) Start screen capture"
            setOnClickListener { requestCapture() }
        }
        val btnAcc = Button(this).apply {
            text = "2) Enable accessibility (taps)"
            setOnClickListener { startActivity(Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS)) }
        }
        status = TextView(this).apply {
            text = "Do (1) then (2). Keep this app open while botting."
            textSize = 15f
        }
        val lp = LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT
        )
        root.addView(ip, lp)
        root.addView(btnCap, lp)
        root.addView(btnAcc, lp)
        root.addView(status, lp)
        setContentView(root)
    }

    private fun requestCapture() {
        val mpm = getSystemService(MediaProjectionManager::class.java)
        startActivityForResult(mpm.createScreenCaptureIntent(), REQ_CAPTURE)
    }

    @Deprecated("Deprecated in Java")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == REQ_CAPTURE && resultCode == Activity.RESULT_OK && data != null) {
            val svc = Intent(this, CaptureService::class.java)
                .putExtra(CaptureService.EXTRA_CODE, resultCode)
                .putExtra(CaptureService.EXTRA_DATA, data)
            startForegroundService(svc)
            status.text = "Capture ON. Bridge listening on ${localIp()}:${CaptureService.PORT}"
        } else if (requestCode == REQ_CAPTURE) {
            status.text = "Screen capture permission denied."
        }
    }

    private fun localIp(): String {
        try {
            for (nif in NetworkInterface.getNetworkInterfaces()) {
                if (!nif.isUp || nif.isLoopback) continue
                for (addr in nif.inetAddresses) {
                    if (addr is Inet4Address && addr.isSiteLocalAddress) {
                        return addr.hostAddress ?: continue
                    }
                }
            }
        } catch (_: Exception) {
        }
        return "unknown"
    }

    companion object {
        const val REQ_CAPTURE = 1001
    }
}
