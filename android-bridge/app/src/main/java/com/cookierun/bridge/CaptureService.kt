package com.cookierun.bridge

import android.app.Activity
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Intent
import android.content.pm.ServiceInfo
import android.graphics.Bitmap
import android.graphics.PixelFormat
import android.hardware.display.DisplayManager
import android.hardware.display.VirtualDisplay
import android.media.ImageReader
import android.media.projection.MediaProjection
import android.media.projection.MediaProjectionManager
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.view.WindowManager
import java.io.ByteArrayOutputStream

/** Foreground service: MediaProjection screen capture + the TCP bridge. */
class CaptureService : Service() {

    private var projection: MediaProjection? = null
    private var imageReader: ImageReader? = null
    private var virtualDisplay: VirtualDisplay? = null
    private var bridge: BridgeServer? = null
    private var vWidth = 0
    private var vHeight = 0

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        startForegroundNotif()
        val code = intent?.getIntExtra(EXTRA_CODE, Activity.RESULT_CANCELED) ?: Activity.RESULT_CANCELED
        @Suppress("DEPRECATION")
        val data = intent?.getParcelableExtra<Intent>(EXTRA_DATA)
        if (code == Activity.RESULT_OK && data != null) {
            startCapture(code, data)
        }
        return START_NOT_STICKY
    }

    private fun startForegroundNotif() {
        val nm = getSystemService(NotificationManager::class.java)
        nm.createNotificationChannel(
            NotificationChannel(CHANNEL, "CR Bridge", NotificationManager.IMPORTANCE_LOW)
        )
        val notif: Notification = Notification.Builder(this, CHANNEL)
            .setContentTitle("CR Bridge running")
            .setContentText("Screen capture + tap bridge active")
            .setSmallIcon(android.R.drawable.ic_menu_camera)
            .build()
        startForeground(1, notif, ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PROJECTION)
    }

    private fun startCapture(code: Int, data: Intent) {
        val mpm = getSystemService(MediaProjectionManager::class.java)
        val proj = mpm.getMediaProjection(code, data)
        projection = proj
        proj.registerCallback(object : MediaProjection.Callback() {
            override fun onStop() { cleanup() }
        }, Handler(Looper.getMainLooper()))

        val wm = getSystemService(WindowManager::class.java)
        val bounds = wm.maximumWindowMetrics.bounds
        vWidth = bounds.width()
        vHeight = bounds.height()
        val density = resources.displayMetrics.densityDpi

        val reader = ImageReader.newInstance(vWidth, vHeight, PixelFormat.RGBA_8888, 2)
        imageReader = reader
        virtualDisplay = proj.createVirtualDisplay(
            "cr_cap", vWidth, vHeight, density,
            DisplayManager.VIRTUAL_DISPLAY_FLAG_AUTO_MIRROR,
            reader.surface, null, null
        )

        bridge = BridgeServer(PORT, ::latestJpeg) { x, y, dur ->
            TapAccessibilityService.instance?.tap(x, y, dur)
        }.also { it.start() }
    }

    private fun latestJpeg(): ByteArray? {
        val reader = imageReader ?: return null
        val image = reader.acquireLatestImage() ?: return null
        return try {
            val plane = image.planes[0]
            val buffer = plane.buffer
            val pixelStride = plane.pixelStride
            val rowStride = plane.rowStride
            val rowPadding = rowStride - pixelStride * vWidth
            val bmpW = vWidth + rowPadding / pixelStride
            val bmp = Bitmap.createBitmap(bmpW, vHeight, Bitmap.Config.ARGB_8888)
            bmp.copyPixelsFromBuffer(buffer)
            val cropped = if (bmpW != vWidth) Bitmap.createBitmap(bmp, 0, 0, vWidth, vHeight) else bmp
            val out = ByteArrayOutputStream()
            cropped.compress(Bitmap.CompressFormat.JPEG, 70, out)
            out.toByteArray()
        } catch (e: Exception) {
            null
        } finally {
            image.close()
        }
    }

    private fun cleanup() {
        try { bridge?.stop() } catch (_: Exception) {}
        try { virtualDisplay?.release() } catch (_: Exception) {}
        try { imageReader?.close() } catch (_: Exception) {}
        try { projection?.stop() } catch (_: Exception) {}
        projection = null
        imageReader = null
        virtualDisplay = null
    }

    override fun onDestroy() {
        cleanup()
        super.onDestroy()
    }

    companion object {
        const val CHANNEL = "cr_bridge"
        const val PORT = 8080
        const val EXTRA_CODE = "code"
        const val EXTRA_DATA = "data"
    }
}
