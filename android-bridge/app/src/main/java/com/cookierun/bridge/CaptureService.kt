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
import android.util.DisplayMetrics
import android.view.WindowManager
import java.io.ByteArrayOutputStream

/** Foreground service: MediaProjection screen capture + the TCP bridge. */
class CaptureService : Service() {

    private var projection: MediaProjection? = null
    private var imageReader: ImageReader? = null
    private var virtualDisplay: VirtualDisplay? = null
    private var bridge: BridgeServer? = null
    @Volatile private var vWidth = 0
    @Volatile private var vHeight = 0

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        startForegroundNotif()
        val code = intent?.getIntExtra(EXTRA_CODE, Activity.RESULT_CANCELED) ?: Activity.RESULT_CANCELED
        @Suppress("DEPRECATION")
        val data = intent?.getParcelableExtra<Intent>(EXTRA_DATA)
        val token = intent?.getStringExtra(EXTRA_TOKEN).orEmpty()
        if (code == Activity.RESULT_OK && data != null && token.isNotBlank()) {
            startCapture(code, data, token)
        } else {
            stopSelf(startId)
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

    private fun startCapture(code: Int, data: Intent, token: String) {
        if (projection != null || bridge != null) return
        val mpm = getSystemService(MediaProjectionManager::class.java)
        val proj = mpm.getMediaProjection(code, data)
        projection = proj
        proj.registerCallback(object : MediaProjection.Callback() {
            override fun onStop() { cleanup() }
        }, Handler(Looper.getMainLooper()))

        val (w, h) = currentSize()
        vWidth = w
        vHeight = h
        val density = resources.displayMetrics.densityDpi

        val reader = ImageReader.newInstance(vWidth, vHeight, PixelFormat.RGBA_8888, 2)
        imageReader = reader
        virtualDisplay = proj.createVirtualDisplay(
            "cr_cap", vWidth, vHeight, density,
            DisplayManager.VIRTUAL_DISPLAY_FLAG_AUTO_MIRROR,
            reader.surface, null, null
        )
        // Keep capture matched to the live display size/orientation (game is landscape),
        // so captured pixels == gesture coordinates (1:1, no letterbox).
        getSystemService(DisplayManager::class.java)
            .registerDisplayListener(displayListener, Handler(Looper.getMainLooper()))

        bridge = BridgeServer(
            PORT,
            token,
            ::latestJpeg,
            getBounds = { Pair(vWidth, vHeight) },
            onTap = { x, y, dur ->
                TapAccessibilityService.instance?.tap(x, y, dur) ?: "no_acc"
            },
            onGlobal = { cmd ->
                val action = when (cmd) {
                    "BACK" -> android.accessibilityservice.AccessibilityService.GLOBAL_ACTION_BACK
                    "HOME" -> android.accessibilityservice.AccessibilityService.GLOBAL_ACTION_HOME
                    "SHADE" -> android.accessibilityservice.AccessibilityService.GLOBAL_ACTION_NOTIFICATIONS
                    else -> -1
                }
                val svc = TapAccessibilityService.instance
                when {
                    action < 0 || svc == null -> -1
                    svc.global(action) -> 1
                    else -> 0
                }
            },
            onProbe = { x, y -> TapAccessibilityService.instance?.showDot(x, y) },
            getInfo = {
                val acc = TapAccessibilityService.instance != null
                val wm = getSystemService(WindowManager::class.java)
                @Suppress("DEPRECATION")
                val disp = wm.defaultDisplay
                val dm = android.util.DisplayMetrics()
                @Suppress("DEPRECATION")
                disp.getRealMetrics(dm)
                @Suppress("DEPRECATION")
                val rot = disp.rotation
                "acc=$acc capture=${vWidth}x${vHeight} real=${dm.widthPixels}x${dm.heightPixels} rot=$rot"
            },
        ).also { it.start() }
    }

    @Suppress("DEPRECATION")
    private fun currentSize(): Pair<Int, Int> {
        val disp = getSystemService(WindowManager::class.java).defaultDisplay
        val dm = DisplayMetrics()
        disp.getRealMetrics(dm)
        return Pair(dm.widthPixels, dm.heightPixels)
    }

    private val displayListener = object : DisplayManager.DisplayListener {
        override fun onDisplayAdded(displayId: Int) {}
        override fun onDisplayRemoved(displayId: Int) {}
        override fun onDisplayChanged(displayId: Int) {
            val (w, h) = currentSize()
            if (w != vWidth || h != vHeight) {
                val newReader = ImageReader.newInstance(w, h, PixelFormat.RGBA_8888, 2)
                virtualDisplay?.resize(w, h, resources.displayMetrics.densityDpi)
                virtualDisplay?.surface = newReader.surface
                imageReader?.close()
                imageReader = newReader
                vWidth = w
                vHeight = h
            }
        }
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
        try {
            getSystemService(DisplayManager::class.java).unregisterDisplayListener(displayListener)
        } catch (_: Exception) {}
        try { bridge?.stop() } catch (_: Exception) {}
        try { virtualDisplay?.release() } catch (_: Exception) {}
        try { imageReader?.close() } catch (_: Exception) {}
        try { projection?.stop() } catch (_: Exception) {}
        projection = null
        imageReader = null
        virtualDisplay = null
        bridge = null
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
        const val EXTRA_TOKEN = "token"
    }
}
