package com.cookierun.bridge

import android.accessibilityservice.AccessibilityService
import android.accessibilityservice.GestureDescription
import android.graphics.Color
import android.graphics.Path
import android.graphics.PixelFormat
import android.os.Handler
import android.os.Looper
import android.view.Gravity
import android.view.View
import android.view.WindowManager
import android.view.accessibility.AccessibilityEvent
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

/** Injects taps/holds via the accessibility gesture API (no root, no dev tools). */
class TapAccessibilityService : AccessibilityService() {

    override fun onServiceConnected() {
        instance = this
    }

    override fun onAccessibilityEvent(event: AccessibilityEvent?) {}

    override fun onInterrupt() {}

    override fun onDestroy() {
        if (instance === this) instance = null
        super.onDestroy()
    }

    private val mainHandler = Handler(Looper.getMainLooper())

    /** Tap/hold at screen coords, dispatched on the main thread with a result callback.
     * Returns "completed" | "cancelled" | "dispatch_false" | "timeout" for diagnostics. */
    fun tap(x: Float, y: Float, durationMs: Long = 30L): String {
        val path = Path().apply { moveTo(x, y) }
        val stroke = GestureDescription.StrokeDescription(path, 0L, durationMs.coerceAtLeast(1L))
        val gesture = GestureDescription.Builder().addStroke(stroke).build()
        val latch = CountDownLatch(1)
        var result = "dispatch_false"
        val cb = object : GestureResultCallback() {
            override fun onCompleted(g: GestureDescription?) { result = "completed"; latch.countDown() }
            override fun onCancelled(g: GestureDescription?) { result = "cancelled"; latch.countDown() }
        }
        mainHandler.post {
            if (!dispatchGesture(gesture, cb, mainHandler)) {
                result = "dispatch_false"; latch.countDown()
            }
        }
        if (!latch.await(2, TimeUnit.SECONDS)) return "timeout"
        return result
    }

    /** Coordinate-free system action (BACK/HOME/notification shade). */
    fun global(action: Int): Boolean = performGlobalAction(action)

    /** Draw a marker at coordinate (x,y) so we can SEE where the accessibility layer
     * places that coordinate on screen — used to calibrate the capture->tap mapping. */
    fun showDot(x: Int, y: Int) {
        mainHandler.post {
            try {
                val wm = getSystemService(WindowManager::class.java)
                val dot = View(this).apply { setBackgroundColor(Color.RED) }
                val s = 60
                val lp = WindowManager.LayoutParams(
                    s, s,
                    WindowManager.LayoutParams.TYPE_ACCESSIBILITY_OVERLAY,
                    WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE or
                        WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE,
                    PixelFormat.TRANSLUCENT,
                )
                lp.gravity = Gravity.TOP or Gravity.START
                lp.x = x - s / 2
                lp.y = y - s / 2
                wm.addView(dot, lp)
                mainHandler.postDelayed({ try { wm.removeView(dot) } catch (_: Exception) {} }, 2500)
            } catch (_: Exception) {
            }
        }
    }

    companion object {
        @Volatile
        var instance: TapAccessibilityService? = null
    }
}
