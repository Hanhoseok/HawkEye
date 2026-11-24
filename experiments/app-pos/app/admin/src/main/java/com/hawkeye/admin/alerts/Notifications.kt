package com.hawkeye.admin.alerts

import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import androidx.core.app.NotificationCompat
import androidx.core.app.NotificationManagerCompat
import androidx.core.content.ContextCompat
import com.hawkeye.admin.MainActivity
import com.hawkeye.admin.R
import com.hawkeye.admin.domain.NotificationAction

object Notifications {
    const val CHANNEL_ALERTS = "alerts"
    const val CHANNEL_CONNECTION = "connection"
    const val SERVICE_ID = 1
    const val EXTRA_CASE_ID = "caseId"
    /** 알림의 '라이브' 단추: 열 카메라 */
    const val EXTRA_LIVE_CAMERA = "liveCamera"

    fun createChannels(context: Context) {
        val nm = context.getSystemService(NotificationManager::class.java)
        nm.createNotificationChannel(
            NotificationChannel(CHANNEL_ALERTS, "미결제 경보", NotificationManager.IMPORTANCE_HIGH)
                .apply { description = "결제 전 출구 접근, 일부만 결제, 결제 없이 퇴장" },
        )
        nm.createNotificationChannel(
            NotificationChannel(CHANNEL_CONNECTION, "서버 연결 상태", NotificationManager.IMPORTANCE_LOW),
        )
    }

    /** 늘 떠 있는 연결 상태 알림 (조용히). */
    fun connection(context: Context, text: String) =
        NotificationCompat.Builder(context, CHANNEL_CONNECTION)
            .setSmallIcon(R.drawable.ic_stat_hawkeye)
            .setColor(ContextCompat.getColor(context, R.color.brand))
            .setContentTitle(text)
            .setContentText("경보가 오면 바로 알려 드려요.")
            .setOngoing(true)
            .setShowWhen(false)
            .setContentIntent(openApp(context, null))
            .build()

    fun handle(context: Context, action: NotificationAction) {
        val nm = NotificationManagerCompat.from(context)
        when (action) {
            is NotificationAction.Show -> {
                if (!nm.areNotificationsEnabled()) return
                val n = NotificationCompat.Builder(context, CHANNEL_ALERTS)
                    .setSmallIcon(R.drawable.ic_stat_hawkeye)
                    .setColor(ContextCompat.getColor(context, R.color.brand))
                    .setContentTitle(action.title)
                    .setContentText(action.text)
                    .setStyle(NotificationCompat.BigTextStyle().bigText(action.text))
                    .setPriority(NotificationCompat.PRIORITY_HIGH)
                    .setCategory(NotificationCompat.CATEGORY_ALARM)
                    .setAutoCancel(true)
                    .setContentIntent(openApp(context, action.caseId))
                    .addAction(0, "영상 보기", openApp(context, action.caseId))
                    .addAction(0, "라이브", openLive(context, action.caseId, action.cameraId))
                    .build()
                try {
                    nm.notify(idFor(action.caseId), n)
                } catch (_: SecurityException) { /* 알림 권한 없음 */ }
            }
            is NotificationAction.Cancel -> nm.cancel(idFor(action.caseId))
            NotificationAction.None -> Unit
        }
    }

    private fun idFor(caseId: Int) = 1000 + caseId

    private fun openApp(context: Context, caseId: Int?): PendingIntent {
        val intent = Intent(context, MainActivity::class.java)
            .addFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP or Intent.FLAG_ACTIVITY_CLEAR_TOP)
        if (caseId != null) intent.putExtra(EXTRA_CASE_ID, caseId)
        return PendingIntent.getActivity(
            context, caseId ?: 0, intent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
        )
    }

    /** 요청 코드를 사건마다 따로(사건 상세 열기와 겹치지 않게) 줘서 서로 덮어쓰지 않는다. */
    private fun openLive(context: Context, caseId: Int, cameraId: String): PendingIntent {
        val intent = Intent(context, MainActivity::class.java)
            .addFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP or Intent.FLAG_ACTIVITY_CLEAR_TOP)
            .putExtra(EXTRA_LIVE_CAMERA, cameraId)
        return PendingIntent.getActivity(
            context, LIVE_REQUEST_BASE + caseId, intent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
        )
    }

    private const val LIVE_REQUEST_BASE = 1_000_000
}
