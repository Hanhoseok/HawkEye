package com.hawkeye.admin.alerts

import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import androidx.core.app.NotificationCompat
import androidx.core.app.NotificationManagerCompat
import com.hawkeye.admin.MainActivity
import com.hawkeye.admin.domain.NotificationAction

object Notifications {
    const val CHANNEL_ALERTS = "alerts"
    const val CHANNEL_CONNECTION = "connection"
    const val SERVICE_ID = 1
    const val EXTRA_CASE_ID = "caseId"

    fun createChannels(context: Context) {
        val nm = context.getSystemService(NotificationManager::class.java)
        nm.createNotificationChannel(
            NotificationChannel(CHANNEL_ALERTS, "미결제 경보", NotificationManager.IMPORTANCE_HIGH)
                .apply { description = "출구 접근·미결제 퇴장·확인 필요" },
        )
        nm.createNotificationChannel(
            NotificationChannel(CHANNEL_CONNECTION, "서버 연결 상태", NotificationManager.IMPORTANCE_LOW),
        )
    }

    fun connection(context: Context, text: String) =
        NotificationCompat.Builder(context, CHANNEL_CONNECTION)
            .setSmallIcon(android.R.drawable.ic_menu_view)
            .setContentTitle("호크아이 관리자")
            .setContentText(text)
            .setOngoing(true)
            .setContentIntent(openApp(context, null))
            .build()

    fun handle(context: Context, action: NotificationAction) {
        val nm = NotificationManagerCompat.from(context)
        when (action) {
            is NotificationAction.Show -> {
                if (!nm.areNotificationsEnabled()) return
                val n = NotificationCompat.Builder(context, CHANNEL_ALERTS)
                    .setSmallIcon(android.R.drawable.ic_dialog_alert)
                    .setContentTitle(action.title)
                    .setContentText(action.text)
                    .setPriority(NotificationCompat.PRIORITY_HIGH)
                    .setCategory(NotificationCompat.CATEGORY_ALARM)
                    .setAutoCancel(true)
                    .setContentIntent(openApp(context, action.caseId))
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
}
