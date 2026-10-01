package com.hawkeye.admin.alerts

import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Build
import androidx.core.app.ServiceCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleService
import androidx.lifecycle.lifecycleScope
import com.hawkeye.admin.AppGraph
import com.hawkeye.admin.data.Protocol
import com.hawkeye.admin.data.ServerMessage
import com.hawkeye.admin.domain.Backoff
import com.hawkeye.admin.domain.NotificationPolicy
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import okhttp3.Request
import okhttp3.Response
import okhttp3.WebSocket
import okhttp3.WebSocketListener

/**
 * 경보 서버와 WebSocket 연결을 유지한다 (앱이 뒤에 있어도).
 * 끊기면 간격을 늘려 다시 붙고, 붙을 때마다 놓친 사건을 다시 받는다 (docs/design.md §4·§6).
 */
class AlertService : LifecycleService() {
    private var loop: Job? = null
    private var socket: WebSocket? = null

    override fun onCreate() {
        super.onCreate()
        AppGraph.init(this)
        Notifications.createChannels(this)
        ServiceCompat.startForeground(
            this, Notifications.SERVICE_ID, Notifications.connection(this, "경보 서버에 연결 중"),
            if (Build.VERSION.SDK_INT >= 29) ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC else 0,
        )
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        super.onStartCommand(intent, flags, startId)
        restart()
        return START_STICKY
    }

    /** 설정이 바뀌면 다시 연결한다. */
    private fun restart() {
        loop?.cancel()
        socket?.cancel()
        loop = lifecycleScope.launch { runLoop() }
    }

    private suspend fun kotlinx.coroutines.CoroutineScope.runLoop() {
        val repo = AppGraph.alerts
        val backoff = Backoff()
        var reconnecting = false // 처음 연결 때는 기존 사건을 알리지 않는다 (알림 폭주 방지)
        while (isActive) {
            repo.setConnection(Connection.Connecting)
            updateServiceText("경보 서버에 연결 중")
            val api = repo.api()
            val closed = CompletableDeferred<String?>()
            socket = api.http.newWebSocket(
                Request.Builder().url(api.webSocketUrl()).build(),
                object : WebSocketListener() {
                    override fun onOpen(webSocket: WebSocket, response: Response) {
                        backoff.reset()
                        repo.setConnection(Connection.Connected)
                        updateServiceText("경보 서버 연결됨")
                        val announce = reconnecting
                        reconnecting = true
                        lifecycleScope.launch {
                            val missed = runCatching { repo.refresh() }.getOrDefault(emptyList())
                            if (announce) {
                                missed.forEach { Notifications.handle(this@AlertService, NotificationPolicy.forMissed(it)) }
                            }
                        }
                    }

                    override fun onMessage(webSocket: WebSocket, text: String) {
                        val m = runCatching { Protocol.parseMessage(text) }.getOrNull()
                        if (m is ServerMessage.CaseUpdated) {
                            repo.apply(m.case)
                            Notifications.handle(this@AlertService, NotificationPolicy.actionFor(m.notify, m.case))
                        }
                    }

                    override fun onClosed(webSocket: WebSocket, code: Int, reason: String) {
                        closed.complete(if (code == 1008) "API 키가 맞지 않습니다" else null)
                    }

                    override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                        closed.complete(t.message ?: t.javaClass.simpleName)
                    }
                },
            )
            val reason = closed.await()
            repo.setConnection(Connection.Disconnected(reason))
            updateServiceText("연결 끊김 — 다시 연결 중")
            delay(backoff.next())
        }
    }

    private fun updateServiceText(text: String) {
        runCatching {
            ContextCompat.getSystemService(this, android.app.NotificationManager::class.java)
                ?.notify(Notifications.SERVICE_ID, Notifications.connection(this, text))
        }
    }

    override fun onDestroy() {
        socket?.cancel()
        AppGraph.alerts.setConnection(Connection.Disconnected(null))
        super.onDestroy()
    }

    companion object {
        fun start(context: Context) {
            ContextCompat.startForegroundService(context, Intent(context, AlertService::class.java))
        }
    }
}
