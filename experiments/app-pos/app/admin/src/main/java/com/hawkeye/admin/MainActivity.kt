package com.hawkeye.admin

import android.content.Intent
import android.graphics.Color
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.SystemBarStyle
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import com.hawkeye.admin.alerts.AlertService
import com.hawkeye.admin.alerts.Notifications
import com.hawkeye.admin.ui.AppNav
import com.hawkeye.admin.ui.theme.HawkEyeTheme
import kotlinx.coroutines.flow.MutableStateFlow

class MainActivity : ComponentActivity() {
    /** 알림을 눌러 들어오면 해당 사건 상세로 바로 간다. */
    private val openCase = MutableStateFlow<Int?>(null)
    /** 알림의 '라이브' 단추: 그 카메라 라이브로 간다. */
    private val openLive = MutableStateFlow<String?>(null)

    override fun onCreate(savedInstanceState: Bundle?) {
        // 밝은 화면만 쓰므로 상태 표시줄·내비게이션 바 아이콘은 항상 어둡게 (폰이 다크 모드여도)
        enableEdgeToEdge(
            statusBarStyle = SystemBarStyle.light(Color.TRANSPARENT, Color.TRANSPARENT),
            navigationBarStyle = SystemBarStyle.light(Color.TRANSPARENT, Color.TRANSPARENT),
        )
        super.onCreate(savedInstanceState)
        AppGraph.init(this)
        Notifications.createChannels(this)
        handle(intent)
        // 알림 권한은 여기서 바로 묻지 않는다: 시작 안내·서버 설정 뒤 경보 화면에서 한 번 (ui/AppNav)
        AlertService.start(this)

        setContent { HawkEyeTheme { AppNav(openCase, openLive) } }
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        handle(intent)
    }

    private fun handle(intent: Intent?) {
        intent ?: return
        intent.getStringExtra(Notifications.EXTRA_LIVE_CAMERA)?.let { openLive.value = it; return }
        if (intent.hasExtra(Notifications.EXTRA_CASE_ID)) {
            intent.getIntExtra(Notifications.EXTRA_CASE_ID, -1).takeIf { it > 0 }?.let { openCase.value = it }
        }
    }
}
