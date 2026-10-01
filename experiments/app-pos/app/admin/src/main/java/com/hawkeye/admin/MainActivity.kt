package com.hawkeye.admin

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.ContextCompat
import com.hawkeye.admin.alerts.AlertService
import com.hawkeye.admin.alerts.Notifications
import com.hawkeye.admin.ui.AppNav
import com.hawkeye.admin.ui.HawkEyeTheme
import kotlinx.coroutines.flow.MutableStateFlow

class MainActivity : ComponentActivity() {
    /** 알림을 눌러 들어오면 해당 사건 상세로 바로 간다. */
    private val openCase = MutableStateFlow<Int?>(null)

    private val askNotifications =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { /* 거절해도 앱은 동작 */ }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        AppGraph.init(this)
        Notifications.createChannels(this)
        openCase.value = intent.caseId()

        if (Build.VERSION.SDK_INT >= 33 &&
            ContextCompat.checkSelfPermission(this, Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED
        ) {
            askNotifications.launch(Manifest.permission.POST_NOTIFICATIONS)
        }
        AlertService.start(this)

        setContent { HawkEyeTheme { AppNav(openCase) } }
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        intent.caseId()?.let { openCase.value = it }
    }

    private fun Intent.caseId(): Int? =
        if (hasExtra(Notifications.EXTRA_CASE_ID)) getIntExtra(Notifications.EXTRA_CASE_ID, -1).takeIf { it > 0 } else null
}
