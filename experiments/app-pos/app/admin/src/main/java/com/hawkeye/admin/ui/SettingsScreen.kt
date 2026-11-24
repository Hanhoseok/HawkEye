package com.hawkeye.admin.ui

import android.content.Context
import android.content.Intent
import android.provider.Settings
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.CheckCircle
import androidx.compose.material.icons.rounded.Error
import androidx.compose.material.icons.rounded.Key
import androidx.compose.material.icons.rounded.WifiOff
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.text.input.VisualTransformation
import androidx.compose.ui.unit.dp
import androidx.core.app.NotificationManagerCompat
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import com.hawkeye.admin.AppGraph
import com.hawkeye.admin.alerts.AlertService
import com.hawkeye.admin.data.ApiClient
import com.hawkeye.admin.data.ServerSettings
import com.hawkeye.admin.domain.CheckResult
import com.hawkeye.admin.domain.SettingsCheck
import com.hawkeye.admin.ui.components.AppBar
import com.hawkeye.admin.ui.components.ButtonKind
import com.hawkeye.admin.ui.components.Group
import com.hawkeye.admin.ui.components.HawkButton
import com.hawkeye.admin.ui.components.Note
import com.hawkeye.admin.ui.components.RowDivider
import com.hawkeye.admin.ui.components.ScreenTitle
import com.hawkeye.admin.ui.components.SectionLabel
import com.hawkeye.admin.ui.theme.Hawk
import com.hawkeye.admin.ui.theme.HawkType
import kotlinx.coroutines.launch

/**
 * 설정: 경보 서버 주소·API 키, 연결 확인(이유별 안내), 저장. 앱 줄: 알림 권한 상태, 버전.
 * onBack = null 이면 아래 탭 화면. firstRun = 시작 안내에서 들어옴 (저장하면 경보 화면으로).
 */
@Composable
fun SettingsScreen(onBack: (() -> Unit)?, firstRun: Boolean = false) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val current = AppGraph.settings.settings.value
    var url by remember { mutableStateOf(if (firstRun) "http://" else current.baseUrl) }
    var key by remember { mutableStateOf(current.apiKey.orEmpty()) }
    var checking by remember { mutableStateOf(false) }
    var result by remember { mutableStateOf<CheckResult?>(null) }
    var saved by remember { mutableStateOf(false) }
    val c = Hawk.colors

    Column(Modifier.fillMaxSize().then(if (firstRun) Modifier.background(c.bg).safeDrawingPadding() else Modifier)
        .verticalScroll(rememberScrollState()).imePadding()) {
        if (onBack != null) AppBar(if (firstRun) "서버 연결" else "설정", onBack) else ScreenTitle("설정")
        Column(Modifier.padding(horizontal = 16.dp).padding(bottom = 24.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
            SectionLabel("경보 서버")
            Field("서버 주소", url, { url = it; result = null; saved = false }, error = result == CheckResult.Unreachable, keyboard = KeyboardType.Uri)
            Field("API 키", key, { key = it; result = null; saved = false }, error = result == CheckResult.BadKey, secret = true)
            Text("렌탈 서버의 Tailscale 주소예요 (예: http://100.x.x.x:8000). 폰에서 Tailscale을 켜야 연결돼요.",
                style = HawkType.caption, color = c.faint, modifier = Modifier.padding(horizontal = 4.dp))
            when (val r = result) {
                null -> Unit
                is CheckResult.Ok -> Note(Icons.Rounded.CheckCircle, c.ok, c.okTint, title = "연결됐어요",
                    text = "카메라 ${r.cameras}대, 녹화 재생 ${if (r.recordings) "켜짐" else "꺼짐"}")
                CheckResult.Unreachable -> Note(Icons.Rounded.WifiOff, c.danger, c.dangerTint, title = "서버에 닿지 않아요",
                    text = "폰에서 Tailscale이 켜져 있는지, 주소가 맞는지 확인해 주세요.")
                CheckResult.BadKey -> Note(Icons.Rounded.Key, c.review, c.reviewTint, title = "API 키가 맞지 않아요",
                    text = "서버 deploy/.env 에 적힌 키를 넣어 주세요.")
                is CheckResult.Failed -> Note(Icons.Rounded.Error, c.danger, c.dangerTint, title = "연결하지 못했어요", text = r.message)
            }
            if (saved) Note(Icons.Rounded.CheckCircle, c.ok, c.okTint, text = "저장했어요. 새 설정으로 다시 연결해요.")
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                HawkButton(if (checking) "확인 중…" else "연결 확인", enabled = !checking, kind = ButtonKind.WHITE, modifier = Modifier.weight(1f), onClick = {
                    checking = true
                    scope.launch {
                        result = runCatching { ApiClient(ServerSettings(url.trim().trimEnd('/'), key.ifBlank { null })).config() }
                            .fold({ CheckResult.Ok(it.rtspStreams().size, it.recordings) }, { SettingsCheck.classify(it) })
                        checking = false
                    }
                })
                HawkButton("저장", kind = ButtonKind.BRAND, modifier = Modifier.weight(1f), onClick = {
                    AppGraph.settings.save(url, key)
                    AlertService.start(context) // 새 설정으로 다시 연결
                    saved = true
                })
            }
            if (!firstRun) {
                SectionLabel("앱", Modifier.padding(top = 8.dp))
                AppRows(context)
            }
        }
    }
}

@Composable
private fun Field(
    label: String, value: String, onChange: (String) -> Unit, error: Boolean,
    secret: Boolean = false, keyboard: KeyboardType = KeyboardType.Text,
) {
    val c = Hawk.colors
    Column(
        Modifier.fillMaxWidth().clip(RoundedCornerShape(16.dp)).background(c.surf)
            .then(if (error) Modifier.border(1.5.dp, c.danger, RoundedCornerShape(16.dp)) else Modifier)
            .padding(horizontal = 16.dp, vertical = 11.dp),
    ) {
        Text(label, style = HawkType.tiny.copy(fontSize = HawkType.caption.fontSize.times(0.92f)), color = c.faint)
        BasicTextField(
            value, onChange, singleLine = true,
            textStyle = HawkType.body.copy(color = c.text, fontWeight = HawkType.label.fontWeight?.let { androidx.compose.ui.text.font.FontWeight.Medium }),
            cursorBrush = SolidColor(c.brand),
            visualTransformation = if (secret) PasswordVisualTransformation() else VisualTransformation.None,
            keyboardOptions = KeyboardOptions(keyboardType = if (secret) KeyboardType.Password else keyboard),
            modifier = Modifier.fillMaxWidth().padding(top = 2.dp),
        )
    }
}

/** 알림 권한(꺼져 있으면 안드로이드 설정 열기)과 버전. 설정에서 돌아오면 다시 확인한다. */
@Composable
@Suppress("DEPRECATION")
private fun AppRows(context: Context) {
    val c = Hawk.colors
    var enabled by remember { mutableStateOf(NotificationManagerCompat.from(context).areNotificationsEnabled()) }
    val owner = androidx.compose.ui.platform.LocalLifecycleOwner.current
    DisposableEffect(owner) {
        val obs = LifecycleEventObserver { _, e ->
            if (e == Lifecycle.Event.ON_RESUME) enabled = NotificationManagerCompat.from(context).areNotificationsEnabled()
        }
        owner.lifecycle.addObserver(obs)
        onDispose { owner.lifecycle.removeObserver(obs) }
    }
    val version = remember { runCatching { context.packageManager.getPackageInfo(context.packageName, 0).versionName }.getOrNull() ?: "-" }
    Group {
        Row(Modifier.fillMaxWidth().then(if (!enabled) Modifier.clickable { openNotificationSettings(context) } else Modifier)
            .padding(horizontal = 18.dp, vertical = 14.dp)) {
            Text("알림", style = HawkType.body, color = c.text, modifier = Modifier.weight(1f))
            Text(if (enabled) "켜짐" else "꺼짐 · 설정 열기", style = HawkType.bodySmall, color = if (enabled) c.faint else c.danger)
        }
        RowDivider()
        Row(Modifier.fillMaxWidth().padding(horizontal = 18.dp, vertical = 14.dp)) {
            Text("버전", style = HawkType.body, color = c.text, modifier = Modifier.weight(1f))
            Text(version, style = HawkType.bodySmall, color = c.faint)
        }
    }
}

private fun openNotificationSettings(context: Context) {
    val intent = Intent(Settings.ACTION_APP_NOTIFICATION_SETTINGS)
        .putExtra(Settings.EXTRA_APP_PACKAGE, context.packageName)
        .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
    runCatching { context.startActivity(intent) }
}
