package com.hawkeye.admin.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material3.Button
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.unit.dp
import com.hawkeye.admin.AppGraph
import com.hawkeye.admin.alerts.AlertService
import com.hawkeye.admin.data.ApiClient
import com.hawkeye.admin.data.ServerSettings
import kotlinx.coroutines.launch

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SettingsScreen(onBack: () -> Unit) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val current = AppGraph.settings.settings.value
    var url by remember { mutableStateOf(current.baseUrl) }
    var key by remember { mutableStateOf(current.apiKey.orEmpty()) }
    var result by remember { mutableStateOf("") }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("설정") },
                navigationIcon = {
                    IconButton(onClick = onBack) { Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = "뒤로") }
                },
            )
        },
    ) { pad ->
        Column(Modifier.padding(pad).padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
            OutlinedTextField(
                value = url, onValueChange = { url = it }, singleLine = true,
                label = { Text("경보 서버 주소") }, modifier = Modifier.fillMaxWidth(),
                supportingText = { Text("에뮬레이터: http://10.0.2.2:8000 · 폰: 노트북 IP (같은 와이파이)") },
            )
            OutlinedTextField(
                value = key, onValueChange = { key = it }, singleLine = true,
                label = { Text("API 키 (서버에 설정한 경우)") }, modifier = Modifier.fillMaxWidth(),
                visualTransformation = PasswordVisualTransformation(),
            )
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                OutlinedButton(onClick = {
                    result = "확인 중…"
                    scope.launch {
                        result = runCatching { ApiClient(ServerSettings(url.trim().trimEnd('/'), key.ifBlank { null })).config() }
                            .fold({ "연결 성공 · 카메라 ${it.streams.size}대" }, { "연결 실패: ${it.message}" })
                    }
                }) { Text("연결 확인") }
                Button(onClick = {
                    AppGraph.settings.save(url, key)
                    AlertService.start(context) // 새 설정으로 다시 연결
                    result = "저장했습니다"
                }) { Text("저장") }
            }
            if (result.isNotEmpty()) Text(result)
        }
    }
}
