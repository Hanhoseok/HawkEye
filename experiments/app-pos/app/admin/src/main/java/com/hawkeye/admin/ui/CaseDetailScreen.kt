package com.hawkeye.admin.ui

import android.graphics.BitmapFactory
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.hawkeye.admin.AppGraph
import com.hawkeye.admin.data.CaseDetail
import com.hawkeye.admin.data.Resolution
import com.hawkeye.admin.domain.CaseFilter
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.launch

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun CaseDetailScreen(caseId: Int, onBack: () -> Unit, onLive: (String) -> Unit) {
    val repo = AppGraph.alerts
    val scope = rememberCoroutineScope()
    // 사건이 갱신되면(단계 상승·다른 폰에서 처리) 상세를 다시 불러온다.
    val summary by remember(caseId) { repo.cases(CaseFilter.ALL).map { repo.case(caseId) } }
        .collectAsState(initial = repo.case(caseId))
    var detail by remember { mutableStateOf<CaseDetail?>(null) }
    var image by remember { mutableStateOf<ImageBitmap?>(null) }
    var error by remember { mutableStateOf<String?>(null) }
    var note by remember { mutableStateOf("") }
    var saving by remember { mutableStateOf(false) }

    LaunchedEffect(caseId, summary?.updatedAt) {
        runCatching { repo.detail(caseId) }
            .onSuccess { d ->
                detail = d
                error = null
                val eventId = d.events.lastOrNull { it.hasSnapshot }?.id
                image = eventId?.let { id ->
                    runCatching { repo.snapshot(id) }.getOrNull()
                        ?.let { BitmapFactory.decodeByteArray(it, 0, it.size)?.asImageBitmap() }
                }
            }
            .onFailure { error = it.message ?: "불러오지 못했습니다" }
    }

    fun resolve(r: Resolution) {
        saving = true
        scope.launch {
            runCatching { repo.resolve(caseId, r, note.ifBlank { null }) }
                .onSuccess { note = "" }
                .onFailure { error = it.message ?: "저장하지 못했습니다" }
            saving = false
        }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("손님 ${summary?.personId ?: detail?.case?.personId ?: ""}") },
                navigationIcon = {
                    IconButton(onClick = onBack) { Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = "뒤로") }
                },
            )
        },
    ) { pad ->
        val d = detail
        Column(
            Modifier.padding(pad).padding(16.dp).verticalScroll(rememberScrollState()),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            error?.let { Text(it, color = Color(0xFFC0392B)) }
            if (d == null) {
                Text("불러오는 중…")
                return@Column
            }
            val c = d.case
            Row {
                LevelBadge(c.level)
                Spacer(Modifier.width(8.dp))
                Text(stateLabel(c.state, c.resolution), fontWeight = FontWeight.Bold)
            }

            if (image != null) {
                Image(
                    bitmap = image!!, contentDescription = "경보 장면",
                    contentScale = ContentScale.FillWidth, modifier = Modifier.fillMaxWidth(),
                )
            } else if (c.hasSnapshot) {
                Text("장면 사진을 불러오는 중…", color = MaterialTheme.colorScheme.onSurfaceVariant)
            }

            Row(horizontalArrangement = Arrangement.spacedBy(24.dp)) {
                Count("감지", c.taken)
                Count("결제", c.paid)
                Count("미결제", c.unpaid, highlight = c.unpaid > 0)
            }

            c.items?.takeIf { it.isNotEmpty() }?.let { items ->
                Text("품목별", fontWeight = FontWeight.Bold)
                items.forEach { Text("${it.name}  감지 ${it.taken} / 결제 ${it.paid}") }
            }

            c.reason?.takeIf { it.isNotBlank() }?.let { Text("사유: $it") }
            c.identityCheck?.let { Text("신원 일치도 ${"%.2f".format(it)} (낮으면 추적 번호가 다른 사람에게 옮겨 붙었을 수 있음)") }

            OutlinedButton(onClick = { onLive(c.cameraId) }, modifier = Modifier.fillMaxWidth()) {
                Text("라이브 보기 (${c.cameraId})")
            }

            HorizontalDivider()
            Text("관리자 처리", fontWeight = FontWeight.Bold)
            OutlinedTextField(
                value = note, onValueChange = { note = it },
                label = { Text("메모 (선택)") }, modifier = Modifier.fillMaxWidth(),
            )
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.fillMaxWidth()) {
                Button(onClick = { resolve(Resolution.PAID_CONFIRMED) }, enabled = !saving, modifier = Modifier.weight(1f)) {
                    Text("결제 확인 완료")
                }
                Button(
                    onClick = { resolve(Resolution.FALSE_ALARM) }, enabled = !saving, modifier = Modifier.weight(1f),
                    colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF7F8C8D)),
                ) { Text("잘못된 알림") }
            }
            d.resolutions.forEach {
                Text(
                    "${localTime(it.at)}  ${resolutionLabel(it.resolution)}" + (it.note?.let { n -> " — $n" } ?: ""),
                    style = MaterialTheme.typography.bodySmall,
                )
            }

            HorizontalDivider()
            Text("경보 이력", fontWeight = FontWeight.Bold)
            d.events.forEach {
                Row {
                    LevelBadge(it.level)
                    Spacer(Modifier.width(8.dp))
                    Text("영상 ${it.timestamp ?: "-"} · 수신 ${localTime(it.receivedAt)}", style = MaterialTheme.typography.bodySmall)
                }
            }
            Spacer(Modifier.height(24.dp))
        }
    }
}

@Composable
private fun Count(label: String, value: Int, highlight: Boolean = false) {
    Column {
        Text(label, style = MaterialTheme.typography.bodySmall)
        Text(
            "$value", style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.Bold,
            color = if (highlight) Color(0xFFC0392B) else Color.Unspecified,
        )
    }
}
