package com.hawkeye.admin.ui

import android.content.Context
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.Close
import androidx.compose.material.icons.rounded.CloudOff
import androidx.compose.material.icons.rounded.Done
import androidx.compose.material.icons.rounded.PlayCircle
import androidx.compose.material.icons.rounded.Refresh
import androidx.compose.material.icons.rounded.Schedule
import androidx.compose.material.icons.rounded.Storage
import androidx.compose.material.icons.rounded.Verified
import androidx.compose.material.icons.rounded.Videocam
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import com.hawkeye.admin.AppGraph
import com.hawkeye.admin.alerts.AlertService
import com.hawkeye.admin.data.CaseSummary
import com.hawkeye.admin.data.Health
import com.hawkeye.admin.data.Stream
import com.hawkeye.admin.domain.CaseFilter
import com.hawkeye.admin.domain.StatusAction
import com.hawkeye.admin.domain.StatusSummaries
import com.hawkeye.admin.domain.streamingLabel
import com.hawkeye.admin.domain.streamingLong
import com.hawkeye.admin.ui.components.AlertRow
import com.hawkeye.admin.ui.components.EmptyState
import com.hawkeye.admin.ui.components.Group
import com.hawkeye.admin.ui.components.groupItem
import com.hawkeye.admin.ui.components.Note
import com.hawkeye.admin.ui.components.RowDivider
import com.hawkeye.admin.ui.components.ScreenTitle
import com.hawkeye.admin.ui.components.Segmented
import com.hawkeye.admin.ui.components.StatusCard
import com.hawkeye.admin.ui.theme.Ellipsis
import com.hawkeye.admin.ui.theme.Hawk
import com.hawkeye.admin.ui.theme.HawkType
import kotlinx.coroutines.delay
import java.time.Instant

/**
 * 경보 (첫 화면): 상태 카드(가장 급한 일 한 문장 + 처리 단추) → 카메라 송출 → 확인 필요/전체 → 경보 목록.
 * 디자인: .ai-review/admin-app-design/c2-screens.html 1번 묶음.
 */
@Composable
fun CaseListScreen(onOpen: (Int) -> Unit, onLive: () -> Unit) {
    val repo = AppGraph.alerts
    val context = LocalContext.current
    var filterAll by rememberSaveable { mutableStateOf(false) }
    val all by remember { repo.cases(CaseFilter.ALL) }.collectAsState(initial = emptyList())
    val offline by repo.offline.collectAsState()
    val sync by repo.sync.collectAsState()
    var streams by remember { mutableStateOf<List<Stream>>(emptyList()) }
    var health by remember { mutableStateOf<Health?>(null) }
    var now by remember { mutableStateOf(Instant.now()) }

    LaunchedEffect(Unit) { runCatching { repo.refresh() } }
    // 카메라 송출 상태·디스크·상태 카드 시각. 30초마다 갱신.
    LaunchedEffect(Unit) {
        while (true) {
            runCatching { repo.config() }.onSuccess { streams = it.rtspStreams() }
            health = runCatching { repo.health() }.getOrNull()
            now = Instant.now()
            delay(30_000)
        }
    }

    val summary = StatusSummaries.of(all, offline = offline, sync = sync, now = now)
    val shown = if (filterAll) all else all.filter { it.state == "active" }
    val activeN = StatusSummaries.activeCount(all)

    LazyColumn(
        Modifier.fillMaxSize(),
        contentPadding = androidx.compose.foundation.layout.PaddingValues(start = 16.dp, end = 16.dp, bottom = 20.dp),
    ) {
        // 묶음 사이 간격은 각 항목 아래 여백으로 (경보 줄끼리는 붙어서 한 묶음처럼 보여야 하므로 LazyColumn 간격을 쓰지 않음)
        val gap = Modifier.padding(bottom = 12.dp)
        item { ScreenTitle("호크아이", brand = true) }
        item {
            Box(gap) {
                StatusCard(
                    summary,
                    actionIcon = when (summary.action) {
                        StatusAction.OPEN_CASE -> Icons.Rounded.PlayCircle
                        StatusAction.LIVE -> Icons.Rounded.Videocam
                        StatusAction.RECONNECT -> Icons.Rounded.Refresh
                    },
                    onAction = {
                        when (summary.action) {
                            StatusAction.OPEN_CASE -> summary.caseId?.let(onOpen)
                            StatusAction.LIVE -> onLive()
                            StatusAction.RECONNECT -> reconnect(context)
                        }
                    },
                )
            }
        }
        if (streams.isNotEmpty()) item { Box(gap) { CameraStrip(streams, now, offline) } }
        streams.firstOrNull { streamingLong(it.live, now) }?.let {
            item {
                Note(Icons.Rounded.Schedule, Hawk.colors.warn, Hawk.colors.warnTint, modifier = gap,
                    title = "송출을 켜 둔 채인가요?", text = "3시간 넘게 송출 중이에요. 촬영이 끝났으면 폰에서 송출을 멈춰 주세요.")
            }
        }
        health?.takeIf { it.status == "degraded" }?.let { h ->
            item {
                Note(Icons.Rounded.Storage, Hawk.colors.danger, Hawk.colors.dangerTint, modifier = gap,
                    title = "서버 저장 공간 ${h.diskFreeGb?.let { "%.0fGB".format(it) } ?: "부족"} 남음",
                    text = "녹화 폴더를 정리해 주세요. 녹화는 자동으로 지우지 않아요.")
            }
        }
        item {
            Segmented(
                options = listOf("확인 필요", "전체"),
                selected = if (filterAll) 1 else 0,
                onSelect = { filterAll = it == 1 },
                trailing = listOf(if (activeN > 0) "$activeN" else null, null),
                modifier = gap,
            )
        }
        if (shown.isEmpty()) {
            item {
                Group {
                    if (offline) EmptyState(Icons.Rounded.CloudOff, "마지막으로 받은 경보까지 보여요", "다시 연결되면 그동안 온 경보를 알려 드려요.")
                    else if (filterAll) EmptyState(Icons.Rounded.Verified, "받은 경보가 없어요", "경보가 오면 여기에 쌓여요.")
                    else EmptyState(Icons.Rounded.Verified, "확인할 경보가 없어요", "새 경보가 오면 맨 위에 보여요.")
                }
            }
        } else {
            // 사건마다 따로 그린다 (보이는 줄만 만들어짐). 첫 줄·마지막 줄 모서리만 둥글게 해서 한 묶음처럼 보이게.
            itemsIndexed(shown, key = { _, c -> c.id }) { i, c ->
                Column(Modifier.groupItem(i, shown.size)) {
                    if (i > 0) RowDivider(start = 66.dp)
                    CaseRow(c, onClick = { onOpen(c.id) })
                }
            }
        }
    }
}

@Composable
private fun CaseRow(c: CaseSummary, onClick: () -> Unit) {
    val handled = c.state != "active"
    val title = if (handled) "손님 ${c.personId} ${stateLabel(c.state, c.resolution)}" else "손님 ${c.personId}"
    val sub = if (handled) "${levelLabel(c.level)} 경보, ${caseSummaryLine(c.level, c.taken, c.paid, c.unpaid)}"
        else caseSummaryLine(c.level, c.taken, c.paid, c.unpaid)
    AlertRow(
        level = c.level, title = title, subtitle = sub, time = localHm(c.updatedAt), onClick = onClick,
        mutedIcon = when {
            !handled -> null
            c.resolution == "false_alarm" -> Icons.Rounded.Close
            else -> Icons.Rounded.Done
        },
    )
}

/** 카메라 송출 상태 두 칸. 서버와 끊겼으면 '확인할 수 없음'. */
@Composable
private fun CameraStrip(streams: List<Stream>, now: Instant, offline: Boolean) {
    val c = Hawk.colors
    Row(Modifier.fillMaxWidth().clip(RoundedCornerShape(20.dp)).background(c.surf).padding(vertical = 14.dp, horizontal = 4.dp)) {
        streams.take(3).forEachIndexed { i, s ->
            if (i > 0) Box(Modifier.width(1.dp).height(38.dp).background(c.line))
            val label = if (offline) null else streamingLabel(s.live, now)
            val long = !offline && streamingLong(s.live, now)
            Column(Modifier.weight(1f).padding(horizontal = 14.dp)) {
                Row(verticalAlignment = androidx.compose.ui.Alignment.CenterVertically) {
                    Box(Modifier.size(7.dp).background(if (label != null) c.ok else c.fill2, CircleShape))
                    Spacer(Modifier.width(7.dp))
                    Text(cameraLabel(s.cameraId), style = HawkType.listTitle.copy(fontSize = HawkType.bodySmall.fontSize), color = c.text, maxLines = 1, overflow = Ellipsis)
                }
                Spacer(Modifier.height(3.dp))
                Text(
                    when {
                        offline -> "확인할 수 없음"
                        label == null -> "송출 전"
                        else -> label.removePrefix("송출 중 · ").let { if (it == "방금 시작") "방금 송출 시작" else "${it}째 송출" }
                    },
                    style = HawkType.caption, color = if (long) c.warnInk else c.faint, maxLines = 1,
                )
            }
        }
    }
}

/** 서버와 다시 연결 (서비스를 다시 시작하면 바로 재시도 + 목록 다시 받기). */
private fun reconnect(context: Context) {
    AlertService.start(context)
}
