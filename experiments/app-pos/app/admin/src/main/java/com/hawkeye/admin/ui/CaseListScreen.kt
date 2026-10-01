package com.hawkeye.admin.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.Card
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Tab
import androidx.compose.material3.TabRow
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.hawkeye.admin.AppGraph
import com.hawkeye.admin.alerts.Connection
import com.hawkeye.admin.data.CaseSummary
import com.hawkeye.admin.domain.CaseFilter

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun CaseListScreen(onOpen: (Int) -> Unit, onLive: () -> Unit, onSettings: () -> Unit) {
    val repo = AppGraph.alerts
    var filter by rememberSaveable { mutableStateOf(CaseFilter.ACTIVE) }
    val cases by remember(filter) { repo.cases(filter) }.collectAsState(initial = emptyList())
    val connection by repo.connection.collectAsState()
    LaunchedEffect(Unit) { runCatching { repo.refresh() } }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("경보") },
                actions = {
                    IconButton(onClick = onLive) { Icon(Icons.Default.PlayArrow, contentDescription = "라이브") }
                    IconButton(onClick = onSettings) { Icon(Icons.Default.Settings, contentDescription = "설정") }
                },
            )
        },
    ) { pad ->
        Column(Modifier.padding(pad).fillMaxSize()) {
            ConnectionBar(connection)
            TabRow(selectedTabIndex = filter.ordinal) {
                Tab(selected = filter == CaseFilter.ACTIVE, onClick = { filter = CaseFilter.ACTIVE }, text = { Text("확인 필요") })
                Tab(selected = filter == CaseFilter.ALL, onClick = { filter = CaseFilter.ALL }, text = { Text("전체") })
            }
            if (cases.isEmpty()) {
                Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                    Text(
                        if (filter == CaseFilter.ACTIVE) "확인할 경보가 없습니다" else "경보 기록이 없습니다",
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            } else {
                LazyColumn(
                    contentPadding = androidx.compose.foundation.layout.PaddingValues(12.dp),
                    verticalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    items(cases, key = { it.id }) { CaseRow(it, onClick = { onOpen(it.id) }) }
                }
            }
        }
    }
}

@Composable
private fun ConnectionBar(c: Connection) {
    val (text, color) = when (c) {
        Connection.Connected -> "경보 서버 연결됨" to Color(0xFF2E7D32)
        Connection.Connecting -> "경보 서버에 연결 중…" to Color(0xFF8D6E00)
        is Connection.Disconnected -> ("연결 끊김" + (c.reason?.let { " · $it" } ?: "") + " — 다시 연결합니다") to Color(0xFFC0392B)
    }
    Text(
        text, color = Color.White,
        modifier = Modifier.fillMaxWidth().background(color).padding(horizontal = 12.dp, vertical = 4.dp),
    )
}

@Composable
private fun CaseRow(c: CaseSummary, onClick: () -> Unit) {
    Card(Modifier.fillMaxWidth().clickable(onClick = onClick)) {
        Row(Modifier.padding(12.dp), verticalAlignment = Alignment.CenterVertically) {
            LevelBadge(c.level)
            Spacer(Modifier.width(12.dp))
            Column(Modifier.weight(1f)) {
                Text("손님 ${c.personId} · ${c.cameraId}", fontWeight = FontWeight.Bold)
                Text("감지 ${c.taken} / 결제 ${c.paid} · 미결제 ${c.unpaid}", style = MaterialTheme.typography.bodyMedium)
                Text(
                    stateLabel(c.state, c.resolution),
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
            Text(localTime(c.updatedAt), style = MaterialTheme.typography.bodySmall)
        }
    }
}
