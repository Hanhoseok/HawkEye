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
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.rounded.KeyboardArrowLeft
import androidx.compose.material.icons.automirrored.rounded.KeyboardArrowRight
import androidx.compose.material.icons.rounded.EventBusy
import androidx.compose.material.icons.rounded.GppBad
import androidx.compose.material.icons.rounded.Info
import androidx.compose.material.icons.rounded.Refresh
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableLongStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import com.hawkeye.admin.AppGraph
import com.hawkeye.admin.data.RecordingSpan
import com.hawkeye.admin.domain.AlertTime
import com.hawkeye.admin.domain.DayBar
import com.hawkeye.admin.domain.RecordingNav
import com.hawkeye.admin.ui.components.AppBar
import com.hawkeye.admin.ui.components.BarMark
import com.hawkeye.admin.ui.components.ButtonKind
import com.hawkeye.admin.ui.components.EmptyState
import com.hawkeye.admin.ui.components.Group
import com.hawkeye.admin.ui.components.HawkButton
import com.hawkeye.admin.ui.components.Note
import com.hawkeye.admin.ui.components.RowDivider
import com.hawkeye.admin.ui.components.ScreenTitle
import com.hawkeye.admin.ui.components.Segmented
import com.hawkeye.admin.ui.components.TimeBar
import com.hawkeye.admin.ui.theme.Hawk
import com.hawkeye.admin.ui.theme.HawkType
import java.time.Duration
import java.time.Instant
import java.time.LocalDate
import java.time.OffsetDateTime
import java.time.ZoneId
import java.time.format.DateTimeFormatter
import java.util.Locale

private val dayTitle = DateTimeFormatter.ofPattern("M월 d일 EEEE", Locale.KOREAN)
private val hmFmt = DateTimeFormatter.ofPattern("HH:mm")

/**
 * 지난 영상: 카메라·날짜를 고르면 그 날을 막대 하나로 (녹화 구간 칠함 + 경보 장면 시각 표시), 아래에 녹화 구간 목록.
 * 막대 구간이나 목록·시각을 고르면 그 시각부터 1분씩 재생한다. startIso 가 오면(사건 상세의 '경보 1분 전부터') 바로 재생.
 * onBack = null 이면 아래 탭 화면.
 */
@Composable
fun RecordingsScreen(cameraId: String, startIso: String?, onBack: (() -> Unit)?, onOpenCase: (Int) -> Unit,
                     tabs: (@Composable () -> Unit)? = null) {
    val repo = AppGraph.alerts
    val api = remember { repo.api() }
    val zone = remember { ZoneId.systemDefault() }
    var cameras by remember { mutableStateOf<List<String>>(emptyList()) }
    var camera by remember { mutableStateOf(cameraId) }
    var spans by remember { mutableStateOf<List<RecordingSpan>>(emptyList()) }
    var alerts by remember { mutableStateOf<List<AlertTime>>(emptyList()) }
    var alertsComplete by remember { mutableStateOf(true) }
    var message by remember { mutableStateOf<String?>(null) }
    var loading by remember { mutableStateOf(true) }
    var chunkStart by remember { mutableStateOf(startIso) }
    var videoStart by remember { mutableStateOf<String?>(null) }
    var position by remember { mutableLongStateOf(0L) }
    var reload by remember { mutableIntStateOf(0) }
    var day by remember {
        mutableStateOf(startIso?.let { runCatching { OffsetDateTime.parse(it).atZoneSameInstant(zone).toLocalDate() }.getOrNull() }
            ?: LocalDate.now(zone))
    }
    var timeText by remember { mutableStateOf("") }
    var recordingsOn by remember { mutableStateOf(true) }

    LaunchedEffect(Unit) {
        runCatching { repo.config() }
            .onSuccess { cfg ->
                recordingsOn = cfg.recordings
                if (!cfg.recordings) { message = "서버에서 녹화 재생이 꺼져 있어요 (HAWKEYE_PLAYBACK_URL)"; loading = false }
                cameras = cfg.rtspStreams().map { it.cameraId }
                if (camera.isBlank() || camera !in cameras) camera = cameras.firstOrNull().orEmpty()
            }
            .onFailure { message = "서버에 연결하지 못했어요: ${it.message}"; loading = false }
    }
    val (fromIso, toIso) = RecordingNav.dayRange(day, zone)
    val dayFrom = DayBar.parse(fromIso)!!
    val dayTo = DayBar.parse(toIso)!!
    LaunchedEffect(camera, day, reload, recordingsOn) {
        // 바뀐 날짜·카메라와 다른 목록을 보여 주지 않게 먼저 비운다 (조회가 실패해도 이전 날짜 구간을 누를 수 없음).
        spans = emptyList()
        alerts = emptyList()
        alertsComplete = true
        if (camera.isBlank() || !recordingsOn) return@LaunchedEffect
        loading = true
        runCatching { repo.recordings(camera, fromIso, toIso) }
            .onSuccess { r -> spans = r.spans.sortedByDescending { it.start }; message = null }
            .onFailure { message = "녹화 목록을 받지 못했어요: ${it.message}" }
        runCatching { repo.alertTimes(camera, dayFrom, dayTo) }
            .onSuccess { (times, complete) -> alerts = times; alertsComplete = complete }
            .onFailure { alertsComplete = false }
        loading = false
    }

    val c = Hawk.colors
    val alertInstants = alerts.map { it.at }
    fun marksIn(from: Instant, to: Instant) = alerts.filter { !it.at.isBefore(from) && it.at.isBefore(to) }
        .map { BarMark(DayBar.fraction(it.at, from, to), it.exact, c.danger) }

    Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState())) {
        val playing = chunkStart
        // ---- 머리 ----
        if (playing != null) {
            val span = spans.firstOrNull { s -> DayBar.countIn(listOf(DayBar.parse(playing) ?: Instant.EPOCH), s) == 1 }
            AppBar(span?.let { spanTitle(it, zone) } ?: "지난 영상", onBack = {
                if (startIso != null && onBack != null && playing == startIso) onBack() else { chunkStart = null; videoStart = null; position = 0 }
            })
        } else if (onBack != null) {
            AppBar("지난 영상", onBack) { IconButton(onClick = { reload++ }) { Icon(Icons.Rounded.Refresh, "새로고침", tint = c.sub) } }
        } else {
            ScreenTitle("지난 영상") { IconButton(onClick = { reload++ }) { Icon(Icons.Rounded.Refresh, "새로고침", tint = c.sub) } }
            tabs?.invoke()
        }

        Column(Modifier.padding(horizontal = 16.dp).padding(bottom = 20.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
            if (cameras.size > 1) {
                Segmented(
                    options = cameras.map(::cameraLabel),
                    selected = cameras.indexOf(camera).coerceAtLeast(0),
                    onSelect = { camera = cameras[it]; videoStart = null },
                )
            }

            if (playing != null) {
                // ---- 재생 중 ----
                val url = api.recordingVideoUrl(camera, playing, RecordingNav.CHUNK_SEC)
                RecordedVideo(url, onReady = { videoStart = it.videoStart ?: playing }, onPosition = { position = it })
                val wall = RecordingNav.wallTime(videoStart, position) ?: playing
                Row(Modifier.fillMaxWidth().padding(horizontal = 4.dp), verticalAlignment = Alignment.Bottom) {
                    Text("보고 있는 장면", style = HawkType.caption, color = c.faint, modifier = Modifier.weight(1f))
                    Text(localTime(wall), style = HawkType.clock, color = c.text)
                }
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    HawkButton("1분 전", onClick = { chunkStart = RecordingNav.shift(playing, -60L); videoStart = null; position = 0 },
                        kind = ButtonKind.WHITE, icon = Icons.AutoMirrored.Rounded.KeyboardArrowLeft, modifier = Modifier.weight(1f))
                    HawkButton("1분 뒤", onClick = { chunkStart = RecordingNav.shift(playing, 60L); videoStart = null; position = 0 },
                        kind = ButtonKind.WHITE, icon = Icons.AutoMirrored.Rounded.KeyboardArrowRight, modifier = Modifier.weight(1f), iconEnd = true)
                }
                // 이 구간 막대 (재생 위치 · 경보)
                val at = DayBar.parse(wall) ?: DayBar.parse(playing)!!
                val span = spans.firstOrNull { DayBar.countIn(listOf(at), it) == 1 }
                val from = span?.let { DayBar.parse(it.start) } ?: at.minus(Duration.ofMinutes(10))
                val to = span?.let { DayBar.parse(it.start)?.plusMillis((it.duration * 1000).toLong()) } ?: at.plus(Duration.ofMinutes(10))
                BarCard {
                    TimeBar(
                        segments = if (span != null) listOf(0f to 1f) else DayBar.segments(spans, from, to).map { it.start to it.end },
                        marks = marksIn(from, to), cursor = DayBar.fraction(at, from, to), barHeight = 22.dp,
                        ticks = listOf(from, from.plus(Duration.between(from, to).dividedBy(2)), to).map { localHm(it.toString()) },
                    )
                }
                alerts.filter { Duration.between(it.at, at).abs() <= Duration.ofMinutes(2) }.minByOrNull { Duration.between(it.at, at).abs() }?.let { near ->
                    Note(Icons.Rounded.GppBad, c.danger, c.dangerTint,
                        text = "${localHm(near.at.toString())} 경보" + (AppGraph.alerts.case(near.caseId)?.let { " 손님 ${it.personId}" } ?: ""),
                        actionLabel = "사건 보기", onAction = { onOpenCase(near.caseId) })
                }
            } else {
                // ---- 날짜 고르기 ----
                Row(verticalAlignment = Alignment.CenterVertically) {
                    NavBox(Icons.AutoMirrored.Rounded.KeyboardArrowLeft, enabled = true) { day = day.minusDays(1) }
                    Text(day.format(dayTitle), style = HawkType.listTitle.copy(fontSize = HawkType.body.fontSize.times(1.07f)), color = c.text,
                        modifier = Modifier.weight(1f), textAlign = androidx.compose.ui.text.style.TextAlign.Center)
                    NavBox(Icons.AutoMirrored.Rounded.KeyboardArrowRight, enabled = day < LocalDate.now(zone)) { day = day.plusDays(1) }
                }
                BarCard {
                    TimeBar(
                        segments = DayBar.segments(spans, dayFrom, dayTo).map { it.start to it.end },
                        marks = marksIn(dayFrom, dayTo),
                        ticks = listOf("0시", "6시", "12시", "18시", "24시"),
                    )
                    Row(Modifier.padding(top = 8.dp), horizontalArrangement = Arrangement.spacedBy(14.dp)) {
                        Legend(c.brand.copy(alpha = 0.45f), "녹화됨")
                        Legend(c.danger, "경보")
                    }
                }
                if (!alertsComplete) {
                    Note(Icons.Rounded.Info, c.faint, c.surf, text = "경보 표시가 일부만 보일 수 있어요. 새로고침하면 다시 받아요.")
                } else if (alerts.any { !it.exact }) {
                    Note(Icons.Rounded.Info, c.faint, c.surf, text = "테두리만 있는 표시는 장면 시각을 몰라 서버가 받은 시각으로 찍은 거예요.")
                }
                // 시각으로 가기
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalAlignment = Alignment.CenterVertically) {
                    Column(Modifier.weight(1f).clip(RoundedCornerShape(16.dp)).background(c.surf).padding(horizontal = 16.dp, vertical = 10.dp)) {
                        Text("시각으로 가기", style = HawkType.tiny, color = c.faint)
                        Box {
                            if (timeText.isEmpty()) Text("14:30", style = HawkType.listTitle, color = c.fill2)
                            BasicTextField(timeText, { timeText = it }, singleLine = true,
                                textStyle = HawkType.listTitle.copy(color = c.text), cursorBrush = SolidColor(c.brand),
                                keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number))
                        }
                    }
                    HawkButton("이동", onClick = {
                        RecordingNav.atLocalTime(day, timeText, zone)?.let { chunkStart = it; videoStart = null; position = 0 }
                            ?: run { message = "시각은 14:30 처럼 입력해 주세요" }
                    }, kind = ButtonKind.BRAND, height = 56.dp)
                }
                message?.let { Text(it, style = HawkType.caption, color = c.faint, modifier = Modifier.padding(horizontal = 4.dp)) }
                // 이 날의 녹화 구간
                if (spans.isEmpty() && !loading && camera.isNotBlank()) {
                    EmptyState(Icons.Rounded.EventBusy, "이날은 녹화가 없어요", "촬영 폰이 송출할 때만 녹화돼요.")
                } else if (spans.isNotEmpty()) {
                    Group {
                        spans.forEachIndexed { i, s ->
                            if (i > 0) RowDivider()
                            val n = DayBar.countIn(alertInstants, s)
                            Row(Modifier.fillMaxWidth().clickable { chunkStart = s.start; videoStart = null; position = 0 }
                                .padding(horizontal = 18.dp, vertical = 13.dp), verticalAlignment = Alignment.CenterVertically) {
                                Text(spanTitle(s, zone), style = HawkType.body.copy(fontWeight = HawkType.label.fontWeight), color = c.text)
                                if (n > 0) {
                                    Spacer(Modifier.width(8.dp))
                                    Text("경보 $n", style = HawkType.label.copy(fontSize = HawkType.tiny.fontSize.times(1.1f)), color = c.dangerInk,
                                        modifier = Modifier.clip(CircleShape).background(c.dangerTint).padding(horizontal = 8.dp, vertical = 2.dp))
                                }
                                Spacer(Modifier.weight(1f))
                                Text(durationLabel(s.duration), style = HawkType.caption, color = c.faint)
                            }
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun BarCard(content: @Composable () -> Unit) {
    Column(Modifier.fillMaxWidth().clip(RoundedCornerShape(20.dp)).background(Hawk.colors.surf).padding(horizontal = 16.dp, vertical = 14.dp)) { content() }
}

@Composable
private fun Legend(color: androidx.compose.ui.graphics.Color, text: String) {
    Row(verticalAlignment = Alignment.CenterVertically) {
        Box(Modifier.size(10.dp).clip(RoundedCornerShape(3.dp)).background(color))
        Spacer(Modifier.width(5.dp))
        Text(text, style = HawkType.caption, color = Hawk.colors.faint)
    }
}

@Composable
private fun NavBox(icon: androidx.compose.ui.graphics.vector.ImageVector, enabled: Boolean, onClick: () -> Unit) {
    Box(Modifier.size(40.dp).clip(RoundedCornerShape(12.dp)).background(Hawk.colors.surf).clickable(enabled = enabled, onClick = onClick),
        contentAlignment = Alignment.Center) {
        Icon(icon, null, tint = if (enabled) Hawk.colors.sub else Hawk.colors.fill2)
    }
}

private fun spanTitle(s: RecordingSpan, zone: ZoneId): String {
    val start = DayBar.parse(s.start) ?: return s.start
    val end = start.plusMillis((s.duration * 1000).toLong())
    return "${start.atZone(zone).format(hmFmt)} ~ ${end.atZone(zone).format(hmFmt)}"
}

private fun durationLabel(sec: Double): String {
    val total = sec.toLong()
    val h = total / 3600
    val m = (total % 3600) / 60
    return when {
        h > 0 -> "${h}시간 ${m}분"
        m > 0 -> "${m}분"
        else -> "${total}초"
    }
}
