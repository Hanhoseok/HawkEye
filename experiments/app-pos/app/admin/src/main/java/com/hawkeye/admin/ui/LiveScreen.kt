package com.hawkeye.admin.ui

import androidx.annotation.OptIn
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.GppBad
import androidx.compose.material.icons.rounded.Videocam
import androidx.compose.material.icons.rounded.VideocamOff
import androidx.compose.material.icons.rounded.VisibilityOff
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.media3.common.MediaItem
import androidx.media3.common.PlaybackException
import androidx.media3.common.Player
import androidx.media3.common.util.UnstableApi
import androidx.media3.exoplayer.ExoPlayer
import androidx.media3.exoplayer.rtsp.RtspMediaSource
import androidx.media3.ui.PlayerView
import com.hawkeye.admin.AppGraph
import com.hawkeye.admin.data.Stream
import com.hawkeye.admin.domain.Backoff
import com.hawkeye.admin.domain.CaseFilter
import com.hawkeye.admin.domain.LiveChoice
import com.hawkeye.admin.ui.components.EmptyState
import com.hawkeye.admin.ui.components.Note
import com.hawkeye.admin.ui.components.ScreenTitle
import com.hawkeye.admin.ui.components.Segmented
import com.hawkeye.admin.ui.components.SwitchRow
import com.hawkeye.admin.ui.components.Veil
import com.hawkeye.admin.ui.components.VideoTag
import com.hawkeye.admin.ui.theme.Hawk
import com.hawkeye.admin.ui.theme.HawkType
import kotlinx.coroutines.delay

/**
 * 라이브 (MediaMTX RTSP). 주소는 서버의 /api/config 에서 받는다.
 * - 카메라 전환(cam1 전체 · cam2 선반), 'AI 탐지 화면' 스위치 — 마지막 선택을 기억
 * - 끊기면 버튼 없이 다시 연결 (1초부터 2배씩, 최대 30초). 화면을 나가면 멈춘다.
 * - 주소에는 보기 계정이 들어 있어 화면에 표시하지 않는다 (시연 화면 녹화에 찍히지 않게).
 */
@OptIn(UnstableApi::class)
@Composable
fun LiveScreen(cameraId: String, request: Int = 0, onOpenCase: (Int) -> Unit) {
    val context = LocalContext.current
    val settings = AppGraph.settings
    val ai by settings.aiOverlay.collectAsState()
    val cases by remember { AppGraph.alerts.cases(CaseFilter.ACTIVE) }.collectAsState(initial = emptyList())
    var streams by remember { mutableStateOf<List<Stream>>(emptyList()) }
    var loadError by remember { mutableStateOf<String?>(null) }
    var selected by remember { mutableStateOf(cameraId) }
    var playing by remember { mutableStateOf(false) }
    var attempt by remember { mutableIntStateOf(0) }
    var failures by remember { mutableIntStateOf(0) }
    var waitSec by remember { mutableIntStateOf(0) }
    val backoff = remember { Backoff() }

    LaunchedEffect(Unit) {
        runCatching { AppGraph.alerts.config() }
            .onSuccess { cfg ->
                streams = cfg.rtspStreams()
                if (streams.none { it.cameraId == selected }) selected = streams.firstOrNull()?.cameraId.orEmpty()
                loadError = if (streams.isEmpty()) "서버에 등록된 카메라가 없어요 (HAWKEYE_STREAMS)" else null
            }
            .onFailure { loadError = "서버에 연결하지 못했어요: ${it.message}" }
    }
    // 알림의 '라이브'(또는 경보 화면)에서 카메라를 요청할 때마다 그 카메라로 — 화면에서 손으로 바꾼 뒤 같은 카메라를 다시 요청해도.
    // 요청한 카메라가 서버 설정에 없으면 보던 카메라(또는 첫 카메라)를 그대로 보여 준다.
    LaunchedEffect(cameraId, request, streams) {
        val next = LiveChoice.pick(cameraId, selected, streams.map { it.cameraId })
        if (next != selected) {
            selected = next
            backoff.reset()
            failures = 0
        }
    }
    val stream = streams.firstOrNull { it.cameraId == selected }
    val showingAi = stream != null && LiveChoice.showingAi(stream, ai)

    val player = remember { ExoPlayer.Builder(context).build() }
    DisposableEffect(player) {
        val listener = object : Player.Listener {
            override fun onPlaybackStateChanged(state: Int) {
                when (state) {
                    Player.STATE_BUFFERING -> playing = false
                    Player.STATE_READY -> { playing = true; backoff.reset() }
                    Player.STATE_ENDED -> { playing = false; failures++ }
                }
            }

            override fun onPlayerError(error: PlaybackException) {
                playing = false
                failures++
            }
        }
        player.addListener(listener)
        onDispose {
            player.removeListener(listener)
            player.release()
        }
    }

    // 끊기면 기다렸다가 다시 연결. 화면을 나가면 이 효과가 취소돼 재시도도 멈춘다. 남은 초를 1초마다 보여 준다.
    LaunchedEffect(failures) {
        if (failures == 0) return@LaunchedEffect
        var left = backoff.next()
        while (left > 0) {
            waitSec = ((left + 999) / 1000).toInt()
            val step = minOf(1000L, left)
            delay(step)
            left -= step
        }
        waitSec = 0
        attempt++
    }

    LaunchedEffect(stream?.cameraId, showingAi, attempt) {
        val s = stream ?: return@LaunchedEffect
        val url = LiveChoice.url(s, ai)
        // 와이파이·LTE 에서 UDP 가 막히는 경우가 많아 TCP 로 받는다.
        val source = RtspMediaSource.Factory().setForceUseRtpTcp(true).createMediaSource(MediaItem.fromUri(url))
        player.setMediaSource(source)
        player.prepare()
        player.playWhenReady = true
    }

    val c = Hawk.colors
    Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState())) {
        ScreenTitle("라이브")
        Column(Modifier.padding(horizontal = 16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
            if (streams.size > 1) {
                Segmented(
                    options = streams.map { cameraLabel(it.cameraId) },
                    selected = streams.indexOfFirst { it.cameraId == selected }.coerceAtLeast(0),
                    onSelect = { i ->
                        val id = streams[i].cameraId
                        if (id != selected) { selected = id; backoff.reset(); failures = 0 }   // 같은 카메라를 다시 누르면 무시
                    },
                    dots = streams.map { if (it.live?.streaming == true) c.ok else c.fill2 },
                )
            }
            if (stream == null) {
                EmptyState(Icons.Rounded.VideocamOff, "볼 수 있는 카메라가 없어요", loadError ?: "서버에서 카메라 목록을 받는 중이에요.")
                return@Column
            }
            VideoFrame {
                AndroidView(
                    factory = { PlayerView(it).apply { this.player = player; useController = false } },
                    modifier = Modifier.fillMaxSize(),
                )
                when {
                    failures > 0 && !playing && showingAi -> Veil(Icons.Rounded.VisibilityOff, "AI 화면을 받을 수 없어요", "탐지 프로그램이 돌 때만 나와요")
                    failures > 0 && !playing -> Veil(Icons.Rounded.VideocamOff, "${cameraLabel(stream.cameraId)} 송출이 멈췄어요",
                        if (waitSec > 0) "${waitSec}초 뒤 다시 연결해요" else "다시 연결하는 중")
                    !playing -> Veil(Icons.Rounded.Videocam, "연결하는 중", null)
                }
                if (playing) {
                    VideoTag("실시간", Modifier.align(Alignment.BottomStart).padding(10.dp), live = true)
                    if (showingAi) VideoTag("AI", Modifier.align(Alignment.BottomEnd).padding(10.dp), light = true)
                }
            }
            if (LiveChoice.hasAi(stream)) {
                SwitchRow(
                    "AI 탐지 화면",
                    if (ai) "사람마다 상자와 번호를 그려서 보여 줘요" else "끄면 카메라 원본을 그대로 보여 줘요",
                    checked = ai,
                    onChange = { settings.setAiOverlay(it); backoff.reset(); failures = 0 },
                )
            }
            when {
                failures > 0 && !playing && showingAi -> Note(Icons.Rounded.Videocam, c.brand, c.brandTint, text = "원본은 볼 수 있어요",
                    actionLabel = "원본으로 보기", onAction = { settings.setAiOverlay(false); backoff.reset(); failures = 0 })
                failures > 0 && !playing -> Text("촬영 폰의 송출 앱(Larix)과 와이파이를 확인해 주세요. 다시 송출하면 저절로 이어져요.",
                    style = HawkType.caption, color = c.faint, modifier = Modifier.padding(horizontal = 4.dp))
                showingAi -> Text("AI 화면은 원본보다 조금 늦어요.", style = HawkType.caption, color = c.faint, modifier = Modifier.padding(horizontal = 4.dp))
            }
            cases.filter { it.level == "HIGH_RISK" }.maxByOrNull { it.updatedAt }?.let { risk ->
                Note(Icons.Rounded.GppBad, c.danger, c.dangerTint, text = "손님 ${risk.personId} 위험 경보 미처리",
                    actionLabel = "보기", onAction = { onOpenCase(risk.id) })
            }
        }
    }
}
