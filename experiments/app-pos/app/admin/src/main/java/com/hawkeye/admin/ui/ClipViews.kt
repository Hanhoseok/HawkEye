package com.hawkeye.admin.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxScope
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.ErrorOutline
import androidx.compose.material.icons.rounded.EventBusy
import androidx.compose.material.icons.rounded.HourglassTop
import androidx.compose.material.icons.rounded.Info
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import com.hawkeye.admin.AppGraph
import com.hawkeye.admin.data.CaseClips
import com.hawkeye.admin.data.VideoState
import com.hawkeye.admin.domain.retryDelayMs
import com.hawkeye.admin.ui.components.Chip
import com.hawkeye.admin.ui.components.Note
import com.hawkeye.admin.ui.components.SectionLabel
import com.hawkeye.admin.ui.components.Veil
import com.hawkeye.admin.ui.theme.Hawk
import kotlinx.coroutines.delay

/** 영상 자리 (16:9, 둥근 모서리, 어두운 바탕). */
@Composable
fun VideoFrame(modifier: Modifier = Modifier, content: @Composable BoxScope.() -> Unit) {
    Box(modifier.fillMaxWidth().aspectRatio(16f / 9f).clip(RoundedCornerShape(18.dp)).background(Color(0xFF1B2129))) { content() }
}

/**
 * 녹화 MP4 하나: 받을 수 있을 때까지 기다렸다가 재생한다.
 * 아직 녹화 중이면 서버가 알려 준 시간만큼 기다렸다 다시 묻는다 (최대 maxTries 번). 상태는 영상 자리 위 가림막으로.
 */
@Composable
fun RecordedVideo(absoluteUrl: String, onReady: (VideoState.Ready) -> Unit = {}, onPosition: ((Long) -> Unit)? = null,
                  maxTries: Int = 20) {
    val api = remember { AppGraph.alerts.api() }
    var state by remember(absoluteUrl) { mutableStateOf<VideoState?>(null) }
    var tries by remember(absoluteUrl) { mutableIntStateOf(0) }

    LaunchedEffect(absoluteUrl, tries) {
        var s = AppGraph.alerts.videoState(absoluteUrl)
        var n = 0
        while (s is VideoState.NotReady && n < maxTries) {
            state = s
            delay(retryDelayMs(s.retryAfterSec))
            s = AppGraph.alerts.videoState(absoluteUrl)
            n++
        }
        state = s
        if (s is VideoState.Ready) onReady(s)
    }

    VideoFrame {
        when (val s = state) {
            null -> Veil(Icons.Rounded.HourglassTop, "영상을 확인하는 중", null)
            is VideoState.NotReady -> Veil(Icons.Rounded.HourglassTop, "영상 준비 중", "아직 녹화되는 구간이에요. ${s.retryAfterSec}초쯤 걸려요.")
            is VideoState.NoRecording -> Veil(Icons.Rounded.EventBusy, "이 구간에는 녹화가 없어요", "그 시간에는 송출하지 않았어요.")
            is VideoState.Unavailable -> Box(Modifier.fillMaxSize().clickable { tries++ }) {
                Veil(Icons.Rounded.ErrorOutline, "녹화 영상을 받지 못했어요", "${s.reason} · 눌러서 다시 시도")
            }
            is VideoState.Ready -> HttpVideoPlayer(url = absoluteUrl, headers = api.authHeaders(), modifier = Modifier.fillMaxSize(), onPosition = onPosition)
        }
    }
}

/** 사건 상세의 '관련 영상': 경보 전후 30초를 카메라별로. 경보 카메라가 먼저. */
@Composable
fun CaseClipsSection(clips: CaseClips) {
    val api = remember { AppGraph.alerts.api() }
    val ordered = remember(clips) { clips.clips.sortedByDescending { it.isCaseCamera } }
    var selected by remember(clips.caseId) { mutableStateOf(ordered.firstOrNull()?.cameraId) }
    val clip = ordered.firstOrNull { it.cameraId == selected } ?: return
    val c = Hawk.colors

    Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
        SectionLabel("관련 영상", trailing = "경보 20초 전부터 10초 뒤까지")
        if (ordered.size > 1) {
            Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                ordered.forEach { cl ->
                    Chip(cameraLabel(cl.cameraId) + if (cl.isCaseCamera) " (경보)" else "", selected = cl.cameraId == selected,
                        onClick = { selected = cl.cameraId })
                }
            }
        }
        when (clip.status) {
            "unavailable" -> VideoFrame { Veil(Icons.Rounded.ErrorOutline, "녹화 서버에 연결하지 못했어요", "잠시 뒤 다시 열어 주세요.") }
            "no_recording" -> VideoFrame { Veil(Icons.Rounded.EventBusy, "이 구간에는 녹화가 없어요", "${cameraLabel(clip.cameraId)}는 그 시간에 송출하지 않았어요.") }
            else -> RecordedVideo(api.clipUrl(clip.url, clips.start))
        }
        if (clip.partial == true && clip.status != "no_recording") {
            Note(Icons.Rounded.Info, c.faint, c.surf,
                text = "${cameraLabel(clip.cameraId)}는 녹화가 일부만 있어요 (${clip.coveredSec?.toInt() ?: 0}초). ${localTime(clip.videoStart ?: clips.start)}부터 시작해요.")
        }
        if (clips.anchor == "received") {
            Note(Icons.Rounded.Info, c.faint, c.surf,
                text = "이 경보에는 장면 시각이 없어 서버가 받은 시각을 기준으로 했어요. 실제 장면과 몇 초 다를 수 있어요.")
        }
        clips.delaySec?.takeIf { it > 5 }?.let {
            Note(Icons.Rounded.Info, c.faint, c.surf, text = "경보가 ${it.toInt()}초 늦게 도착했어요. 영상은 실제 장면 시각 기준이에요.")
        }
    }
}
