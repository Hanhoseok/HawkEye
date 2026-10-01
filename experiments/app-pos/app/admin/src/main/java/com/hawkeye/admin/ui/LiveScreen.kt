package com.hawkeye.admin.ui

import androidx.annotation.OptIn
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material3.Button
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
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

/** 중계기(MediaMTX)의 RTSP 영상을 그대로 보여준다. 주소는 서버의 /api/config 에서 받는다. */
@OptIn(UnstableApi::class)
@kotlin.OptIn(ExperimentalMaterial3Api::class)
@Composable
fun LiveScreen(cameraId: String, onBack: () -> Unit) {
    val context = LocalContext.current
    var stream by remember { mutableStateOf<Stream?>(null) }
    var status by remember { mutableStateOf("주소를 받는 중…") }
    var attempt by remember { mutableIntStateOf(0) }

    LaunchedEffect(cameraId) {
        runCatching { AppGraph.alerts.config() }
            .onSuccess { cfg ->
                stream = cfg.streams.firstOrNull { it.cameraId == cameraId } ?: cfg.streams.firstOrNull()
                if (stream == null) status = "서버에 등록된 카메라가 없습니다 (HAWKEYE_STREAMS)"
            }
            .onFailure { status = "서버에 연결하지 못했습니다: ${it.message}" }
    }

    val player = remember { ExoPlayer.Builder(context).build() }
    DisposableEffect(player) {
        val listener = object : Player.Listener {
            override fun onPlaybackStateChanged(state: Int) {
                status = when (state) {
                    Player.STATE_BUFFERING -> "연결 중…"
                    Player.STATE_READY -> ""
                    Player.STATE_ENDED -> "영상이 끝났습니다"
                    else -> status
                }
            }

            override fun onPlayerError(error: PlaybackException) {
                status = "영상을 열지 못했습니다: ${error.errorCodeName}"
            }
        }
        player.addListener(listener)
        onDispose {
            player.removeListener(listener)
            player.release()
        }
    }

    LaunchedEffect(stream, attempt) {
        val s = stream ?: return@LaunchedEffect
        // 에뮬레이터·와이파이에서 UDP 가 막히는 경우가 많아 TCP 로 받는다.
        val source = RtspMediaSource.Factory().setForceUseRtpTcp(true).createMediaSource(MediaItem.fromUri(s.url))
        player.setMediaSource(source)
        player.prepare()
        player.playWhenReady = true
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("라이브 · ${stream?.cameraId ?: cameraId.ifBlank { "카메라" }}") },
                navigationIcon = {
                    IconButton(onClick = onBack) { Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = "뒤로") }
                },
            )
        },
    ) { pad ->
        Column(Modifier.padding(pad).fillMaxSize()) {
            Box(Modifier.fillMaxWidth().aspectRatio(16f / 9f)) {
                AndroidView(
                    factory = { PlayerView(it).apply { this.player = player; useController = false } },
                    modifier = Modifier.fillMaxSize(),
                )
            }
            if (status.isNotEmpty()) {
                Text(status, color = Color(0xFFC0392B), modifier = Modifier.padding(16.dp))
                if (stream != null) Button(onClick = { attempt++ }, modifier = Modifier.padding(horizontal = 16.dp)) { Text("다시 시도") }
            }
            stream?.let { Text(it.url, modifier = Modifier.padding(16.dp)) }
        }
    }
}
