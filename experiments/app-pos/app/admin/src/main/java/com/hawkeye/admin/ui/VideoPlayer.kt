package com.hawkeye.admin.ui

import androidx.annotation.OptIn
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.viewinterop.AndroidView
import androidx.media3.common.MediaItem
import androidx.media3.common.util.UnstableApi
import androidx.media3.datasource.DefaultHttpDataSource
import androidx.media3.exoplayer.ExoPlayer
import androidx.media3.exoplayer.source.DefaultMediaSourceFactory
import androidx.media3.ui.PlayerView
import kotlinx.coroutines.delay

/**
 * 경보 서버의 녹화 MP4(사건 클립·지난 영상)를 재생한다. 요청마다 API 키 헤더를 붙인다.
 * 앞뒤로 넘기면 플레이어가 Range 요청을 보내고, 서버는 캐시 파일에서 206 으로 준다.
 */
@OptIn(UnstableApi::class)
@Composable
fun HttpVideoPlayer(
    url: String,
    headers: Map<String, String>,
    modifier: Modifier = Modifier.fillMaxWidth().aspectRatio(16f / 9f),
    onPosition: ((Long) -> Unit)? = null,
) {
    val context = LocalContext.current
    val player = remember(headers) {
        val http = DefaultHttpDataSource.Factory().setDefaultRequestProperties(headers)
        ExoPlayer.Builder(context).setMediaSourceFactory(DefaultMediaSourceFactory(http)).build()
    }
    DisposableEffect(player) { onDispose { player.release() } }

    LaunchedEffect(player, url) {
        player.setMediaItem(MediaItem.fromUri(url))
        player.prepare()
        player.playWhenReady = true
    }
    if (onPosition != null) {
        LaunchedEffect(player) {
            while (true) {                 // 재생 위치 → 실제 시각 표시용
                onPosition(player.currentPosition)
                delay(500)
            }
        }
    }
    AndroidView(
        factory = { PlayerView(it).apply { this.player = player; useController = true } },
        update = { it.player = player },
        modifier = modifier,
    )
}
