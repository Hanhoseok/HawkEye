package com.hawkeye.admin.ui

import androidx.annotation.OptIn
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.viewinterop.AndroidView
import androidx.media3.common.MediaItem
import androidx.media3.common.PlaybackException
import androidx.media3.common.Player
import androidx.media3.common.util.UnstableApi
import androidx.media3.datasource.DefaultHttpDataSource
import androidx.media3.exoplayer.ExoPlayer
import androidx.media3.exoplayer.source.DefaultMediaSourceFactory
import androidx.media3.ui.PlayerView
import kotlinx.coroutines.delay

/**
 * 경보 서버의 MP4(사건 클립·지난 영상·보관함)를 재생한다. 요청마다 API 키 헤더를 붙인다.
 * 앞뒤로 넘기면 플레이어가 Range 요청을 보내고, 서버는 파일에서 206 으로 준다.
 * startMs = 주소가 바뀔 때 시작할 위치 (보관함에서 카메라를 바꿀 때 같은 순간부터).
 */
@OptIn(UnstableApi::class)
@Composable
fun HttpVideoPlayer(
    url: String,
    headers: Map<String, String>,
    modifier: Modifier = Modifier.fillMaxWidth().aspectRatio(16f / 9f),
    startMs: Long = 0L,
    onPosition: ((Long) -> Unit)? = null,
    onError: ((PlaybackException) -> Unit)? = null,
) {
    val context = LocalContext.current
    val player = remember(headers) {
        val http = DefaultHttpDataSource.Factory().setDefaultRequestProperties(headers)
        ExoPlayer.Builder(context).setMediaSourceFactory(DefaultMediaSourceFactory(http)).build()
    }
    val errorCallback = rememberUpdatedState(onError)
    DisposableEffect(player) {
        val listener = object : Player.Listener {
            override fun onPlayerError(error: PlaybackException) {
                errorCallback.value?.invoke(error)
            }
        }
        player.addListener(listener)
        onDispose {
            player.removeListener(listener)
            player.release()
        }
    }

    LaunchedEffect(player, url) {
        player.setMediaItem(MediaItem.fromUri(url), startMs.coerceAtLeast(0L))
        player.prepare()
        player.playWhenReady = true
    }
    if (onPosition != null) {
        val positionCallback = rememberUpdatedState(onPosition)
        LaunchedEffect(player) {
            while (true) {                 // 재생 위치 → 실제 시각 표시 · 카메라 바꿀 때 이어 보기
                positionCallback.value?.invoke(player.currentPosition)
                delay(250)
            }
        }
    }
    AndroidView(
        factory = { PlayerView(it).apply { this.player = player; useController = true } },
        update = { it.player = player },
        modifier = modifier,
    )
}
