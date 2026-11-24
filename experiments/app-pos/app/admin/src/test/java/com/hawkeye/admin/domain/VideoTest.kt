package com.hawkeye.admin.domain

import com.hawkeye.admin.data.Health
import com.hawkeye.admin.data.LiveStatus
import com.hawkeye.admin.data.Stream
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.time.Instant
import java.time.LocalDate
import java.time.ZoneId

class VideoTest {
    private val withAi = Stream("cam1", url = "rtsp://h/cam1", aiUrl = "rtsp://h/cam1_ai")
    private val noAi = Stream("cam2", url = "rtsp://h/cam2")

    @Test fun liveUrlUsesAiOnlyWhenOnAndAvailable() {
        assertEquals("rtsp://h/cam1_ai", LiveChoice.url(withAi, ai = true))
        assertEquals("rtsp://h/cam1", LiveChoice.url(withAi, ai = false))
        assertEquals("rtsp://h/cam2", LiveChoice.url(noAi, ai = true))      // AI 주소 없는 카메라는 원본
        assertTrue(LiveChoice.showingAi(withAi, true))
        assertFalse(LiveChoice.showingAi(noAi, true))
        assertFalse(LiveChoice.hasAi(Stream("cam3", url = "rtsp://h/cam3", aiUrl = "")))
    }

    @Test fun safeUrlHidesCredentials() {
        assertEquals("rtsp://100.1.2.3:8554/cam1", safeUrl("rtsp://viewer:secret@100.1.2.3:8554/cam1"))
        assertEquals("rtsp://h/cam1", safeUrl("rtsp://h/cam1"))
    }

    @Test fun streamingLabelShowsHowLong() {
        val now = Instant.parse("2026-11-24T03:12:00Z")
        assertNull(streamingLabel(null, now))
        assertNull(streamingLabel(LiveStatus(streaming = false), now))
        assertEquals("송출 중", streamingLabel(LiveStatus(true, null), now))
        assertEquals("송출 중 · 방금 시작", streamingLabel(LiveStatus(true, "2026-11-24T03:11:30Z"), now))
        assertEquals("송출 중 · 12분", streamingLabel(LiveStatus(true, "2026-11-24T03:00:00Z"), now))
        assertEquals("송출 중 · 3시간 12분", streamingLabel(LiveStatus(true, "2026-11-24T00:00:00Z"), now))
        assertEquals("송출 중 · 방금 시작", streamingLabel(LiveStatus(true, "2026-11-24T03:13:00Z"), now)) // 시계 차이로 미래
    }

    @Test fun longStreamingIsFlaggedAfterTwoHours() {
        val now = Instant.parse("2026-11-24T03:00:00Z")
        assertFalse(streamingLong(LiveStatus(true, "2026-11-24T01:30:00Z"), now))
        assertTrue(streamingLong(LiveStatus(true, "2026-11-24T01:00:00Z"), now))
        assertFalse(streamingLong(LiveStatus(false, "2026-11-24T00:00:00Z"), now))
    }

    @Test fun diskWarningOnlyWhenDegraded() {
        assertNull(diskWarning(null))
        assertNull(diskWarning(Health("ok", 500.0)))
        assertEquals("서버 디스크 여유 60GB — 녹화 폴더를 정리하세요", diskWarning(Health("degraded", 60.4)))
        assertEquals("서버 디스크 여유 부족 — 녹화 폴더를 정리하세요", diskWarning(Health("degraded", null)))
    }

    @Test fun dayRangeAndLocalTimeUsePhoneTimeZone() {
        val seoul = ZoneId.of("Asia/Seoul")
        val day = LocalDate.of(2026, 11, 24)
        assertEquals("2026-11-23T15:00:00Z" to "2026-11-24T15:00:00Z", RecordingNav.dayRange(day, seoul))
        assertEquals("2026-11-24T05:30:00Z", RecordingNav.atLocalTime(day, "14:30", seoul))
        assertEquals("2026-11-24T00:05:00Z", RecordingNav.atLocalTime(day, " 9:05 ", seoul))
        assertNull(RecordingNav.atLocalTime(day, "2시 반", seoul))
        assertNull(RecordingNav.atLocalTime(day, "25:00", seoul))
    }

    @Test fun retryDelayIsClamped() {
        assertEquals(1_000L, retryDelayMs(0))
        assertEquals(2_000L, retryDelayMs(2))
        assertEquals(10_000L, retryDelayMs(32))
    }

    @Test fun recordingNavShiftsAndMapsPositionToWallTime() {
        assertEquals("2026-11-24T00:18:26.033627Z", RecordingNav.shift("2026-11-24T00:19:26.033627Z", -60))
        assertEquals("2026-11-24T00:20:00Z", RecordingNav.shift("2026-11-24T09:19:00+09:00", 60))
        assertNull(RecordingNav.shift("00:00:34.766", 60))
        assertEquals("2026-11-24T00:19:13.638344Z", RecordingNav.wallTime("2026-11-24T00:19:11.138344Z", 2_500))
        assertNull(RecordingNav.wallTime(null, 1_000))
        assertEquals("2026-11-24T00:19:39.171344Z", RecordingNav.spanEnd("2026-11-24T00:19:11.138344Z", 28.033))
    }

    @Test fun liveCameraPick() {
        val cams = listOf("cam1", "cam2")
        assertEquals("cam2", LiveChoice.pick("cam2", "cam1", cams))      // 요청한 카메라
        assertEquals("cam2", LiveChoice.pick("cam9", "cam2", cams))      // 설정에 없는 카메라 알림 → 보던 카메라 유지
        assertEquals("cam1", LiveChoice.pick("cam9", "", cams))          // 보던 게 없으면 첫 카메라
        assertEquals("cam1", LiveChoice.pick("", "cam1", cams))          // 탭으로 그냥 열기
        assertEquals("cam1", LiveChoice.pick("cam9", "cam7", listOf("cam1")))
        assertEquals("cam9", LiveChoice.pick("cam9", "", emptyList()))   // 목록 받기 전: 요청 유지 (받은 뒤 다시 고름)
    }
}
