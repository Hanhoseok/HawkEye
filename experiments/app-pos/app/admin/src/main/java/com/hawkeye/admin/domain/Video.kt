package com.hawkeye.admin.domain

import com.hawkeye.admin.data.Health
import com.hawkeye.admin.data.LiveStatus
import com.hawkeye.admin.data.Stream
import java.time.Duration
import java.time.Instant
import java.time.LocalDate
import java.time.LocalTime
import java.time.OffsetDateTime
import java.time.ZoneId
import java.time.ZoneOffset
import java.time.format.DateTimeFormatter

/** 라이브 재생 주소 고르기: AI 화면을 켰고 그 카메라에 AI 주소가 있으면 AI, 아니면 원본. */
object LiveChoice {
    fun hasAi(s: Stream): Boolean = !s.aiUrl.isNullOrBlank()

    fun url(s: Stream, ai: Boolean): String = if (ai && hasAi(s)) s.aiUrl!! else s.url

    fun showingAi(s: Stream, ai: Boolean): Boolean = ai && hasAi(s)

    /**
     * 볼 카메라 고르기 (알림의 '라이브'·탭 열기마다): 요청한 카메라가 서버 설정에 있으면 그것,
     * 없으면 지금 보던 카메라(있으면), 그것도 없으면 첫 카메라. 카메라 목록을 아직 못 받았으면 요청을 그대로 둔다.
     */
    fun pick(requested: String, current: String, available: List<String>): String = when {
        available.isEmpty() -> requested.ifBlank { current }
        requested in available -> requested
        current in available -> current
        else -> available.first()
    }
}

/** 화면·로그에 보일 주소: 계정(user:pass@)을 지운다. 라이브 주소에는 보기 계정이 들어 있다. */
fun safeUrl(url: String): String = url.replace(Regex("//[^/@]+@"), "//")

private fun parseInstant(iso: String?): Instant? =
    iso?.let { runCatching { OffsetDateTime.parse(it).toInstant() }.getOrNull() }

/** 목록 화면에 띄울 송출 상태: "송출 중 · 3시간 12분". 송출 중이 아니면 null. */
fun streamingLabel(live: LiveStatus?, now: Instant): String? {
    if (live == null || !live.streaming) return null
    val since = parseInstant(live.since) ?: return "송출 중"
    val d = Duration.between(since, now).let { if (it.isNegative) Duration.ZERO else it }
    val h = d.toHours()
    val m = d.toMinutes() % 60
    return when {
        h > 0 -> "송출 중 · ${h}시간 ${m}분"
        m > 0 -> "송출 중 · ${m}분"
        else -> "송출 중 · 방금 시작"
    }
}

/** 오래 송출 중이면(켜 둔 채 잊었을 수 있음) 눈에 띄게 표시한다. */
fun streamingLong(live: LiveStatus?, now: Instant, hours: Long = 2): Boolean {
    if (live == null || !live.streaming) return false
    val since = parseInstant(live.since) ?: return false
    return Duration.between(since, now).toHours() >= hours
}

/** 서버 디스크 여유가 부족하면 목록 화면에 띄울 경고. 괜찮으면 null. 녹화는 자동으로 지우지 않으므로 사람이 정리해야 한다. */
fun diskWarning(h: Health?): String? {
    if (h == null || h.status != "degraded") return null
    val free = h.diskFreeGb?.let { "%.0f".format(it) + "GB" } ?: "부족"
    return "서버 디스크 여유 $free — 녹화 폴더를 정리하세요"
}

/** 아직 녹화 중인 클립을 다시 물어보기까지 기다릴 시간 (서버 Retry-After, 1~10초). */
fun retryDelayMs(retryAfterSec: Int): Long = retryAfterSec.coerceIn(1, 10) * 1000L

/** 지난 영상 화면: 1분 조각으로 앞뒤 이동, 재생 위치 ↔ 실제 시각. 시각은 UTC ISO8601 로 서버와 주고받는다. */
object RecordingNav {
    const val CHUNK_SEC = 60
    private val iso = DateTimeFormatter.ISO_OFFSET_DATE_TIME

    fun format(t: Instant): String = OffsetDateTime.ofInstant(t, ZoneOffset.UTC).format(iso)

    /** start 에서 seconds 만큼 옮긴 시각. 읽을 수 없으면 null. */
    fun shift(startIso: String, seconds: Long): String? = parseInstant(startIso)?.let { format(it.plusSeconds(seconds)) }

    /** 영상이 시작된 실제 시각 + 재생 위치 = 지금 보고 있는 장면의 실제 시각. */
    fun wallTime(videoStartIso: String?, positionMs: Long): String? =
        parseInstant(videoStartIso)?.let { format(it.plusMillis(positionMs.coerceAtLeast(0))) }

    /** 폰 시간대의 하루(0시 ~ 다음 날 0시)를 서버에 물을 UTC 구간으로. */
    fun dayRange(day: LocalDate, zone: ZoneId): Pair<String, String> =
        format(day.atStartOfDay(zone).toInstant()) to format(day.plusDays(1).atStartOfDay(zone).toInstant())

    /** 그 날의 "HH:mm" (폰 시간대) → UTC ISO8601. 형식이 틀리면 null. */
    fun atLocalTime(day: LocalDate, hhmm: String, zone: ZoneId): String? = runCatching {
        format(day.atTime(LocalTime.parse(hhmm.trim(), DateTimeFormatter.ofPattern("H:mm"))).atZone(zone).toInstant())
    }.getOrNull()

    /** 녹화 구간의 끝 시각. */
    fun spanEnd(startIso: String, durationSec: Double): String? =
        parseInstant(startIso)?.let { format(it.plusMillis((durationSec * 1000).toLong())) }
}
