package com.hawkeye.admin.domain

import com.hawkeye.admin.data.RecordingSpan
import java.time.Instant
import java.time.OffsetDateTime

/**
 * 지난 영상의 막대: 녹화된 구간과 경보 시각을 [from, to) 안의 0~1 비율로 바꾼다.
 * 하루 막대(0시~24시)와 재생 중인 구간 막대에 같이 쓴다. 시각을 읽을 수 없는 항목은 건너뛴다.
 */
object DayBar {
    data class Segment(val start: Float, val end: Float)

    fun fraction(t: Instant, from: Instant, to: Instant): Float {
        val total = (to.toEpochMilli() - from.toEpochMilli()).toFloat()
        if (total <= 0f) return 0f
        return ((t.toEpochMilli() - from.toEpochMilli()) / total).coerceIn(0f, 1f)
    }

    /** 녹화 구간을 막대 위 칠할 자리로. 막대 밖으로 나간 부분은 잘라낸다. */
    fun segments(spans: List<RecordingSpan>, from: Instant, to: Instant): List<Segment> =
        spans.mapNotNull { s ->
            val start = parse(s.start) ?: return@mapNotNull null
            val end = start.plusMillis((s.duration * 1000).toLong())
            if (!end.isAfter(from) || !start.isBefore(to)) return@mapNotNull null
            Segment(fraction(maxOf(start, from), from, to), fraction(minOf(end, to), from, to))
        }.sortedBy { it.start }

    /** 막대 안에 드는 경보 시각의 위치. */
    fun marks(times: List<Instant>, from: Instant, to: Instant): List<Float> =
        times.filter { !it.isBefore(from) && it.isBefore(to) }.map { fraction(it, from, to) }.sorted()

    /** 녹화 구간 하나 안에 든 경보 수 (목록의 '경보 3'). */
    fun countIn(times: List<Instant>, span: RecordingSpan): Int {
        val start = parse(span.start) ?: return 0
        val end = start.plusMillis((span.duration * 1000).toLong())
        return times.count { !it.isBefore(start) && it.isBefore(end) }
    }

    fun parse(iso: String?): Instant? = iso?.let { runCatching { OffsetDateTime.parse(it).toInstant() }.getOrNull() }
}
