package com.hawkeye.admin.domain

import com.hawkeye.admin.data.CaseDetail
import com.hawkeye.admin.data.CaseSummary
import java.time.Duration
import java.time.Instant
import java.time.OffsetDateTime

/** 사건 하나의 경보 장면 시각. exact = false 면 장면 시각을 몰라 서버가 받은 시각을 대신 쓴 것. */
data class AlertTime(val caseId: Int, val at: Instant, val exact: Boolean)

/**
 * 지난 영상 막대에 찍을 경보 시각. 서버 클립 기준(case_window)과 같은 규칙:
 * 마지막 CLEAR 아닌 경보의 timestamp(오프셋 있는 절대 시각) → 못 읽으면 그 경보의 received_at (부정확).
 * 사건 created_at 은 첫 수신 시각이라 쓰지 않는다 (늦게 받거나 단계가 올라간 사건에서 어긋남).
 */
object AlertTimes {
    /**
     * 상세를 받아 볼 사건: 그 카메라의 경보가 있었던 사건 중 막대 범위에 장면이 있을 수 있는 것.
     * 장면 시각은 서버가 받은 시각(마지막 갱신 updated_at)보다 늦을 수 없으므로, updated_at 이 막대 시작보다
     * 앞선 사건만 뺀다 (같은 서버 시계라 오차는 skew 만큼). 늦게 받은 경보도 빠지지 않게 위쪽 제한은 두지 않는다.
     */
    fun candidates(
        cases: List<CaseSummary>, cameraId: String, from: Instant,
        skew: Duration = Duration.ofMinutes(2),
    ): List<CaseSummary> = cases.filter { c ->
        // 자동 해제(주의 → CLEAR)된 사건은 지금 단계가 CLEAR 여도 경보가 있었으므로 넣는다. 경보 없이 통과한 손님(pass)만 뺀다.
        if (c.cameraId != cameraId || c.state == "pass") return@filter false
        val updated = parse(c.updatedAt) ?: return@filter true   // 시각을 못 읽으면 일단 조회
        !updated.isBefore(from.minus(skew))
    }

    fun anchor(detail: CaseDetail): AlertTime? {
        val e = detail.events.lastOrNull { it.level != "CLEAR" } ?: return null   // 경보가 없던 기록만 있으면 표시 안 함
        parse(e.timestamp)?.let { return AlertTime(detail.case.id, it, exact = true) }
        parse(e.receivedAt)?.let { return AlertTime(detail.case.id, it, exact = false) }
        return null
    }

    private fun parse(iso: String?): Instant? =
        iso?.takeIf { it.isNotBlank() }?.let { runCatching { OffsetDateTime.parse(it).toInstant() }.getOrNull() }
}
