package com.hawkeye.admin.ui

import com.hawkeye.admin.domain.CameraNames
import java.time.OffsetDateTime
import java.time.ZoneId
import java.time.format.DateTimeFormatter

/** 단계 이름 (색 + 아이콘과 함께 보인다 — ui/components/LevelChip). */
fun levelLabel(level: String): String = when (level) {
    "HIGH_RISK" -> "위험"
    "REVIEW" -> "확인 필요"
    "WARNING" -> "주의"
    "CLEAR" -> "정상"
    else -> level
}

fun stateLabel(state: String, resolution: String?): String = when (state) {
    "active" -> "미처리"
    "resolved" -> resolutionLabel(resolution)
    "auto_cleared" -> "자동 해제"
    "pass" -> "통과"
    else -> state
}

fun resolutionLabel(resolution: String?): String = when (resolution) {
    "paid_confirmed" -> "결제 확인 완료"
    "false_alarm" -> "잘못된 알림"
    else -> "처리됨"
}

/** 목록 요약 한 줄: "결제 없이 퇴장, 미결제 2" / "3개 중 2개만 결제" / "결제 전에 출구로 감". */
fun caseSummaryLine(level: String, taken: Int, paid: Int, unpaid: Int): String = when (level) {
    "HIGH_RISK" -> "결제 없이 퇴장, 미결제 $unpaid"
    "REVIEW" -> "${taken}개 중 ${paid}개만 결제"
    "WARNING" -> "결제 전에 출구로 감"
    else -> "가져감 $taken, 결제 $paid"
}

/** 화면에 보일 카메라 이름. */
fun cameraLabel(cameraId: String): String = CameraNames.label(cameraId)

private val clock = DateTimeFormatter.ofPattern("HH:mm:ss")
private val hm = DateTimeFormatter.ofPattern("HH:mm")
private val dayClock = DateTimeFormatter.ofPattern("MM-dd HH:mm:ss")

/** 서버 시각(ISO8601)을 폰 시간대의 월-일 시:분:초로. */
fun localDateTime(iso: String?): String = iso?.let {
    runCatching { OffsetDateTime.parse(it).atZoneSameInstant(ZoneId.systemDefault()).format(dayClock) }.getOrDefault(it)
} ?: "-"

/** 서버 시각(ISO8601, UTC)을 폰 시간대의 시:분:초로. */
fun localTime(iso: String): String =
    runCatching { OffsetDateTime.parse(iso).atZoneSameInstant(ZoneId.systemDefault()).format(clock) }.getOrDefault(iso)

/** 서버 시각을 폰 시간대의 시:분으로 (목록). */
fun localHm(iso: String): String =
    runCatching { OffsetDateTime.parse(iso).atZoneSameInstant(ZoneId.systemDefault()).format(hm) }.getOrDefault("")
