package com.hawkeye.admin.domain

import com.hawkeye.admin.data.CaseSummary
import java.time.Instant
import java.time.OffsetDateTime
import java.time.ZoneId
import java.time.format.DateTimeFormatter

/** 상태 카드의 색. 경보 목록에선 가장 급한 미처리 경보의 단계, 사건 상세에선 그 사건 (처리됨 = OK). */
enum class Tone { DANGER, REVIEW, WARN, CALM, OFF, OK }

/** 상태 카드 단추가 하는 일. */
enum class StatusAction { OPEN_CASE, LIVE, RECONNECT }

/**
 * 사건 목록을 서버와 맞췄는가. 앱이 들고 있는 목록은 일부일 수 있어서(서버 limit), 미처리 사건은 따로 전부 받는다.
 * 한 번도 못 받았거나 실패했으면 '경보 없음'이라고 단정하지 않는다.
 */
sealed interface SyncState {
    data object NotYet : SyncState
    data object Syncing : SyncState
    /** truncated: 미처리 사건이 서버 최대 개수(1000)만큼 와서 더 있을 수 있음 */
    data class Done(val truncated: Boolean = false) : SyncState
    data class Failed(val reason: String?) : SyncState
}

data class StatusSummary(
    val tone: Tone,
    /** 카드 왼쪽 위 작은 표시 ("위험 경보", "정상" ...) */
    val label: String,
    /** 카드 오른쪽 위 (시각, "오늘 경보 4건" ...) */
    val meta: String,
    /** 한 문장 ("손님 3이 결제 없이 나갔어요") */
    val title: String,
    val detail: String,
    val action: StatusAction,
    val actionLabel: String,
    /** OPEN_CASE 일 때 열 사건 */
    val caseId: Int? = null,
)

/**
 * 사건 목록과 서버 연결 상태로 상태 카드 내용을 정한다 (디자인: .ai-review/admin-app-design/c2-screens.html 1번 묶음).
 * - 서버와 끊겼으면(연결 시도 중은 제외) 회색 카드 — 목록이 오래된 것일 수 있어서 가장 먼저 알린다.
 * - 아니면 미처리 사건 중 위험 > 확인 필요 > 주의 순, 같은 단계면 최근 것.
 * - 미처리가 없으면: 목록을 아직 못 맞췄으면 '확인 중', 실패했으면 '받지 못함', 맞췄으면 평소 카드.
 *   (받은 미처리 경보가 있으면 동기화 상태와 상관없이 보여 준다 — 실제로 있는 경보라서.)
 */
object StatusSummaries {
    private val clock = DateTimeFormatter.ofPattern("HH:mm")

    fun of(
        cases: List<CaseSummary>,
        offline: Boolean,
        sync: SyncState,
        now: Instant,
        zone: ZoneId = ZoneId.systemDefault(),
    ): StatusSummary {
        if (offline) {
            return StatusSummary(
                tone = Tone.OFF, label = "연결 끊김", meta = "다시 연결하는 중",
                title = "경보 서버와 연결이 끊겼어요",
                detail = "폰의 Tailscale이 켜져 있는지 확인해 주세요.",
                action = StatusAction.RECONNECT, actionLabel = "지금 다시 연결",
            )
        }
        val top = cases.filter { it.state == "active" && rank(it.level) > 0 }
            .maxWithOrNull(compareBy<CaseSummary>({ rank(it.level) }, { it.updatedAt }))
        if (top == null) return noActive(cases, sync, now, zone)
        val who = "손님 ${top.personId}"
        val time = localClock(top.updatedAt, zone)
        val cam = CameraNames.label(top.cameraId)
        return when (top.level) {
            "HIGH_RISK" -> StatusSummary(
                tone = Tone.DANGER, label = "위험 경보", meta = time,
                title = "${Korean.subject(who)} 결제 없이 나갔어요",
                detail = "${unpaidPhrase(top)} $cam",
                action = StatusAction.OPEN_CASE, actionLabel = "영상 보고 처리하기", caseId = top.id,
            )
            "REVIEW" -> StatusSummary(
                tone = Tone.REVIEW, label = "확인 필요", meta = time,
                title = "${Korean.subject(who)} ${top.taken}개 중 ${top.paid}개만 결제했어요",
                detail = "영상으로 확인해 주세요. $cam",
                action = StatusAction.OPEN_CASE, actionLabel = "영상 보고 처리하기", caseId = top.id,
            )
            else -> StatusSummary(
                tone = Tone.WARN, label = "주의", meta = time,
                title = "${Korean.subject(who)} 결제 전에 출구로 갔어요",
                detail = "출구 근처를 영상으로 확인해 주세요. $cam",
                action = StatusAction.OPEN_CASE, actionLabel = "영상 보고 처리하기", caseId = top.id,
            )
        }
    }

    private fun noActive(cases: List<CaseSummary>, sync: SyncState, now: Instant, zone: ZoneId): StatusSummary = when (sync) {
        is SyncState.Failed -> StatusSummary(
            tone = Tone.OFF, label = "목록 오류", meta = "",
            title = "경보 목록을 받지 못했어요",
            detail = "서버 주소와 연결 상태를 확인해 주세요." + (sync.reason?.let { " ($it)" } ?: ""),
            action = StatusAction.RECONNECT, actionLabel = "다시 불러오기",
        )
        is SyncState.Done -> {
            val today = countToday(cases, now, zone)
            StatusSummary(
                tone = Tone.CALM, label = "정상", meta = "오늘 받은 경보 ${today}건",
                title = "지금은 확인할 경보가 없어요",
                detail = if (today == 0) "새 경보가 오면 여기에 보여요." else "오늘 받은 경보는 모두 처리했어요.",
                action = StatusAction.LIVE, actionLabel = "라이브 보기",
            )
        }
        else -> StatusSummary(
            tone = Tone.CALM, label = "확인 중", meta = "",
            title = "경보 목록을 확인하고 있어요",
            detail = "서버에서 처리하지 않은 경보를 받아 오는 중이에요.",
            action = StatusAction.LIVE, actionLabel = "라이브 보기",
        )
    }

    /** 아래 탭 배지 글자. 서버가 최대 개수만큼 줬으면 더 있을 수 있어 '+'. */
    fun badgeText(count: Int, sync: SyncState): String? = when {
        count <= 0 -> null
        sync is SyncState.Done && sync.truncated -> "${count}+"
        count > 99 -> "99+"
        else -> "$count"
    }

    /** 미처리 사건 수 (아래 탭 배지). */
    fun activeCount(cases: List<CaseSummary>): Int = cases.count { it.state == "active" && rank(it.level) > 0 }

    /** 단계 순위: 높을수록 급함. CLEAR 등은 0. */
    fun rank(level: String): Int = when (level) {
        "HIGH_RISK" -> 3
        "REVIEW" -> 2
        "WARNING" -> 1
        else -> 0
    }

    /** "과자 1, 음료 1을 결제하지 않았어요." 품목이 없으면 "미결제 2개예요." */
    fun unpaidPhrase(c: CaseSummary): String {
        val missing = c.items.orEmpty().filter { it.taken > it.paid }.map { "${it.name} ${it.taken - it.paid}" }
        return if (missing.isEmpty()) "미결제 ${c.unpaid}개예요." else "${Korean.objectOf(missing.joinToString(", "))} 결제하지 않았어요."
    }

    private fun countToday(cases: List<CaseSummary>, now: Instant, zone: ZoneId): Int {
        val today = now.atZone(zone).toLocalDate()
        return cases.count { c ->
            c.state != "pass" && parse(c.createdAt.ifBlank { c.updatedAt })?.atZone(zone)?.toLocalDate() == today
        }
    }

    private fun localClock(iso: String, zone: ZoneId): String =
        parse(iso)?.atZone(zone)?.format(clock) ?: ""

    private fun parse(iso: String): Instant? = runCatching { OffsetDateTime.parse(iso).toInstant() }.getOrNull()
}

/** 화면에 보일 카메라 이름. */
object CameraNames {
    fun label(cameraId: String): String = when (cameraId) {
        "cam1" -> "cam1 전체"
        "cam2" -> "cam2 선반"
        else -> cameraId
    }
}

/** 숫자로 끝나는 말 뒤의 조사 (손님 3이 / 손님 5가, 음료 1을 / 음료 2를). 마지막 글자가 숫자면 읽는 소리로 받침을 본다. */
object Korean {
    fun subject(word: String): String = word + if (hasFinal(word)) "이" else "가"

    fun objectOf(word: String): String = word + if (hasFinal(word)) "을" else "를"

    fun hasFinal(word: String): Boolean {
        val c = word.trimEnd().lastOrNull() ?: return false
        if (c.isDigit()) return c in "0136780"   // 영·일·삼·육·칠·팔, 0으로 끝나면 십·백·천·만 — 모두 받침
        if (c in '가'..'힣') return (c - '가') % 28 != 0
        return false
    }
}
