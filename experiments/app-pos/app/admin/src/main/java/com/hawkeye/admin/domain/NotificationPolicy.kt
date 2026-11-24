package com.hawkeye.admin.domain

import com.hawkeye.admin.data.CaseSummary

sealed interface NotificationAction {
    /** cameraId: 알림의 '라이브' 단추가 열 카메라 */
    data class Show(val caseId: Int, val title: String, val text: String, val cameraId: String = "") : NotificationAction
    data class Cancel(val caseId: Int) : NotificationAction
    data object None : NotificationAction
}

/** 서버 메시지 하나에 대해 폰 알림을 띄울지, 지울지 정한다. 띄울지는 서버의 notify 를 따른다. */
object NotificationPolicy {
    fun actionFor(notify: Boolean, case: CaseSummary): NotificationAction = when {
        notify -> NotificationAction.Show(case.id, titleFor(case.level), sentence(case), case.cameraId)
        case.state == "resolved" || case.state == "auto_cleared" -> NotificationAction.Cancel(case.id)
        else -> NotificationAction.None
    }

    /** 연결이 끊긴 동안 생겨서 다시 받아온 사건. 아직 처리 안 된 것만 알린다. */
    fun forMissed(case: CaseSummary): NotificationAction =
        if (case.state == "active") {
            NotificationAction.Show(case.id, titleFor(case.level), sentence(case) + " 연결이 끊긴 동안 온 경보예요.", case.cameraId)
        } else {
            NotificationAction.None
        }

    fun titleFor(level: String): String = when (level) {
        "HIGH_RISK" -> "위험: 결제 없이 퇴장"
        "REVIEW" -> "확인 필요: 일부만 결제"
        "WARNING" -> "주의: 결제 전 출구 접근"
        else -> "알림"
    }

    /** 알림 본문 한 문장: "손님 3이 과자 1, 음료 1을 결제하지 않고 나갔어요." */
    fun sentence(c: CaseSummary): String {
        val who = Korean.subject("손님 ${c.personId}")
        return when (c.level) {
            "HIGH_RISK" -> {
                val missing = c.items.orEmpty().filter { it.taken > it.paid }.map { "${it.name} ${it.taken - it.paid}" }
                if (missing.isEmpty()) "$who 결제하지 않고 나갔어요 (미결제 ${c.unpaid}개)."
                else "$who ${Korean.objectOf(missing.joinToString(", "))} 결제하지 않고 나갔어요."
            }
            "REVIEW" -> "$who ${c.taken}개 중 ${c.paid}개만 결제했어요."
            "WARNING" -> "$who 결제 전에 출구로 갔어요."
            else -> "$who 가져감 ${c.taken}, 결제 ${c.paid}."
        }
    }
}
