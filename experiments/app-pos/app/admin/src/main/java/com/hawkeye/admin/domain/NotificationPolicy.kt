package com.hawkeye.admin.domain

import com.hawkeye.admin.data.CaseSummary

sealed interface NotificationAction {
    data class Show(val caseId: Int, val title: String, val text: String) : NotificationAction
    data class Cancel(val caseId: Int) : NotificationAction
    data object None : NotificationAction
}

/** 서버 메시지 하나에 대해 폰 알림을 띄울지, 지울지 정한다. 띄울지는 서버의 notify 를 따른다. */
object NotificationPolicy {
    fun actionFor(notify: Boolean, case: CaseSummary): NotificationAction = when {
        notify -> NotificationAction.Show(
            caseId = case.id,
            title = titleFor(case.level),
            text = "손님 ${case.personId} · 감지 ${case.taken} / 결제 ${case.paid}",
        )
        case.state == "resolved" || case.state == "auto_cleared" -> NotificationAction.Cancel(case.id)
        else -> NotificationAction.None
    }

    /** 연결이 끊긴 동안 생겨서 다시 받아온 사건. 아직 처리 안 된 것만 알린다. */
    fun forMissed(case: CaseSummary): NotificationAction =
        if (case.state == "active") {
            NotificationAction.Show(
                caseId = case.id,
                title = titleFor(case.level),
                text = "손님 ${case.personId} · 감지 ${case.taken} / 결제 ${case.paid} (연결이 끊긴 동안 발생)",
            )
        } else {
            NotificationAction.None
        }

    fun titleFor(level: String): String = when (level) {
        "HIGH_RISK" -> "미결제 퇴장"
        "REVIEW" -> "확인 필요"
        "WARNING" -> "출구 접근 · 미결제 의심"
        else -> "알림"
    }
}
