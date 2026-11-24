package com.hawkeye.admin.domain

import org.junit.Assert.assertEquals
import org.junit.Test

class NotificationPolicyTest {
    @Test fun notifyShowsAlertWithLevelTitleAndCounts() {
        val a = NotificationPolicy.actionFor(notify = true, case = case(7, level = "WARNING"))
        assertEquals(
            NotificationAction.Show(caseId = 7, title = "주의: 결제 전 출구 접근", text = "손님 7이 결제 전에 출구로 갔어요.", cameraId = "cam1"),
            a,
        )
    }

    @Test fun eachAlarmLevelHasItsOwnTitle() {
        assertEquals("위험: 결제 없이 퇴장", (NotificationPolicy.actionFor(true, case(1, level = "HIGH_RISK")) as NotificationAction.Show).title)
        assertEquals("확인 필요: 일부만 결제", (NotificationPolicy.actionFor(true, case(1, level = "REVIEW")) as NotificationAction.Show).title)
    }

    @Test fun handledCaseCancelsItsNotification() {
        assertEquals(NotificationAction.Cancel(3), NotificationPolicy.actionFor(false, case(3, state = "resolved")))
        assertEquals(NotificationAction.Cancel(3), NotificationPolicy.actionFor(false, case(3, state = "auto_cleared")))
    }

    @Test fun quietUpdateDoesNothing() {
        assertEquals(NotificationAction.None, NotificationPolicy.actionFor(false, case(3, state = "active")))
    }

    @Test fun missedActiveCaseIsAnnouncedAfterReconnect() {
        assertEquals(
            NotificationAction.Show(caseId = 8, title = "확인 필요: 일부만 결제", text = "손님 8이 1개 중 0개만 결제했어요. 연결이 끊긴 동안 온 경보예요.", cameraId = "cam1"),
            NotificationPolicy.forMissed(case(8, level = "REVIEW", state = "active")),
        )
    }

    @Test fun missedCaseAlreadyHandledStaysSilent() {
        assertEquals(NotificationAction.None, NotificationPolicy.forMissed(case(8, state = "resolved")))
        assertEquals(NotificationAction.None, NotificationPolicy.forMissed(case(8, state = "auto_cleared")))
    }
}
