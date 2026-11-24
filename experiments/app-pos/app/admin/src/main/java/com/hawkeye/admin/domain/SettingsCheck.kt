package com.hawkeye.admin.domain

import com.hawkeye.admin.data.ApiException
import java.io.IOException

/** 설정 화면 '연결 확인' 결과. 실패는 이유별로 무엇을 확인할지 다르게 안내한다. */
sealed interface CheckResult {
    data class Ok(val cameras: Int, val recordings: Boolean) : CheckResult
    /** 서버에 닿지 않음: Tailscale 꺼짐, 주소 틀림, 서버 꺼짐, 시간 초과 */
    data object Unreachable : CheckResult
    /** 서버는 닿았지만 API 키가 틀림 (401/403) */
    data object BadKey : CheckResult
    data class Failed(val message: String) : CheckResult
}

object SettingsCheck {
    fun classify(t: Throwable): CheckResult = when {
        t is ApiException && (t.code == 401 || t.code == 403) -> CheckResult.BadKey
        t is ApiException -> CheckResult.Failed("서버 오류 (HTTP ${t.code})")
        t is IOException -> CheckResult.Unreachable
        t is IllegalArgumentException -> CheckResult.Failed("주소 형식이 맞지 않아요. http://로 시작해야 해요.")
        else -> CheckResult.Failed(t.message ?: t.javaClass.simpleName)
    }
}
