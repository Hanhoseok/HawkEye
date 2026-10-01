package com.hawkeye.admin.domain

import com.hawkeye.admin.data.CaseSummary

enum class CaseFilter { ACTIVE, ALL }

/**
 * 앱이 들고 있는 사건 목록.
 *
 * 같은 사건이 WebSocket 과 '놓친 사건 다시 받기' 두 경로로 순서 없이 들어올 수 있어서,
 * updated_at 이 더 최신인 것만 받아들인다 (서버 시각은 항상 증가하는 같은 형식의 ISO8601 문자열).
 */
class CaseStore {
    private val cases = LinkedHashMap<Int, CaseSummary>()

    /** 반영했으면 true (새로 생겼거나 더 최신). */
    @Synchronized
    fun apply(case: CaseSummary): Boolean {
        val old = cases[case.id]
        if (old != null && old.updatedAt >= case.updatedAt) return false
        cases[case.id] = case
        return true
    }

    @Synchronized
    fun get(id: Int): CaseSummary? = cases[id]

    @Synchronized
    fun list(filter: CaseFilter): List<CaseSummary> = cases.values
        .filter {
            when (filter) {
                CaseFilter.ACTIVE -> it.state == "active"
                CaseFilter.ALL -> it.state != "pass" // 경보 없이 통과한 손님은 목록에 안 보인다
            }
        }
        .sortedByDescending { it.updatedAt }

    @Synchronized
    fun latestUpdatedAt(): String? = cases.values.maxOfOrNull { it.updatedAt }
}
