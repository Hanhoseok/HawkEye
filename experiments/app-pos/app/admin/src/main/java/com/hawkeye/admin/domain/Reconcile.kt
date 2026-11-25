package com.hawkeye.admin.domain

import com.hawkeye.admin.data.CaseSummary

/**
 * 재연결 뒤 미처리 목록 맞추기.
 * 증분 조회(updated_since + limit)는 잘릴 수 있어서, 끊긴 동안 다른 폰에서 처리된 사건이 로컬에 '미처리'로 남을 수 있다.
 * 서버의 미처리 전체 응답(잘리지 않은 것)에 없는데 로컬에선 미처리인 사건은 상세로 다시 확인해야 한다.
 * 응답이 잘렸으면(truncated) 빠졌다고 처리된 것으로 보지 않는다 -> 다시 확인할 것 없음.
 */
object Reconcile {
    fun staleActive(localActive: List<CaseSummary>, serverActive: List<CaseSummary>, truncated: Boolean): List<Int> {
        if (truncated) return emptyList()
        val ids = serverActive.mapTo(HashSet()) { it.id }
        return localActive.filter { it.id !in ids }.map { it.id }
    }
}
