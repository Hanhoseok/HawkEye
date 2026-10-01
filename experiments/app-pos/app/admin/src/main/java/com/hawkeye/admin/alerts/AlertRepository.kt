package com.hawkeye.admin.alerts

import com.hawkeye.admin.data.ApiClient
import com.hawkeye.admin.data.AppConfig
import com.hawkeye.admin.data.CaseDetail
import com.hawkeye.admin.data.CaseSummary
import com.hawkeye.admin.data.Resolution
import com.hawkeye.admin.domain.CaseFilter
import com.hawkeye.admin.domain.CaseStore
import com.hawkeye.admin.settings.SettingsStore
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.map

sealed interface Connection {
    data object Connecting : Connection
    data object Connected : Connection
    data class Disconnected(val reason: String?) : Connection
}

/** 화면과 백그라운드 연결(AlertService)이 함께 쓰는 사건 목록·서버 호출 창구. */
class AlertRepository(private val settings: SettingsStore) {
    private val store = CaseStore()
    private val version = MutableStateFlow(0L)
    private val _connection = MutableStateFlow<Connection>(Connection.Disconnected(null))
    val connection: StateFlow<Connection> = _connection.asStateFlow()

    fun api(): ApiClient = ApiClient(settings.settings.value)

    fun cases(filter: CaseFilter): Flow<List<CaseSummary>> = version.map { store.list(filter) }

    fun case(id: Int): CaseSummary? = store.get(id)

    fun apply(case: CaseSummary) {
        if (store.apply(case)) version.value++
    }

    fun setConnection(c: Connection) {
        _connection.value = c
    }

    /** 놓친 사건까지 다시 받는다 (처음이면 전부). 이번에 처음 본 사건을 돌려준다. */
    suspend fun refresh(): List<CaseSummary> =
        api().listCases(updatedSince = store.latestUpdatedAt()).mapNotNull { c ->
            val isNew = store.get(c.id) == null
            apply(c)
            c.takeIf { isNew }
        }

    suspend fun detail(caseId: Int): CaseDetail = api().getCase(caseId).also { apply(it.case) }

    suspend fun resolve(caseId: Int, resolution: Resolution, note: String?) {
        apply(api().resolve(caseId, resolution, note))
    }

    suspend fun snapshot(eventId: Int): ByteArray? = api().snapshot(eventId)

    suspend fun config(): AppConfig = api().config()
}
