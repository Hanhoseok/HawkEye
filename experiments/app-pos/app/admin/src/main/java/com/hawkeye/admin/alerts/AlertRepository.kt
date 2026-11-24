package com.hawkeye.admin.alerts

import com.hawkeye.admin.data.ApiClient
import com.hawkeye.admin.data.ApiException
import com.hawkeye.admin.data.AppConfig
import com.hawkeye.admin.data.CaseClips
import com.hawkeye.admin.data.CaseDetail
import com.hawkeye.admin.data.CaseSummary
import com.hawkeye.admin.data.Health
import com.hawkeye.admin.data.RecordingList
import com.hawkeye.admin.data.Resolution
import com.hawkeye.admin.data.VideoState
import com.hawkeye.admin.domain.AlertTime
import com.hawkeye.admin.domain.AlertTimes
import com.hawkeye.admin.domain.CaseFilter
import com.hawkeye.admin.domain.CaseStore
import com.hawkeye.admin.domain.Links
import com.hawkeye.admin.domain.Reconcile
import com.hawkeye.admin.domain.SyncState
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.map
import java.time.Instant

sealed interface Connection {
    data object Connecting : Connection
    data object Connected : Connection
    data class Disconnected(val reason: String?) : Connection
}

/** 화면과 백그라운드 연결(AlertService)이 함께 쓰는 사건 목록·서버 호출 창구. */
class AlertRepository(private val apiFactory: () -> ApiClient) {
    private val store = CaseStore()
    private val version = MutableStateFlow(0L)
    private val _connection = MutableStateFlow<Connection>(Connection.Connecting)
    val connection: StateFlow<Connection> = _connection.asStateFlow()
    private var everConnected = false
    private val _offline = MutableStateFlow(false)
    /** 경보를 못 받는 상태인가 (끊김, 또는 한 번 연결됐다 끊긴 뒤 다시 붙는 중). 상태 카드가 쓴다. */
    val offline: StateFlow<Boolean> = _offline.asStateFlow()
    private val _sync = MutableStateFlow<SyncState>(SyncState.NotYet)
    /** 사건 목록을 서버와 맞췄는가. 맞추기 전·실패하면 화면이 '경보 없음'이라고 단정하지 않는다. */
    val sync: StateFlow<SyncState> = _sync.asStateFlow()
    /** 경보 장면 시각 캐시: 사건 id → (그때의 updated_at, 시각) */
    private val alertTimeCache = HashMap<Int, Pair<String, AlertTime?>>()

    fun api(): ApiClient = apiFactory()

    fun cases(filter: CaseFilter): Flow<List<CaseSummary>> = version.map { store.list(filter) }

    fun case(id: Int): CaseSummary? = store.get(id)

    fun apply(case: CaseSummary) {
        if (store.apply(case)) version.value++
    }

    fun setConnection(c: Connection) {
        _connection.value = c
        if (c is Connection.Connected) everConnected = true
        _offline.value = Links.offline(c is Connection.Connected, c is Connection.Disconnected, everConnected)
    }

    /**
     * 놓친 사건까지 다시 받는다 (처음이면 최근 것부터). 이번에 처음 본 사건을 돌려준다.
     * 증분 조회는 서버 limit(기본 100)에 잘릴 수 있어서, 처리 안 된 사건은 따로 전부(최대 1000) 받는다.
     * 실패하면 sync 를 Failed 로 두고 예외를 그대로 던진다 (빈 목록으로 삼키지 않음).
     */
    suspend fun refresh(): List<CaseSummary> {
        if (_sync.value !is SyncState.Done) _sync.value = SyncState.Syncing
        try {
            val api = api()
            val fresh = mutableListOf<CaseSummary>()
            fun take(c: CaseSummary) {
                val isNew = store.get(c.id) == null
                apply(c)
                if (isNew) fresh += c
            }
            api.listCases(updatedSince = store.latestUpdatedAt()).forEach(::take)
            val active = api.listCases(state = "active", limit = ACTIVE_LIMIT)
            val truncated = active.size >= ACTIVE_LIMIT
            active.forEach(::take)
            // 로컬엔 미처리인데 서버 미처리 응답에 없는 사건: 끊긴 동안 처리됐을 수 있으니 상세로 다시 확인.
            // apply 는 updated_at 이 더 최신일 때만 바꾸므로 그 사이 WebSocket 으로 온 더 새 갱신을 덮지 않는다.
            // 404(서버에서 사라짐)만 지우고, 그 밖의 오류는 밖으로 던져 Failed 로 둔다 — 대조를 못 마쳤는데 Done 이라 하지 않게.
            Reconcile.staleActive(store.list(CaseFilter.ACTIVE), active, truncated).forEach { id ->
                try {
                    apply(api.getCase(id).case)
                } catch (e: ApiException) {
                    if (e.code != 404) throw e
                    if (store.remove(id)) version.value++
                }
            }
            _sync.value = SyncState.Done(truncated = truncated)
            return fresh
        } catch (e: Exception) {
            if (e is kotlinx.coroutines.CancellationException) throw e
            _sync.value = SyncState.Failed(e.message)
            throw e
        }
    }

    /**
     * 지난 영상 막대에 찍을 경보 장면 시각 (그 카메라, [from, to) 근처 사건). 사건 상세를 받아 계산하고 캐시한다.
     * 한 번에 최대 [maxFetch] 건만 새로 받는다 — 넘거나 조회에 실패한 사건이 있으면 complete = false (실패는 캐시하지 않아 다음에 다시 받음).
     */
    suspend fun alertTimes(cameraId: String, from: Instant, to: Instant, maxFetch: Int = 50): Pair<List<AlertTime>, Boolean> {
        val candidates = AlertTimes.candidates(store.list(CaseFilter.ALL), cameraId, from)
        var fetched = 0
        var complete = true
        val out = mutableListOf<AlertTime>()
        for (c in candidates) {
            val cached = synchronized(alertTimeCache) { alertTimeCache[c.id] }
            val time = if (cached != null && cached.first == c.updatedAt) {
                cached.second
            } else if (fetched < maxFetch) {
                fetched++
                val detail = try {
                    api().getCase(c.id)
                } catch (e: Exception) {
                    if (e is kotlinx.coroutines.CancellationException) throw e
                    null
                }
                if (detail == null) {
                    complete = false          // 조회 실패: 캐시하지 않음 → 다음에 다시 시도
                    null
                } else {
                    val t = AlertTimes.anchor(detail)
                    synchronized(alertTimeCache) { alertTimeCache[c.id] = c.updatedAt to t }
                    t
                }
            } else {
                complete = false
                null
            }
            if (time != null && !time.at.isBefore(from) && time.at.isBefore(to)) out += time
        }
        return out.sortedBy { it.at } to complete
    }

    suspend fun detail(caseId: Int): CaseDetail = api().getCase(caseId).also { apply(it.case) }

    suspend fun resolve(caseId: Int, resolution: Resolution, note: String?) {
        apply(api().resolve(caseId, resolution, note))
    }

    suspend fun snapshot(eventId: Int): ByteArray? = api().snapshot(eventId)

    suspend fun config(): AppConfig = api().config()

    suspend fun caseClips(caseId: Int): CaseClips = api().caseClips(caseId)

    suspend fun health(): Health = api().health()

    suspend fun recordings(cameraId: String, start: String? = null, end: String? = null): RecordingList =
        api().recordings(cameraId, start, end)

    suspend fun videoState(absoluteUrl: String): VideoState = api().videoState(absoluteUrl)

    companion object {
        /** 서버 /api/cases 의 limit 최대값 */
        const val ACTIVE_LIMIT = 1000
    }
}
