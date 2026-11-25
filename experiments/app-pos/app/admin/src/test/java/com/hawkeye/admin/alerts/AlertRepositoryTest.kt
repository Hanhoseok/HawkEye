package com.hawkeye.admin.alerts

import com.hawkeye.admin.data.ApiClient
import com.hawkeye.admin.data.ServerSettings
import com.hawkeye.admin.domain.CaseFilter
import com.hawkeye.admin.domain.StatusSummaries
import com.hawkeye.admin.domain.SyncState
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.test.runTest
import okhttp3.mockwebserver.Dispatcher
import okhttp3.mockwebserver.MockResponse
import okhttp3.mockwebserver.MockWebServer
import okhttp3.mockwebserver.RecordedRequest
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import java.time.Instant

/** 저장소의 서버 맞추기: 미처리 대조와 경보 장면 시각 조회의 실패 처리 (계획 검토 3회차 R3-F1, R3-F2). */
class AlertRepositoryTest {
    private val server = MockWebServer()
    private lateinit var repo: AlertRepository
    /** 경로 → 응답들 (차례로, 끝나면 마지막 반복) */
    private val routes = HashMap<String, MutableList<MockResponse>>()

    private fun route(path: String, vararg r: MockResponse) { routes[path] = r.toMutableList() }
    private fun json(body: String) = MockResponse().setBody(body).setHeader("Content-Type", "application/json")
    private fun status(code: Int) = MockResponse().setResponseCode(code).setBody("{\"detail\":\"x\"}")

    private fun case(id: Int, state: String, updated: String, level: String = "HIGH_RISK", camera: String = "cam1") =
        """{"id":$id,"run_id":"r","camera_id":"$camera","person_id":$id,"level":"$level","state":"$state",""" +
            """"taken":2,"paid":0,"unpaid":2,"has_snapshot":false,"updated_at":"$updated","created_at":"$updated"}"""

    private fun detail(caseJson: String, ts: String?, received: String) =
        """{${caseJson.removePrefix("{").removeSuffix("}")},"events":[{"id":1,"level":"HIGH_RISK",""" +
            """"timestamp":${ts?.let { "\"$it\"" } ?: "null"},"received_at":"$received"}],"resolutions":[]}"""

    @Before fun setUp() {
        server.dispatcher = object : Dispatcher() {
            override fun dispatch(request: RecordedRequest): MockResponse {
                val url = request.requestUrl!!
                val key = if (url.encodedPath == "/api/cases" && url.queryParameter("state") == "active") "active" else url.encodedPath
                val list = routes[key] ?: return status(404)
                return if (list.size > 1) list.removeAt(0) else list[0]
            }
        }
        server.start()
        val api = ApiClient(ServerSettings(baseUrl = server.url("/").toString().trimEnd('/'), apiKey = null))
        repo = AlertRepository { api }
    }

    @After fun tearDown() = server.shutdown()

    @Test fun recheckFailureIsReportedAndRetried() = runTest {
        // 1) 처음: A 미처리
        route("/api/cases", json("[${case(1, "active", "2026-11-24T05:00:00Z")}]"))
        route("active", json("[${case(1, "active", "2026-11-24T05:00:00Z")}]"))
        repo.refresh()
        assertEquals(SyncState.Done(false), repo.sync.value)

        // 2) 재연결: 증분엔 다른 사건만, 미처리 응답엔 A 없음, A 상세는 503
        route("/api/cases", json("[${case(2, "auto_cleared", "2026-11-24T06:00:00Z", level = "WARNING")}]"))
        route("active", json("[]"))
        route("/api/cases/1", status(503))
        runCatching { repo.refresh() }
        assertTrue(repo.sync.value is SyncState.Failed)
        assertEquals("active", repo.case(1)!!.state)                    // 아직 못 바로잡음 — 그래도 Done 이라 하지 않음

        // 3) 다음 재시도: 상세가 처리됨으로 옴 → 바로잡힘
        route("/api/cases", json("[]"))
        route("/api/cases/1", json(detail(case(1, "resolved", "2026-11-24T05:30:00Z"), "2026-11-24T05:00:00Z", "2026-11-24T05:00:01Z")))
        repo.refresh()
        assertEquals(SyncState.Done(false), repo.sync.value)
        assertEquals("resolved", repo.case(1)!!.state)
        assertEquals(0, StatusSummaries.activeCount(repo.cases(CaseFilter.ALL).first()))
    }

    @Test fun vanishedCaseIsRemoved() = runTest {
        route("/api/cases", json("[${case(1, "active", "2026-11-24T05:00:00Z")}]"))
        route("active", json("[${case(1, "active", "2026-11-24T05:00:00Z")}]"))
        repo.refresh()
        route("/api/cases", json("[]"))
        route("active", json("[]"))
        route("/api/cases/1", status(404))
        repo.refresh()
        assertEquals(null, repo.case(1))
        assertEquals(SyncState.Done(false), repo.sync.value)
    }

    @Test fun firstRefreshFailureIsNotEmptyList() = runTest {
        route("/api/cases", status(500))
        runCatching { repo.refresh() }
        assertTrue(repo.sync.value is SyncState.Failed)
    }

    @Test fun alertTimeLookupFailureIsRetried() = runTest {
        val c = case(1, "active", "2026-11-24T05:32:10Z")
        route("/api/cases", json("[$c]"))
        route("active", json("[$c]"))
        repo.refresh()
        val from = Instant.parse("2026-11-23T15:00:00Z")
        val to = Instant.parse("2026-11-24T15:00:00Z")

        route("/api/cases/1", status(500))
        val (first, complete1) = repo.alertTimes("cam1", from, to)
        assertTrue(first.isEmpty())
        assertFalse(complete1)                                           // 실패는 '일부만' 으로 알림

        route("/api/cases/1", json(detail(c, "2026-11-24T14:32:08+09:00", "2026-11-24T05:32:10Z")))
        val (second, complete2) = repo.alertTimes("cam1", from, to)      // 같은 updated_at 이라도 다시 조회
        assertEquals(listOf(Instant.parse("2026-11-24T05:32:08Z")), second.map { it.at })
        assertTrue(complete2)
    }
}
