package com.hawkeye.admin.data

import com.hawkeye.admin.Fixtures
import kotlinx.coroutines.test.runTest
import okhttp3.mockwebserver.MockResponse
import okhttp3.mockwebserver.MockWebServer
import okio.Buffer
import org.junit.After
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Before
import org.junit.Test

class ApiClientTest {
    private val server = MockWebServer()
    private lateinit var api: ApiClient

    @Before fun setUp() {
        server.start()
        api = ApiClient(ServerSettings(baseUrl = server.url("/").toString().trimEnd('/'), apiKey = "secret"))
    }

    @After fun tearDown() = server.shutdown()

    @Test fun listCasesSendsKeyAndSinceAndParses() = runTest {
        server.enqueue(MockResponse().setBody(Fixtures.read("cases.json")))
        val cases = api.listCases(updatedSince = "2026-10-01T10:00:00+00:00")
        val req = server.takeRequest()
        assertEquals("secret", req.getHeader("X-API-Key"))
        assertEquals("/api/cases", req.requestUrl!!.encodedPath)
        assertEquals("2026-10-01T10:00:00+00:00", req.requestUrl!!.queryParameter("updated_since"))
        assertEquals(3, cases.size)
    }

    @Test fun resolvePostsChoiceAndNote() = runTest {
        server.enqueue(MockResponse().setBody(Fixtures.read("case_detail.json")))
        api.resolve(caseId = 1, resolution = Resolution.FALSE_ALARM, note = "지갑 꺼냄")
        val req = server.takeRequest()
        assertEquals("POST", req.method)
        assertEquals("/api/cases/1/resolution", req.path)
        assertEquals("""{"resolution":"false_alarm","note":"지갑 꺼냄"}""", req.body.readUtf8())
    }

    @Test fun snapshotReturnsBytesOrNullWhenMissing() = runTest {
        val jpeg = byteArrayOf(0xFF.toByte(), 0xD8.toByte(), 1, 2)
        server.enqueue(MockResponse().setBody(Buffer().write(jpeg)))
        server.enqueue(MockResponse().setResponseCode(404))
        assertArrayEquals(jpeg, api.snapshot(eventId = 3))
        assertNull(api.snapshot(eventId = 4))
        assertEquals("/api/events/3/snapshot.jpg", server.takeRequest().path)
    }

    @Test fun webSocketUrlCarriesKey() {
        assertEquals(server.url("/ws").toString().replace("http", "ws") + "?key=secret", api.webSocketUrl())
    }

    @Test(expected = ApiException::class)
    fun unauthorizedIsAnError() = runTest {
        server.enqueue(MockResponse().setResponseCode(401))
        api.listCases()
    }
}
