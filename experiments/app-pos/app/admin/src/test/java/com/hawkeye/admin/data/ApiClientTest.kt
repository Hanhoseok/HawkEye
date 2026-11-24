package com.hawkeye.admin.data

import com.hawkeye.admin.Fixtures
import kotlinx.coroutines.test.runTest
import okhttp3.mockwebserver.MockResponse
import okhttp3.mockwebserver.MockWebServer
import okhttp3.HttpUrl.Companion.toHttpUrl
import okio.Buffer
import org.junit.After
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
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

    @Test fun caseClipsSendsKeyAndParses() = runTest {
        server.enqueue(MockResponse().setBody(Fixtures.read("case_clips.json")))
        val c = api.caseClips(1)
        val req = server.takeRequest()
        assertEquals("/api/cases/1/clips", req.requestUrl!!.encodedPath)
        assertEquals("secret", req.getHeader("X-API-Key"))
        assertEquals(2, c.clips.size)
    }

    @Test fun recordingsSendsCameraAndRange() = runTest {
        server.enqueue(MockResponse().setBody(Fixtures.read("recordings.json")))
        api.recordings("cam1", start = "2026-11-24T09:00:00+09:00")
        val url = server.takeRequest().requestUrl!!
        assertEquals("cam1", url.queryParameter("camera_id"))
        assertEquals("2026-11-24T09:00:00+09:00", url.queryParameter("start"))   // '+' 가 공백으로 바뀌지 않게
        assertNull(url.queryParameter("end"))
    }

    @Test fun recordingVideoUrlKeepsOffsetIntact() {
        val u = api.recordingVideoUrl("cam1", "2026-11-24T09:00:00+09:00", 60).toHttpUrl()
        assertEquals("/api/recordings/cam1/video", u.encodedPath)
        assertEquals("2026-11-24T09:00:00+09:00", u.queryParameter("start"))
        assertEquals("60", u.queryParameter("duration"))
        assertEquals(mapOf("X-API-Key" to "secret"), api.authHeaders())
    }

    @Test fun clipUrlChangesWhenTheClipWindowChanges() {
        // 같은 사건·카메라라도 경보가 올라가 구간이 바뀌면 다른 주소 → 화면이 새 클립을 다시 받는다
        val a = api.clipUrl("/api/cases/1/clips/cam1.mp4", "2026-11-24T00:19:06Z")
        val b = api.clipUrl("/api/cases/1/clips/cam1.mp4", "2026-11-24T00:21:40Z")
        assertTrue(a != b)
        assertEquals("/api/cases/1/clips/cam1.mp4", a.toHttpUrl().encodedPath)
        assertEquals("2026-11-24T00:19:06Z", a.toHttpUrl().queryParameter("window"))
    }

    @Test fun healthParsesDegraded() = runTest {
        server.enqueue(MockResponse().setBody("""{"status": "degraded", "disk_free_gb": 60.4, "detail": "x"}"""))
        assertEquals(Health("degraded", 60.4, "x"), api.health())
        assertEquals("/healthz", server.takeRequest().requestUrl!!.encodedPath)
    }

    @Test fun videoStateMapsServerAnswers() = runTest {
        val url = api.absoluteUrl("/api/cases/1/clips/cam1.mp4")
        server.enqueue(MockResponse().setResponseCode(206).setHeader("X-Video-Start", "2026-11-24T00:19:11Z").setBody("x"))
        assertEquals(VideoState.Ready("2026-11-24T00:19:11Z"), api.videoState(url))
        val probe = server.takeRequest()
        assertEquals("bytes=0-0", probe.getHeader("Range"))              // 1바이트만 받아 본다
        assertEquals("secret", probe.getHeader("X-API-Key"))

        server.enqueue(MockResponse().setResponseCode(202).setHeader("Retry-After", "9").setBody("{}"))
        assertEquals(VideoState.NotReady(9), api.videoState(url))
        server.enqueue(MockResponse().setResponseCode(202).setBody("""{"retry_after": 4}"""))
        assertEquals(VideoState.NotReady(4), api.videoState(url))
        server.enqueue(MockResponse().setResponseCode(404))
        assertEquals(VideoState.NoRecording, api.videoState(url))
        server.enqueue(MockResponse().setResponseCode(503))
        assertTrue(api.videoState(url) is VideoState.Unavailable)
        server.enqueue(MockResponse().setResponseCode(401))
        assertEquals(VideoState.Unavailable("API 키가 맞지 않습니다"), api.videoState(url))
    }
}
