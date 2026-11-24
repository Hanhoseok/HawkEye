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

    // --- 보관함 ---

    @Test fun libraryListAndFolderParse() = runTest {
        server.enqueue(MockResponse().setBody(Fixtures.read("library.json")))
        server.enqueue(MockResponse().setBody(Fixtures.read("library_folder.json")))
        val folders = api.libraryFolders()
        assertEquals(listOf("2차 촬영", "시연용"), folders.map { it.name })
        assertEquals(3, folders[0].sceneCount)
        assertEquals("/api/library", server.takeRequest().requestUrl!!.encodedPath)

        val d = api.libraryFolder(1)
        assertEquals("/api/library/folders/1", server.takeRequest().requestUrl!!.encodedPath)
        assertEquals(listOf("C01", "C20", "N08-1"), d.scenes.map { it.name })
        val c20 = d.scenes[1]
        assertEquals(-2863.9, c20.videos.first { it.cameraId == "cam2" }.offsetMs, 1e-6)
        assertTrue(d.scenes[0].hasThumb && d.scenes[0].mediaAt != null)
        assertEquals(1, d.scenes[2].videos.size)
    }

    @Test fun libraryWritesUseRightMethodsAndBodies() = runTest {
        server.enqueue(MockResponse().setResponseCode(201).setBody("""{"id": 5, "name": "시연용", "scene_count": 0, "total_sec": 0.0}"""))
        assertEquals(5, api.createFolder("시연용").id)
        server.takeRequest().let { assertEquals("POST", it.method); assertEquals("/api/library/folders", it.path); assertEquals("""{"name":"시연용"}""", it.body.readUtf8()) }

        server.enqueue(MockResponse().setBody("""{"id": 5, "name": "발표", "scene_count": 0, "total_sec": 0.0}"""))
        api.renameFolder(5, "발표")
        server.takeRequest().let { assertEquals("PATCH", it.method); assertEquals("/api/library/folders/5", it.path) }

        server.enqueue(MockResponse().setBody("""{"deleted_scenes": 3}"""))
        assertEquals(3, api.deleteFolder(5, withScenes = true))
        server.takeRequest().let { assertEquals("DELETE", it.method); assertEquals("/api/library/folders/5?with_scenes=true", it.path) }

        server.enqueue(MockResponse().setBody("""{"moved": 2}"""))
        assertEquals(2, api.moveScenes(listOf(1, 2), 7))
        server.takeRequest().let { assertEquals("/api/library/scenes/move", it.path); assertEquals("""{"scene_ids":[1,2],"folder_id":7}""", it.body.readUtf8()) }

        server.enqueue(MockResponse().setBody("""{"deleted": 1}"""))
        assertEquals(1, api.deleteScenes(listOf(4)))
        server.takeRequest().let { assertEquals("/api/library/scenes/delete", it.path); assertEquals("""{"scene_ids":[4]}""", it.body.readUtf8()) }

        server.enqueue(MockResponse().setBody("""{"id": 4, "name": "도난", "origin": "2차 촬영", "videos": []}"""))
        assertEquals("도난", api.renameScene(4, "도난").name)
        server.takeRequest().let { assertEquals("PATCH", it.method); assertEquals("/api/library/scenes/4", it.path) }
    }

    @Test fun libraryErrorsCarryServerReason() = runTest {
        server.enqueue(MockResponse().setResponseCode(409).setBody(Fixtures.read("library_error_duplicate.json")))
        val e = runCatching { api.createFolder("2차 촬영") }.exceptionOrNull() as ApiException
        assertEquals(409, e.code)
        assertEquals("duplicate_name", e.reason)
        server.enqueue(MockResponse().setResponseCode(500).setBody("oops"))
        assertNull((runCatching { api.libraryFolders() }.exceptionOrNull() as ApiException).reason)
    }

    @Test fun libraryVideoUrlAndThumb() = runTest {
        assertTrue(api.libraryVideoUrl(12).endsWith("/api/library/videos/12.mp4"))
        assertTrue(api.libraryAiVideoUrl(12).endsWith("/api/library/videos/12/ai.mp4"))
        server.enqueue(MockResponse().setBody(Buffer().write(byteArrayOf(1, 2))))
        server.enqueue(MockResponse().setResponseCode(404))
        assertArrayEquals(byteArrayOf(1, 2), api.sceneThumb(3))
        assertEquals("/api/library/scenes/3/thumb.jpg", server.takeRequest().requestUrl!!.encodedPath)
        assertNull(api.sceneThumb(4))
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
