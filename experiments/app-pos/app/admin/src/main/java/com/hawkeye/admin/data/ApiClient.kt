package com.hawkeye.admin.data

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put
import okhttp3.HttpUrl.Companion.toHttpUrl
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import java.util.concurrent.TimeUnit

/** 경보 서버 REST 호출. 모든 함수는 IO 스레드에서 실행된다. */
class ApiClient(
    private val settings: ServerSettings,
    val http: OkHttpClient = defaultClient(),
) {
    private val base = settings.baseUrl.trimEnd('/')

    suspend fun listCases(state: String? = null, updatedSince: String? = null, limit: Int? = null): List<CaseSummary> =
        Protocol.parseCases(get(url("/api/cases") {
            state?.let { addQueryParameter("state", it) }
            updatedSince?.let { addQueryParameter("updated_since", it) }
            limit?.let { addQueryParameter("limit", it.toString()) }
        }))

    suspend fun getCase(caseId: Int): CaseDetail = Protocol.parseCaseDetail(get(url("/api/cases/$caseId")))

    suspend fun resolve(caseId: Int, resolution: Resolution, note: String?): CaseSummary {
        val body = buildJsonObject {
            put("resolution", resolution.wire)
            if (!note.isNullOrBlank()) put("note", note)
        }.toString()
        val req = request(url("/api/cases/$caseId/resolution"))
            .post(body.toRequestBody(JSON)).build()
        return Protocol.parseCase(call(req))
    }

    /** 장면 사진(JPEG). 없으면 null. */
    suspend fun snapshot(eventId: Int): ByteArray? = withContext(Dispatchers.IO) {
        http.newCall(request(url("/api/events/$eventId/snapshot.jpg")).build()).execute().use { r ->
            when {
                r.code == 404 -> null
                !r.isSuccessful -> throw ApiException(r.code, "사진을 불러오지 못했습니다 (HTTP ${r.code})")
                else -> r.body!!.bytes()
            }
        }
    }

    suspend fun config(): AppConfig = Protocol.parseConfig(get(url("/api/config")))

    // --- 녹화 (서버 docs/design.md §10) ---

    suspend fun caseClips(caseId: Int): CaseClips = Protocol.parseCaseClips(get(url("/api/cases/$caseId/clips")))

    /** 녹화 구간 목록. start·end 는 오프셋이 있는 ISO8601 (없으면 서버 기본: 최근 24시간). */
    suspend fun recordings(cameraId: String, start: String? = null, end: String? = null): RecordingList =
        Protocol.parseRecordings(get(url("/api/recordings") {
            addQueryParameter("camera_id", cameraId)
            start?.let { addQueryParameter("start", it) }
            end?.let { addQueryParameter("end", it) }
        }))

    /** 서버 상태. 디스크 여유가 부족하면 status = degraded. */
    suspend fun health(): Health = Protocol.parseHealth(get(url("/healthz")))

    /** 서버 기준 경로(/api/...)를 전체 주소로. */
    fun absoluteUrl(path: String): String = url(path).toString()

    /**
     * 사건 클립 주소. 같은 사건·카메라라도 경보가 올라가면 클립 구간이 바뀌므로, 구간 시작을 주소에 넣어
     * 화면(상태·플레이어)이 새 클립으로 다시 준비되게 한다. 서버는 이 값을 쓰지 않는다.
     */
    fun clipUrl(path: String, windowStart: String): String =
        url(path) { addQueryParameter("window", windowStart) }.toString()

    /** 지난 영상 조각 MP4 주소. */
    fun recordingVideoUrl(cameraId: String, start: String, durationSec: Int): String =
        url("/api/recordings/$cameraId/video") {
            addQueryParameter("start", start)
            addQueryParameter("duration", durationSec.toString())
        }.toString()

    /** 영상 플레이어가 요청에 붙일 헤더 (API 키). */
    fun authHeaders(): Map<String, String> =
        settings.apiKey?.takeIf { it.isNotBlank() }?.let { mapOf("X-API-Key" to it) } ?: emptyMap()

    /**
     * MP4 를 지금 받을 수 있는지 1바이트만 요청해 본다 (서버는 이때 클립을 만들어 캐시에 둔다 → 플레이어는 캐시에서 받음).
     * 206/200 = 준비됨, 202 = 아직 녹화 중 (Retry-After), 404 = 녹화 없음, 그 밖 = 쓸 수 없음.
     */
    suspend fun videoState(absoluteUrl: String): VideoState = withContext(Dispatchers.IO) {
        val req = request(absoluteUrl.toHttpUrl()).header("Range", "bytes=0-0").build()
        runCatching {
            http.newCall(req).execute().use { r ->
                when (r.code) {
                    200, 206 -> VideoState.Ready(r.header("X-Video-Start"))
                    202 -> VideoState.NotReady(
                        r.header("Retry-After")?.toIntOrNull()
                            ?: Protocol.parseRetryAfter(r.body?.string().orEmpty()) ?: 2,
                    )
                    404 -> VideoState.NoRecording
                    401 -> VideoState.Unavailable("API 키가 맞지 않습니다")
                    else -> VideoState.Unavailable("녹화 서버 오류 (HTTP ${r.code})")
                }
            }
        }.getOrElse { VideoState.Unavailable(it.message ?: "연결하지 못했습니다") }
    }

    fun webSocketUrl(): String {
        val ws = base.replaceFirst("http", "ws") + "/ws"
        return if (settings.apiKey.isNullOrBlank()) ws else "$ws?key=${settings.apiKey}"
    }

    // --- 내부 ---
    private fun url(path: String, block: okhttp3.HttpUrl.Builder.() -> Unit = {}) =
        (base + path).toHttpUrl().newBuilder().apply(block).build()

    private fun request(url: okhttp3.HttpUrl): Request.Builder = Request.Builder().url(url).apply {
        settings.apiKey?.takeIf { it.isNotBlank() }?.let { header("X-API-Key", it) }
    }

    private suspend fun get(url: okhttp3.HttpUrl): String = call(request(url).build())

    private suspend fun call(req: Request): String = withContext(Dispatchers.IO) {
        http.newCall(req).execute().use { r ->
            if (!r.isSuccessful) {
                val reason = if (r.code == 401) "API 키가 맞지 않습니다" else "서버 오류"
                throw ApiException(r.code, "$reason (HTTP ${r.code})")
            }
            r.body!!.string()
        }
    }

    companion object {
        private val JSON = "application/json".toMediaType()

        fun defaultClient(): OkHttpClient = OkHttpClient.Builder()
            .connectTimeout(5, TimeUnit.SECONDS)
            .readTimeout(10, TimeUnit.SECONDS)
            .pingInterval(20, TimeUnit.SECONDS) // WebSocket 끊김 감지
            .build()
    }
}
