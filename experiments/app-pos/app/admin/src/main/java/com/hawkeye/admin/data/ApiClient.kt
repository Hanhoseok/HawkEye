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

    suspend fun listCases(state: String? = null, updatedSince: String? = null): List<CaseSummary> =
        Protocol.parseCases(get(url("/api/cases") {
            state?.let { addQueryParameter("state", it) }
            updatedSince?.let { addQueryParameter("updated_since", it) }
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
