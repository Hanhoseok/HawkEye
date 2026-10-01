package com.hawkeye.admin.data

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

/** 경보 서버 응답 형식. 서버 쪽 정의: experiments/app-pos/server/hawkeye_server (docs/design.md §5). */

@Serializable
data class Item(val name: String, val taken: Int, val paid: Int)

@Serializable
data class CaseSummary(
    val id: Int,
    @SerialName("run_id") val runId: String,
    @SerialName("camera_id") val cameraId: String,
    @SerialName("person_id") val personId: Int,
    /** CLEAR / WARNING / REVIEW / HIGH_RISK */
    val level: String,
    /** active / auto_cleared / pass / resolved */
    val state: String,
    val taken: Int = 0,
    val paid: Int = 0,
    val unpaid: Int = 0,
    val reason: String? = null,
    @SerialName("identity_check") val identityCheck: Double? = null,
    /** 품목별 개수. 선반 지도 3단계 이후 채워진다. */
    val items: List<Item>? = null,
    /** paid_confirmed / false_alarm */
    val resolution: String? = null,
    @SerialName("resolution_note") val resolutionNote: String? = null,
    @SerialName("has_snapshot") val hasSnapshot: Boolean = false,
    @SerialName("updated_at") val updatedAt: String,
    @SerialName("created_at") val createdAt: String = "",
    @SerialName("last_event_id") val lastEventId: Int? = null,
)

@Serializable
data class EventRecord(
    val id: Int,
    val level: String,
    @SerialName("time_sec") val timeSec: Double? = null,
    val frame: Int? = null,
    val timestamp: String? = null,
    val taken: Int = 0,
    val paid: Int = 0,
    val unpaid: Int = 0,
    val reason: String? = null,
    @SerialName("identity_check") val identityCheck: Double? = null,
    @SerialName("has_snapshot") val hasSnapshot: Boolean = false,
    @SerialName("received_at") val receivedAt: String = "",
)

@Serializable
data class ResolutionRecord(val resolution: String, val note: String? = null, val at: String = "")

data class CaseDetail(
    val case: CaseSummary,
    val events: List<EventRecord>,
    val resolutions: List<ResolutionRecord>,
)

@Serializable
data class Stream(
    @SerialName("camera_id") val cameraId: String,
    /** RTSP 주소 (앱 라이브). 브라우저 전용 카메라면 비어 있다. */
    val url: String = "",
    /** WebRTC 주소 (PC 대시보드). 앱은 쓰지 않는다. */
    @SerialName("web_url") val webUrl: String? = null,
)

@Serializable
data class AppConfig(val streams: List<Stream> = emptyList()) {
    /** 앱이 재생할 수 있는(RTSP 주소가 있는) 카메라. */
    fun rtspStreams(): List<Stream> = streams.filter { it.url.isNotBlank() }
}

enum class Resolution(val wire: String) {
    PAID_CONFIRMED("paid_confirmed"),
    FALSE_ALARM("false_alarm"),
}

sealed interface ServerMessage {
    data class CaseUpdated(val notify: Boolean, val case: CaseSummary) : ServerMessage
    data object Unknown : ServerMessage
}

data class ServerSettings(val baseUrl: String, val apiKey: String?)

class ApiException(val code: Int, message: String) : Exception(message)
