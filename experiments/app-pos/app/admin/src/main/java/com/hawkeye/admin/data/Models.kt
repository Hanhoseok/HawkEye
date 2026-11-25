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

/** 카메라가 지금 송출 중인가 (서버가 녹화 구간으로 판단). since = 이어서 송출한 시작 시각. */
@Serializable
data class LiveStatus(val streaming: Boolean = false, val since: String? = null)

@Serializable
data class Stream(
    @SerialName("camera_id") val cameraId: String,
    /** RTSP 주소 (앱 라이브). 브라우저 전용 카메라면 비어 있다. 보기 계정이 들어 있을 수 있어 화면에 그대로 보이면 안 된다. */
    val url: String = "",
    /** WebRTC 주소 (PC 대시보드). 앱은 쓰지 않는다. */
    @SerialName("web_url") val webUrl: String? = null,
    /** AI 화면(파이프라인이 사람 상자·번호를 그린 영상) RTSP 주소. 없으면 체크박스를 숨긴다. */
    @SerialName("ai_url") val aiUrl: String? = null,
    /** 녹화 기능이 켜진 서버만 준다. */
    val live: LiveStatus? = null,
)

@Serializable
data class AppConfig(
    val streams: List<Stream> = emptyList(),
    /** 서버가 녹화 재생(사건 클립·지난 영상)을 지원하는가. */
    val recordings: Boolean = false,
) {
    /** 앱이 재생할 수 있는(RTSP 주소가 있는) 카메라. */
    fun rtspStreams(): List<Stream> = streams.filter { it.url.isNotBlank() }
}

/** 사건 클립 하나 (카메라별). status: cached / recorded / no_recording / unavailable */
@Serializable
data class ClipInfo(
    @SerialName("camera_id") val cameraId: String,
    /** 서버 기준 경로 (예: /api/cases/1/clips/cam1.mp4) */
    val url: String,
    @SerialName("is_case_camera") val isCaseCamera: Boolean = false,
    val status: String = "recorded",
    @SerialName("retry_after") val retryAfter: Int? = null,
    @SerialName("covered_sec") val coveredSec: Double? = null,
    val partial: Boolean? = null,
    @SerialName("has_recording") val hasRecording: Boolean? = null,
    @SerialName("video_start") val videoStart: String? = null,
)

/** 사건 클립 정보. anchor = event(경보 장면 시각) / received(서버가 받은 시각 — 정확하지 않을 수 있음) */
@Serializable
data class CaseClips(
    @SerialName("case_id") val caseId: Int,
    val anchor: String,
    @SerialName("anchor_time") val anchorTime: String,
    val start: String,
    val end: String,
    @SerialName("delay_sec") val delaySec: Double? = null,
    val clips: List<ClipInfo> = emptyList(),
)

@Serializable
data class RecordingSpan(val start: String, val duration: Double)

@Serializable
data class RecordingList(
    @SerialName("camera_id") val cameraId: String,
    val spans: List<RecordingSpan> = emptyList(),
)

/** 서버 상태 (/healthz). status = ok / degraded(디스크 여유 부족). */
@Serializable
data class Health(
    val status: String = "ok",
    @SerialName("disk_free_gb") val diskFreeGb: Double? = null,
    val detail: String? = null,
)

/** 녹화 영상(MP4)을 지금 받을 수 있는가. */
sealed interface VideoState {
    /** videoStart = 영상이 시작되는 실제 시각 (서버 X-Video-Start) */
    data class Ready(val videoStart: String? = null) : VideoState
    /** 아직 녹화 중인 구간 — retryAfterSec 뒤에 다시 */
    data class NotReady(val retryAfterSec: Int) : VideoState
    data object NoRecording : VideoState
    data class Unavailable(val reason: String) : VideoState
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

/** reason = 서버 detail 앞의 코드 (예: duplicate_name, not_empty). 앱이 한국어 문구를 고른다. */
class ApiException(val code: Int, message: String, val reason: String? = null) : Exception(message)

// --- 보관함 (서버 docs/design.md §11) ---

@Serializable
data class LibraryFolder(
    val id: Int,
    val name: String,
    @SerialName("scene_count") val sceneCount: Int = 0,
    @SerialName("total_sec") val totalSec: Double = 0.0,
)

@Serializable
data class LibraryFolders(val folders: List<LibraryFolder> = emptyList())

/** offsetMs: 카메라 시차. 공통 시각 = 재생 위치 + offsetMs (전체 카메라 0). */
@Serializable
data class LibraryVideo(
    val id: Int,
    @SerialName("camera_id") val cameraId: String,
    val duration: Double? = null,
    val width: Int? = null,
    val height: Int? = null,
    @SerialName("offset_ms") val offsetMs: Double = 0.0,
    /** AI 화면(사람 상자·번호를 그린 같은 시간축 영상)이 있는가. */
    @SerialName("has_ai") val hasAi: Boolean = false,
)

/** AI 판정 하나 (추적 도구가 지금 설정으로 다시 낸 것). ms = 전체 카메라(cam1) 재생 시각. */
@Serializable
data class LibraryVerdict(
    val ms: Double,
    val person: Int,
    /** CLEAR / WARNING / REVIEW / HIGH_RISK */
    val level: String,
    val zone: String = "",
    val reason: String = "",
    val taken: Int = 0,
    val paid: Int = 0,
    val unpaid: Int = 0,
)

/** origin = 가져올 때의 회차 폴더 이름 (다른 폴더로 옮겨도 남는다). mediaAt = 영상을 넣거나 바꾼 시각 (썸네일 캐시 키). */
@Serializable
data class LibraryScene(
    val id: Int,
    @SerialName("folder_id") val folderId: Int = 0,
    val name: String,
    val origin: String = "",
    val duration: Double? = null,
    @SerialName("media_at") val mediaAt: String? = null,
    val videos: List<LibraryVideo> = emptyList(),
    @SerialName("has_thumb") val hasThumb: Boolean = false,
    /** AI 판정 (시각 순). 없으면 빈 목록. */
    val verdicts: List<LibraryVerdict> = emptyList(),
    /** 판정을 어디서 가져왔나 (예: '추적 도구가 지금 설정으로 다시 낸 판정 · 2026-11-24 17:32'). */
    @SerialName("verdict_source") val verdictSource: String? = null,
)

@Serializable
data class FolderDetail(val folder: LibraryFolder, val scenes: List<LibraryScene> = emptyList())
