package com.hawkeye.admin.data

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.booleanOrNull
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive

/** 서버 JSON ↔ 앱 모델. 모르는 필드는 무시한다 (서버가 필드를 늘려도 앱이 깨지지 않게). */
object Protocol {
    val json = Json { ignoreUnknownKeys = true }

    fun parseMessage(text: String): ServerMessage {
        val obj = json.parseToJsonElement(text).jsonObject
        return when (obj["type"]?.jsonPrimitive?.contentOrNull) {
            "case.updated" -> ServerMessage.CaseUpdated(
                notify = obj["notify"]?.jsonPrimitive?.booleanOrNull ?: false,
                case = json.decodeFromJsonElement(CaseSummary.serializer(), obj.getValue("case")),
            )
            else -> ServerMessage.Unknown
        }
    }

    fun parseCases(text: String): List<CaseSummary> =
        json.parseToJsonElement(text).jsonArray.map { json.decodeFromJsonElement(CaseSummary.serializer(), it) }

    fun parseCase(text: String): CaseSummary = json.decodeFromString(CaseSummary.serializer(), text)

    fun parseCaseDetail(text: String): CaseDetail {
        val obj: JsonObject = json.parseToJsonElement(text).jsonObject
        return CaseDetail(
            case = json.decodeFromJsonElement(CaseSummary.serializer(), obj),
            events = (obj["events"] as? JsonArray).orEmpty()
                .map { json.decodeFromJsonElement(EventRecord.serializer(), it) },
            resolutions = (obj["resolutions"] as? JsonArray).orEmpty()
                .map { json.decodeFromJsonElement(ResolutionRecord.serializer(), it) },
        )
    }

    fun parseConfig(text: String): AppConfig = json.decodeFromString(AppConfig.serializer(), text)

    fun parseHealth(text: String): Health = json.decodeFromString(Health.serializer(), text)

    fun parseCaseClips(text: String): CaseClips = json.decodeFromString(CaseClips.serializer(), text)

    fun parseRecordings(text: String): RecordingList = json.decodeFromString(RecordingList.serializer(), text)

    /** 202 응답 본문의 retry_after (헤더가 없을 때 대신). */
    fun parseRetryAfter(text: String): Int? = runCatching {
        json.parseToJsonElement(text).jsonObject["retry_after"]?.jsonPrimitive?.contentOrNull?.toIntOrNull()
    }.getOrNull()
}
