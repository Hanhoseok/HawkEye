package com.hawkeye.admin.domain

import com.hawkeye.admin.data.ApiException
import com.hawkeye.admin.data.LibraryScene
import com.hawkeye.admin.data.LibraryVerdict
import com.hawkeye.admin.data.LibraryVideo
import java.io.IOException
import kotlin.math.roundToLong

/** 보관함 이름 규칙 (서버 library.clean_name 과 같음): 앞뒤 공백 제외 1~40자, 줄바꿈·제어 문자 없음. */
object LibraryNames {
    const val MAX = 40

    /** 문제가 없으면 null, 있으면 화면에 보일 문장. */
    fun problem(name: String): String? {
        val n = name.trim()
        return when {
            n.isEmpty() -> "이름을 입력해 주세요"
            n.length > MAX -> "이름은 ${MAX}자까지 쓸 수 있어요"
            n.any { it.code < 0x20 || it.code == 0x7f } -> "줄바꿈 없이 써 주세요"
            else -> null
        }
    }
}

/** 보관함 API 오류 → 화면 문장. */
object LibraryErrors {
    fun message(t: Throwable): String = when {
        t is ApiException && t.reason == "duplicate_name" -> "같은 이름의 폴더가 있어요"
        t is ApiException && t.reason == "invalid_name" -> "이름은 1~${LibraryNames.MAX}자, 줄바꿈 없이 써 주세요"
        t is ApiException && t.reason == "not_empty" -> "폴더에 장면이 있어요"
        t is ApiException && t.reason == "library_unavailable" -> "서버에 보관함이 꺼져 있어요 (HAWKEYE_LIBRARY)"
        t is ApiException && t.code == 404 -> "이미 지워졌어요. 목록을 새로 받을게요"
        t is ApiException && t.code == 401 -> "API 키가 맞지 않아요"
        t is ApiException -> "잠시 뒤 다시 해 주세요 (HTTP ${t.code})"
        t is IOException -> "서버에 연결하지 못했어요"
        else -> t.message ?: "알 수 없는 오류예요"
    }

    /** 다시 받아야 하는 오류인가 (다른 폰이 이미 지운 것). */
    fun isGone(t: Throwable): Boolean = t is ApiException && t.code == 404
}

/** 여러 장면 고르기. 고른 순서를 지킨다. */
object Selection {
    fun toggle(selected: Set<Int>, id: Int): Set<Int> = if (id in selected) selected - id else selected + id

    /** 다 골랐으면 모두 풀고, 아니면 모두 고른다. */
    fun all(selected: Set<Int>, ids: List<Int>): Set<Int> = if (ids.isNotEmpty() && selected.containsAll(ids)) emptySet() else ids.toSet()

    /** 목록이 바뀌면 없어진 장면은 빼낸다. */
    fun keepExisting(selected: Set<Int>, ids: List<Int>): Set<Int> = selected.intersect(ids.toSet())
}

/** 폴더 안 이전·다음 장면. */
object SceneNav {
    fun neighbors(scenes: List<LibraryScene>, id: Int): Pair<LibraryScene?, LibraryScene?> {
        val i = scenes.indexOfFirst { it.id == id }
        if (i < 0) return null to null
        return scenes.getOrNull(i - 1) to scenes.getOrNull(i + 1)
    }
}

/**
 * 카메라 바꾸기: 같은 순간을 이어 본다.
 * 공통 시각 = 재생 위치 + offset_ms 이므로 새 위치 = 지금 위치 + 이전 offset − 새 offset.
 * 새 영상에 그 순간이 없으면 처음·끝으로 자르고 그 사실을 알려 준다.
 */
object CameraSwitch {
    enum class Clip { NONE, BEFORE_START, AFTER_END }

    data class Result(val positionMs: Long, val clip: Clip)

    fun position(posMs: Long, fromOffsetMs: Double, toOffsetMs: Double, toDurationMs: Long?): Result {
        val target = posMs + (fromOffsetMs - toOffsetMs).roundToLong()
        return when {
            target < 0 -> Result(0, Clip.BEFORE_START)
            toDurationMs != null && toDurationMs > 0 && target >= toDurationMs ->
                Result((toDurationMs - 1000).coerceAtLeast(0), Clip.AFTER_END)
            else -> Result(target, Clip.NONE)
        }
    }
}

/** AI 판정 시간표: 누르면 그 판정 몇 초 전부터 지금 카메라로. */
object VerdictNav {
    const val LEAD_MS = 3000L

    /**
     * 판정 시각(공통 = 전체 카메라 시각) → 지금 카메라에서 시작할 위치.
     * 공통 시각 = 재생 위치 + offset 이므로 위치 = ms − offset − LEAD. 0 ~ (길이 − 1초) 로 자른다.
     */
    fun seekMs(verdictMs: Double, cameraOffsetMs: Double, durationMs: Long?): Long {
        val target = (verdictMs - cameraOffsetMs).toLong() - LEAD_MS
        val end = durationMs?.takeIf { it > 0 }?.let { (it - 1000).coerceAtLeast(0) }
        return target.coerceAtLeast(0).let { if (end != null) it.coerceAtMost(end) else it }
    }

    /** 지금 보는 순간(공통 시각)까지 나온 판정인가. */
    fun passed(verdictMs: Double, positionMs: Long, cameraOffsetMs: Double): Boolean =
        verdictMs <= positionMs + cameraOffsetMs + 1e-6
}

object LibraryFormat {
    /** 26.4 → "0:26" */
    fun clipLength(sec: Double?): String {
        if (sec == null || sec < 0) return "-"
        val s = sec.toLong()
        return "%d:%02d".format(s / 60, s % 60)
    }

    /** 폴더 합계: 59초 → "59초", 970초 → "16분", 4000초 → "1시간 6분" */
    fun total(sec: Double): String {
        val s = sec.toLong()
        val h = s / 3600
        val m = (s % 3600) / 60
        return when {
            h > 0 -> "${h}시간 ${m}분"
            m > 0 -> "${m}분"
            else -> "${s}초"
        }
    }

    /** 이유 글에서 '[집음 3개]' 같은 보조 표시는 빼고 공백을 정리한다. */
    fun verdictReason(reason: String): String =
        reason.replace(Regex("\\s*\\[[^\\]]*\\]"), "").replace(Regex("\\s+"), " ").trim()

    /** "가져감 3 · 결제 1 · 미결제 2" — 셋 다 0 이면 빈 글자. */
    fun verdictCounts(v: LibraryVerdict): String =
        if (v.taken == 0 && v.paid == 0 && v.unpaid == 0) "" else "가져감 ${v.taken} · 결제 ${v.paid} · 미결제 ${v.unpaid}"

    /** 26666.0 → "0:26" (판정 시각, 전체 카메라 기준). */
    fun verdictTime(ms: Double): String = clipLength(ms / 1000.0)

    /** "전체·선반" — 카메라 1대면 빈 글자 (고를 것이 없음). */
    fun cameras(videos: List<LibraryVideo>): String =
        if (videos.size < 2) "" else videos.joinToString("·") { shortCamera(it.cameraId) }

    fun shortCamera(cameraId: String): String = when (cameraId) {
        "cam1" -> "전체"
        "cam2" -> "선반"
        else -> cameraId
    }

    /** 장면 줄 아래 글: "0:26 · 전체·선반 · 2차 촬영" (회차는 지금 폴더와 다를 때만). */
    fun sceneLine(scene: LibraryScene, folderName: String): String =
        listOfNotNull(
            clipLength(scene.duration),
            cameras(scene.videos).ifEmpty { null },
            "AI".takeIf { scene.videos.any { it.hasAi } },
            scene.origin.takeIf { it.isNotBlank() && it != folderName },
        ).joinToString(" · ")
}
