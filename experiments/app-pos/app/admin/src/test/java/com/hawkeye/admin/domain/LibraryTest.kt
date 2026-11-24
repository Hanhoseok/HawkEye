package com.hawkeye.admin.domain

import com.hawkeye.admin.data.ApiException
import com.hawkeye.admin.data.LibraryScene
import com.hawkeye.admin.data.LibraryVerdict
import com.hawkeye.admin.data.LibraryVideo
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.IOException

class LibraryTest {
    @Test fun namesFollowServerRule() {
        assertNull(LibraryNames.problem("  시연용 "))
        assertEquals("이름을 입력해 주세요", LibraryNames.problem("   "))
        assertNull(LibraryNames.problem("a".repeat(40)))
        assertEquals("이름은 40자까지 쓸 수 있어요", LibraryNames.problem("a".repeat(41)))
        assertEquals("줄바꿈 없이 써 주세요", LibraryNames.problem("줄\n바꿈"))
    }

    @Test fun errorsBecomeSentences() {
        assertEquals("같은 이름의 폴더가 있어요", LibraryErrors.message(ApiException(409, "x", "duplicate_name")))
        assertEquals("폴더에 장면이 있어요", LibraryErrors.message(ApiException(409, "x", "not_empty")))
        assertEquals("이미 지워졌어요. 목록을 새로 받을게요", LibraryErrors.message(ApiException(404, "x", "not_found")))
        assertEquals("서버에 보관함이 꺼져 있어요 (HAWKEYE_LIBRARY)", LibraryErrors.message(ApiException(503, "x", "library_unavailable")))
        assertEquals("API 키가 맞지 않아요", LibraryErrors.message(ApiException(401, "x")))
        assertEquals("서버에 연결하지 못했어요", LibraryErrors.message(IOException("timeout")))
        assertTrue(LibraryErrors.isGone(ApiException(404, "x")))
        assertFalse(LibraryErrors.isGone(ApiException(409, "x")))
    }

    @Test fun selection() {
        assertEquals(setOf(1), Selection.toggle(emptySet(), 1))
        assertEquals(emptySet<Int>(), Selection.toggle(setOf(1), 1))
        assertEquals(setOf(1, 2, 3), Selection.all(setOf(2), listOf(1, 2, 3)))
        assertEquals(emptySet<Int>(), Selection.all(setOf(1, 2, 3), listOf(1, 2, 3)))
        assertEquals(setOf(2), Selection.keepExisting(setOf(2, 9), listOf(1, 2)))
    }

    private fun scene(id: Int, name: String, origin: String = "2차 촬영", vararg cams: String) =
        LibraryScene(id = id, name = name, origin = origin, duration = 26.4, videos = cams.mapIndexed { i, c -> LibraryVideo(i, c) })

    @Test fun neighbors() {
        val list = listOf(scene(1, "C01"), scene(2, "C02"), scene(3, "C03"))
        assertEquals(null to list[1], SceneNav.neighbors(list, 1))
        assertEquals(list[0] to list[2], SceneNav.neighbors(list, 2))
        assertEquals(list[1] to null, SceneNav.neighbors(list, 3))
        assertEquals(null to null, SceneNav.neighbors(list, 99))
    }

    @Test fun cameraSwitchKeepsTheSameMoment() {
        // 2차 C20: 선반 카메라 offset −2863.9ms (공통 시각 = 재생 위치 + offset)
        val toShelf = CameraSwitch.position(10_000, 0.0, -2863.9, 30_000)
        assertEquals(CameraSwitch.Result(12_864, CameraSwitch.Clip.NONE), toShelf)
        val back = CameraSwitch.position(12_864, -2863.9, 0.0, 26_000)
        assertEquals(10_000, back.positionMs)
        // 선반 1.0초 = 공통 −1.86초 → 전체 카메라에는 없는 순간 → 처음부터
        assertEquals(CameraSwitch.Result(0, CameraSwitch.Clip.BEFORE_START), CameraSwitch.position(1_000, -2863.9, 0.0, 26_000))
        // 끝을 넘으면 끝 1초 전
        assertEquals(CameraSwitch.Result(25_000, CameraSwitch.Clip.AFTER_END), CameraSwitch.position(25_000, 0.0, -2863.9, 26_000))
        // 시차 없음 · 길이 모름
        assertEquals(CameraSwitch.Result(5_000, CameraSwitch.Clip.NONE), CameraSwitch.position(5_000, 0.0, 0.0, null))
    }

    @Test fun formats() {
        assertEquals("0:26", LibraryFormat.clipLength(26.9))
        assertEquals("1:05", LibraryFormat.clipLength(65.0))
        assertEquals("-", LibraryFormat.clipLength(null))
        assertEquals("59초", LibraryFormat.total(59.0))
        assertEquals("16분", LibraryFormat.total(970.0))
        assertEquals("1시간 6분", LibraryFormat.total(4000.0))
        assertEquals("0:26 · 전체·선반", LibraryFormat.sceneLine(scene(1, "C01", "2차 촬영", "cam1", "cam2"), "2차 촬영"))
        assertEquals("0:26 · 전체·선반 · 2차 촬영", LibraryFormat.sceneLine(scene(1, "C01", "2차 촬영", "cam1", "cam2"), "시연용"))
        assertEquals("0:26", LibraryFormat.sceneLine(scene(1, "C01", "1차 촬영", "cam1"), "1차 촬영"))
        val withAi = scene(1, "C01", "2차 촬영", "cam1", "cam2").let { s -> s.copy(videos = s.videos.map { it.copy(hasAi = it.cameraId == "cam1") }) }
        assertEquals("0:26 · 전체·선반 · AI", LibraryFormat.sceneLine(withAi, "2차 촬영"))
    }

    @Test fun verdictTimeline() {
        // 4차 C05: 26.67초 위험. 전체 카메라(offset 0) → 23.67초부터, 선반(offset −2863.9) → 26.53초부터
        assertEquals(23_666, VerdictNav.seekMs(26666.0, 0.0, 26_700))
        assertEquals(26_529, VerdictNav.seekMs(26666.0, -2863.9, 40_000))
        assertEquals(25_700, VerdictNav.seekMs(29_000.0, 0.0, 26_700))         // 영상 끝을 넘으면 끝 1초 전
        assertEquals(0, VerdictNav.seekMs(1_000.0, 0.0, 26_700))
        assertTrue(VerdictNav.passed(17066.0, 17_100, 0.0))
        assertFalse(VerdictNav.passed(17066.0, 14_000, 0.0))
        assertTrue(VerdictNav.passed(17066.0, 20_000, -2863.9))                 // 선반 20초 = 공통 17.1초
        val v = LibraryVerdict(26666.0, 1, "HIGH_RISK", "door", "결제가 물건보다 적음 (물건 3 / 결제 1) [집음 3개] — 출구 쪽에서 사라짐", 3, 1, 2)
        assertEquals("0:26", LibraryFormat.verdictTime(v.ms))
        assertEquals("결제가 물건보다 적음 (물건 3 / 결제 1) — 출구 쪽에서 사라짐", LibraryFormat.verdictReason(v.reason))
        assertEquals("가져감 3 · 결제 1 · 미결제 2", LibraryFormat.verdictCounts(v))
        assertEquals("", LibraryFormat.verdictCounts(v.copy(taken = 0, paid = 0, unpaid = 0)))
    }
}
