package com.hawkeye.admin.data

import com.hawkeye.admin.Fixtures
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class ProtocolTest {
    @Test fun parsesWebSocketCaseUpdate() {
        val m = Protocol.parseMessage(Fixtures.read("ws_case_updated.json"))
        assertTrue(m is ServerMessage.CaseUpdated)
        m as ServerMessage.CaseUpdated
        assertEquals(true, m.notify)
        assertEquals(5, m.case.personId)
        assertEquals("WARNING", m.case.level)
        assertEquals("active", m.case.state)
        assertTrue(m.case.hasSnapshot)
    }

    @Test fun parsesCaseListIncludingNullFields() {
        val cases = Protocol.parseCases(Fixtures.read("cases.json"))
        assertEquals(setOf(5, 6, 2), cases.map { it.personId }.toSet())
        val pass = cases.first { it.personId == 6 }
        assertEquals("pass", pass.state)
        assertNull(pass.identityCheck)
        assertNull(pass.items)
    }

    @Test fun parsesCaseDetailWithItemsEventsAndResolutions() {
        val d = Protocol.parseCaseDetail(Fixtures.read("case_detail.json"))
        assertEquals("HIGH_RISK", d.case.level)
        assertEquals("false_alarm", d.case.resolution)
        assertEquals(listOf(Item("과자", 2, 1)), d.case.items)
        assertEquals(listOf("WARNING", "HIGH_RISK"), d.events.map { it.level })
        assertEquals(listOf("false_alarm"), d.resolutions.map { it.resolution })
        assertEquals("지갑 꺼냄", d.resolutions.single().note)
    }

    @Test fun parsesConfigStreams() {
        val c = Protocol.parseConfig(Fixtures.read("config.json"))
        assertEquals(listOf(Stream("cam1", "rtsp://10.0.2.2:8554/cam1")), c.streams)
    }

    @Test fun configToleratesBrowserOnlyCameraAndKeepsWebUrl() {
        val c = Protocol.parseConfig(
            """{"streams":[{"camera_id":"cam1","url":"rtsp://h:9554/cam1","web_url":"http://127.0.0.1:8889/cam1"},""" +
                """{"camera_id":"cam2","web_url":"http://127.0.0.1:8889/cam2"}]}""",
        )
        assertEquals("http://127.0.0.1:8889/cam1", c.streams[0].webUrl)
        assertEquals("", c.streams[1].url)
        assertEquals(listOf("cam1"), c.rtspStreams().map { it.cameraId })
    }

    @Test fun unknownMessageTypeIsIgnored() {
        assertEquals(ServerMessage.Unknown, Protocol.parseMessage("""{"type":"hello","x":1}"""))
    }

    @Test fun parsesRecordingConfigWithAiUrlAndLiveStatus() {
        val c = Protocol.parseConfig(Fixtures.read("config_recordings.json"))
        assertTrue(c.recordings)
        val (cam1, cam2) = c.streams
        assertEquals("rtsp://viewer:pw@100.64.0.1:8554/cam1_ai", cam1.aiUrl)
        assertEquals(LiveStatus(true, "2026-11-24T00:19:11.138344Z"), cam1.live)
        assertNull(cam2.aiUrl)
        assertEquals(LiveStatus(false, null), cam2.live)
    }

    @Test fun oldConfigHasRecordingsOff() {
        val c = Protocol.parseConfig(Fixtures.read("config.json"))
        assertFalse(c.recordings)
        assertNull(c.streams.single().aiUrl)
    }

    @Test fun parsesCaseClipsIncludingUnavailableCamera() {
        val c = Protocol.parseCaseClips(Fixtures.read("case_clips.json"))
        assertEquals("event", c.anchor)
        assertEquals(0.42, c.delaySec!!, 1e-9)
        val (cam1, cam2) = c.clips
        assertTrue(cam1.isCaseCamera)
        assertEquals("recorded", cam1.status)
        assertEquals(true, cam1.partial)
        assertEquals("2026-11-24T00:19:11.138344Z", cam1.videoStart)
        assertEquals("unavailable", cam2.status)
        assertNull(cam2.hasRecording)
    }

    @Test fun parsesRecordingsAndRetryAfterBody() {
        val r = Protocol.parseRecordings(Fixtures.read("recordings.json"))
        assertEquals(28.033322222, r.spans.single().duration, 1e-9)
        assertEquals(9, Protocol.parseRetryAfter("""{"detail": "not_ready", "retry_after": 9}"""))
        assertNull(Protocol.parseRetryAfter("not json"))
    }
}
