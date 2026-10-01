package com.hawkeye.admin.data

import com.hawkeye.admin.Fixtures
import org.junit.Assert.assertEquals
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

    @Test fun unknownMessageTypeIsIgnored() {
        assertEquals(ServerMessage.Unknown, Protocol.parseMessage("""{"type":"hello","x":1}"""))
    }
}
