package com.hawkeye.admin.ui

import android.graphics.BitmapFactory
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.rounded.Undo
import androidx.compose.material.icons.rounded.Check
import androidx.compose.material.icons.rounded.History
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.unit.dp
import com.hawkeye.admin.AppGraph
import com.hawkeye.admin.data.CaseClips
import com.hawkeye.admin.data.CaseDetail
import com.hawkeye.admin.data.CaseSummary
import com.hawkeye.admin.data.Resolution
import com.hawkeye.admin.domain.CaseFilter
import com.hawkeye.admin.domain.RecordingNav
import com.hawkeye.admin.domain.StatusAction
import com.hawkeye.admin.domain.StatusSummaries
import com.hawkeye.admin.domain.StatusSummary
import com.hawkeye.admin.domain.Tone
import com.hawkeye.admin.ui.components.AppBar
import com.hawkeye.admin.ui.components.ButtonKind
import com.hawkeye.admin.ui.components.DoneChip
import com.hawkeye.admin.ui.components.Facts
import com.hawkeye.admin.ui.components.Group
import com.hawkeye.admin.ui.components.HawkButton
import com.hawkeye.admin.ui.components.LevelChip
import com.hawkeye.admin.ui.components.LinkRow
import com.hawkeye.admin.ui.components.RowDivider
import com.hawkeye.admin.ui.components.SectionLabel
import com.hawkeye.admin.ui.components.StatusCard
import com.hawkeye.admin.ui.components.TimelineItem
import com.hawkeye.admin.ui.components.levelTone
import com.hawkeye.admin.ui.theme.Hawk
import com.hawkeye.admin.ui.theme.HawkType
import com.hawkeye.admin.ui.theme.tone
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.launch

/**
 * 사건 상세: 무슨 일인지(작은 상태 카드) → 증거(관련 영상·숫자) → 품목별·판단 근거 → 메모·기록. 처리 단추는 아래에 고정.
 * 처리한 뒤에는 초록 카드 + '처리 다시 하기' (서버는 처리를 다시 받아 기록을 쌓는다).
 */
@Composable
fun CaseDetailScreen(caseId: Int, onBack: () -> Unit, onRecordings: (camera: String, startIso: String) -> Unit) {
    val repo = AppGraph.alerts
    val scope = rememberCoroutineScope()
    // 사건이 갱신되면(단계 상승·다른 폰에서 처리) 상세를 다시 불러온다.
    val summary by remember(caseId) { repo.cases(CaseFilter.ALL).map { repo.case(caseId) } }
        .collectAsState(initial = repo.case(caseId))
    var detail by remember { mutableStateOf<CaseDetail?>(null) }
    var image by remember { mutableStateOf<ImageBitmap?>(null) }
    var error by remember { mutableStateOf<String?>(null) }
    var note by remember { mutableStateOf("") }
    var saving by remember { mutableStateOf(false) }
    var reopen by remember { mutableStateOf(false) }
    var clips by remember { mutableStateOf<CaseClips?>(null) }

    LaunchedEffect(caseId, summary?.updatedAt) {
        runCatching { repo.detail(caseId) }
            .onSuccess { d ->
                detail = d
                error = null
                val eventId = d.events.lastOrNull { it.hasSnapshot }?.id
                image = eventId?.let { id ->
                    runCatching { repo.snapshot(id) }.getOrNull()
                        ?.let { BitmapFactory.decodeByteArray(it, 0, it.size)?.asImageBitmap() }
                }
            }
            .onFailure { error = it.message ?: "불러오지 못했어요" }
        // 관련 영상 (서버가 녹화를 지원할 때만). 단계가 올라가면 기준 경보가 바뀌므로 다시 받는다.
        clips = runCatching { if (repo.config().recordings) repo.caseClips(caseId) else null }.getOrNull()
    }

    fun resolve(r: Resolution) {
        saving = true
        scope.launch {
            runCatching { repo.resolve(caseId, r, note.trim().ifBlank { null }) }
                .onSuccess { note = ""; reopen = false }
                .onFailure { error = it.message ?: "저장하지 못했어요" }
            saving = false
        }
    }

    val c = detail?.case ?: summary
    val handled = c != null && c.state != "active" && !reopen
    Scaffold(
        containerColor = Hawk.colors.bg,
        topBar = {
            Box(Modifier.statusBarsPadding()) { AppBar("손님 ${c?.personId ?: ""}", onBack) {
                if (c != null) {
                    if (c.state == "resolved" || c.state == "auto_cleared") DoneChip(stateLabel(c.state, c.resolution).let { if (it == "자동 해제") it else "처리됨" })
                    else LevelChip(c.level)
                }
            } }
        },
        bottomBar = {
            if (c != null) {
                Row(Modifier.fillMaxWidth().background(Hawk.colors.bg).navigationBarsPadding().padding(start = 16.dp, end = 16.dp, top = 12.dp, bottom = 16.dp),
                    horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    if (handled && c.state == "resolved") {
                        HawkButton("처리 다시 하기", onClick = { reopen = true }, kind = ButtonKind.SOFT, icon = Icons.AutoMirrored.Rounded.Undo, modifier = Modifier.weight(1f))
                    } else if (!handled) {
                        HawkButton("잘못된 알림", onClick = { resolve(Resolution.FALSE_ALARM) }, kind = ButtonKind.SOFT, enabled = !saving, modifier = Modifier.weight(1f))
                        HawkButton("결제 확인 완료", onClick = { resolve(Resolution.PAID_CONFIRMED) }, icon = Icons.Rounded.Check, enabled = !saving, modifier = Modifier.weight(1.5f))
                    }
                }
            }
        },
    ) { pad ->
        Column(
            Modifier.padding(pad).imePadding().fillMaxSize().verticalScroll(rememberScrollState()).padding(horizontal = 16.dp).padding(bottom = 16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            error?.let { Text(it, style = HawkType.caption, color = Hawk.colors.danger) }
            if (c == null) {
                Text("불러오는 중…", style = HawkType.body, color = Hawk.colors.faint)
                return@Column
            }
            StatusCard(detailSummary(c, detail, reopen), compact = true)

            val cl = clips
            if (cl != null) {
                CaseClipsSection(cl)
            } else if (image != null) {
                Image(image!!, contentDescription = "경보 장면", contentScale = ContentScale.FillWidth,
                    modifier = Modifier.fillMaxWidth().clip(RoundedCornerShape(18.dp)))
            }
            Facts(c.taken, c.paid, c.unpaid, highlightUnpaid = c.state == "active")
            cl?.let { clipsInfo ->
                RecordingNav.shift(clipsInfo.anchorTime, -60)?.let { from ->
                    LinkRow(Icons.Rounded.History, "경보 1분 전부터 지난 영상") { onRecordings(c.cameraId, from) }
                }
            }

            c.items?.takeIf { it.isNotEmpty() }?.let { items ->
                SectionLabel("품목별")
                Group {
                    items.forEachIndexed { i, it ->
                        if (i > 0) RowDivider()
                        Row(Modifier.fillMaxWidth().padding(horizontal = 18.dp, vertical = 12.dp)) {
                            Text(it.name, style = HawkType.body, color = Hawk.colors.sub, modifier = Modifier.weight(1f))
                            Text("가져감 ${it.taken} ", style = HawkType.listTitle, color = Hawk.colors.text)
                            Text("결제 ${it.paid}", style = HawkType.listTitle, color = if (it.paid < it.taken) Hawk.colors.danger else Hawk.colors.text)
                        }
                    }
                }
            }

            val reason = listOfNotNull(
                c.reason?.takeIf { it.isNotBlank() },
                c.identityCheck?.let { "같은 사람으로 이어 본 정도 ${"%.2f".format(it)} (낮으면 번호가 다른 사람에게 옮겨 붙었을 수 있어요)" },
            )
            if (reason.isNotEmpty()) {
                SectionLabel("판단 근거")
                Text(reason.joinToString("\n"), style = HawkType.bodySmall, color = Hawk.colors.sub,
                    modifier = Modifier.fillMaxWidth().clip(RoundedCornerShape(20.dp)).background(Hawk.colors.surf).padding(horizontal = 18.dp, vertical = 16.dp))
            }

            if (!handled) {
                SectionLabel("메모")
                Box(Modifier.fillMaxWidth().heightIn(min = 64.dp).clip(RoundedCornerShape(16.dp)).background(Hawk.colors.surf).padding(horizontal = 16.dp, vertical = 14.dp)) {
                    if (note.isEmpty()) Text("처리 내용을 남겨 두세요 (선택)", style = HawkType.body, color = Hawk.colors.faint)
                    BasicTextField(note, { note = it }, textStyle = HawkType.body.copy(color = Hawk.colors.text),
                        cursorBrush = SolidColor(Hawk.colors.brand), modifier = Modifier.fillMaxWidth())
                }
            }

            detail?.let { d ->
                if (d.resolutions.isNotEmpty()) {
                    SectionLabel("처리 기록")
                    Group(Modifier.padding(0.dp)) {
                        Column(Modifier.padding(horizontal = 18.dp, vertical = 4.dp)) {
                            d.resolutions.forEachIndexed { i, r ->
                                TimelineItem(if (r.resolution == "false_alarm") Hawk.colors.faint else Hawk.colors.ok,
                                    resolutionLabel(r.resolution), r.note?.let { "\"$it\"" }, localTime(r.at), last = i == d.resolutions.lastIndex)
                            }
                        }
                    }
                }
                if (d.events.isNotEmpty()) {
                    SectionLabel("경보 기록")
                    Group {
                        Column(Modifier.padding(horizontal = 18.dp, vertical = 4.dp)) {
                            d.events.forEachIndexed { i, e ->
                                val tone = Hawk.colors.tone(levelTone(e.level))
                                TimelineItem(if (e.level == "CLEAR") Hawk.colors.faint else tone.main, levelLabel(e.level),
                                    e.reason?.takeIf { it.isNotBlank() }, eventTime(e), last = i == d.events.lastIndex)
                            }
                        }
                    }
                }
            }
        }
    }
}

/** 상세 위 작은 상태 카드 내용. */
private fun detailSummary(c: CaseSummary, d: CaseDetail?, reopen: Boolean): StatusSummary {
    val time = d?.events?.lastOrNull { it.level != "CLEAR" }?.let(::eventTime) ?: localTime(c.updatedAt)
    val title = when (c.level) {
        "HIGH_RISK" -> "결제 없이 나갔어요"
        "REVIEW" -> "${c.taken}개 중 ${c.paid}개만 결제했어요"
        "WARNING" -> "결제 전에 출구로 갔어요"
        else -> "결제 확인이 필요해요"
    }
    return when {
        c.state == "resolved" && !reopen -> StatusSummary(
            tone = Tone.OK, label = resolutionLabel(c.resolution), meta = c.updatedAt.let(::localHm).let { "$it 처리" },
            title = "처리를 마쳤어요", detail = c.resolutionNote?.let { "\"$it\"" } ?: title,
            action = StatusAction.LIVE, actionLabel = "",
        )
        c.state == "auto_cleared" -> StatusSummary(
            tone = Tone.OK, label = "자동 해제", meta = time,
            title = "경보가 풀렸어요", detail = "되돌아가 결제했거나 매장 안으로 돌아왔어요.",
            action = StatusAction.LIVE, actionLabel = "",
        )
        else -> StatusSummary(
            tone = levelTone(c.level), label = if (reopen) "처리를 다시 고르는 중" else "아직 처리하지 않았어요", meta = time,
            title = title, detail = StatusSummaries.unpaidPhrase(c).takeIf { c.level == "HIGH_RISK" }?.let { "$it ${cameraLabel(c.cameraId)}" } ?: cameraLabel(c.cameraId),
            action = StatusAction.LIVE, actionLabel = "",
        )
    }
}

/** 경보 기록 시각: 장면 시각(절대 시각)이면 그것, 아니면(상대 시각) 서버가 받은 시각. */
private fun eventTime(e: com.hawkeye.admin.data.EventRecord): String {
    val ts = e.timestamp
    if (ts != null) {
        val t = localTime(ts)
        if (t != ts) return t
    }
    return localTime(e.receivedAt)
}
