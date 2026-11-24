package com.hawkeye.admin.ui.components

import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.RowScope
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.rounded.ArrowBack
import androidx.compose.material.icons.automirrored.rounded.Help
import androidx.compose.material.icons.automirrored.rounded.KeyboardArrowRight
import androidx.compose.material.icons.rounded.CheckCircle
import androidx.compose.material.icons.rounded.Done
import androidx.compose.material.icons.rounded.GppBad
import androidx.compose.material.icons.rounded.Warning
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.Switch
import androidx.compose.material3.SwitchDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.composed
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.RectangleShape
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import com.hawkeye.admin.R
import com.hawkeye.admin.domain.StatusSummary
import com.hawkeye.admin.domain.Tone
import com.hawkeye.admin.ui.levelLabel
import com.hawkeye.admin.ui.theme.Ellipsis
import com.hawkeye.admin.ui.theme.Hawk
import com.hawkeye.admin.ui.theme.HawkType
import com.hawkeye.admin.ui.theme.ToneColors
import com.hawkeye.admin.ui.theme.tone

// ---- 로고 · 단계 ----

/** 앱 로고 타일 (가격표 독수리). */
@Composable
fun BrandMark(size: Dp = 26.dp, modifier: Modifier = Modifier) {
    Image(painterResource(R.drawable.ic_brand_tile), contentDescription = null, modifier = modifier.size(size))
}

fun levelTone(level: String): Tone = when (level) {
    "HIGH_RISK" -> Tone.DANGER
    "REVIEW" -> Tone.REVIEW
    "WARNING" -> Tone.WARN
    else -> Tone.CALM
}

fun levelIcon(level: String): ImageVector = when (level) {
    "HIGH_RISK" -> Icons.Rounded.GppBad
    "REVIEW" -> Icons.AutoMirrored.Rounded.Help
    "WARNING" -> Icons.Rounded.Warning
    else -> Icons.Rounded.CheckCircle
}

/** 단계 표시: 색 + 아이콘 + 글자 (색만으로 뜻을 전하지 않음). */
@Composable
fun LevelChip(level: String, modifier: Modifier = Modifier) {
    val t = Hawk.colors.tone(levelTone(level))
    Row(
        modifier.clip(CircleShape).background(t.tint.takeIf { levelTone(level) != Tone.CALM } ?: Hawk.colors.okTint)
            .padding(start = 7.dp, end = 10.dp, top = 5.dp, bottom = 5.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(4.dp),
    ) {
        Icon(levelIcon(level), null, tint = if (levelTone(level) == Tone.CALM) Hawk.colors.ok else t.main, modifier = Modifier.size(16.dp))
        Text(levelLabel(level), style = HawkType.label, color = if (levelTone(level) == Tone.CALM) Hawk.colors.okInk else t.ink)
    }
}

/** 처리된 사건 표시 (초록). */
@Composable
fun DoneChip(text: String, modifier: Modifier = Modifier) {
    Row(
        modifier.clip(CircleShape).background(Hawk.colors.okTint).padding(start = 7.dp, end = 10.dp, top = 5.dp, bottom = 5.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(4.dp),
    ) {
        Icon(Icons.Rounded.CheckCircle, null, tint = Hawk.colors.ok, modifier = Modifier.size(16.dp))
        Text(text, style = HawkType.label, color = Hawk.colors.okInk)
    }
}

@Composable
private fun Dot(color: Color) {
    Box(Modifier.size(16.dp).background(color.copy(alpha = 0.18f), CircleShape), contentAlignment = Alignment.Center) {
        Box(Modifier.size(8.dp).background(color, CircleShape))
    }
}

// ---- 상태 카드 (상태등) ----

/**
 * 경보 목록 맨 위 상태 카드. 평소엔 흰 카드, 경보가 있으면 그 단계 색의 옅은 카드, 끊기면 회색.
 * compact = 사건 상세 위의 작은 카드 (단추 없음).
 */
@Composable
fun StatusCard(
    s: StatusSummary,
    modifier: Modifier = Modifier,
    compact: Boolean = false,
    actionIcon: ImageVector? = null,
    onAction: (() -> Unit)? = null,
) {
    val c = Hawk.colors
    val t = c.tone(s.tone)
    Column(
        modifier.fillMaxWidth().clip(RoundedCornerShape(22.dp)).background(t.tint)
            .padding(start = 18.dp, end = 18.dp, top = 16.dp, bottom = if (compact) 16.dp else 16.dp),
    ) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Dot(t.main)
            Spacer(Modifier.width(4.dp))
            Text(s.label, style = HawkType.label, color = t.ink, modifier = Modifier.weight(1f), maxLines = 1, overflow = Ellipsis)
            if (s.meta.isNotBlank()) Text(s.meta, style = HawkType.caption, color = c.faint)
        }
        Spacer(Modifier.height(if (compact) 6.dp else 10.dp))
        Text(s.title, style = if (compact) HawkType.statusTitle.copy(fontSize = HawkType.appBar.fontSize.times(1.06f)) else HawkType.statusTitle, color = c.text)
        if (s.detail.isNotBlank()) {
            Spacer(Modifier.height(4.dp))
            Text(s.detail, style = HawkType.bodySmall, color = c.sub)
        }
        if (onAction != null && !compact) {
            Spacer(Modifier.height(16.dp))
            val soft = s.tone == Tone.CALM
            val bg = when {
                soft -> c.brandTint
                s.tone == Tone.OFF -> c.text
                else -> t.main
            }
            val fg = if (soft) c.brandInk else Color.White
            Row(
                Modifier.fillMaxWidth().height(46.dp).clip(RoundedCornerShape(14.dp)).background(bg).clickable(onClick = onAction),
                horizontalArrangement = Arrangement.Center,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                if (actionIcon != null) {
                    Icon(actionIcon, null, tint = fg, modifier = Modifier.size(20.dp))
                    Spacer(Modifier.width(6.dp))
                }
                Text(s.actionLabel, style = HawkType.button, color = fg)
            }
        }
    }
}

// ---- 묶음 · 줄 ----

/** 흰 묶음 (모서리 20). 안의 줄은 옅은 구분선으로 나눈다. */
@Composable
fun Group(modifier: Modifier = Modifier, content: @Composable ColumnScope.() -> Unit) {
    Column(modifier.fillMaxWidth().clip(RoundedCornerShape(20.dp)).background(Hawk.colors.surf), content = content)
}

/**
 * LazyColumn 에서 한 묶음을 줄마다 따로 그릴 때: 첫 줄은 위 모서리, 마지막 줄은 아래 모서리만 둥글게.
 * (묶음 전체를 한 항목으로 만들면 줄이 많을 때 화면 밖 줄까지 모두 그려진다.)
 */
fun Modifier.groupItem(index: Int, count: Int): Modifier = composed {
    val r = 20.dp
    val shape = when {
        count == 1 -> RoundedCornerShape(r)
        index == 0 -> RoundedCornerShape(topStart = r, topEnd = r)
        index == count - 1 -> RoundedCornerShape(bottomStart = r, bottomEnd = r)
        else -> RectangleShape
    }
    fillMaxWidth().clip(shape).background(Hawk.colors.surf)
}

/** 묶음 안 줄 사이 구분선 (왼쪽 들여쓰기). */
@Composable
fun RowDivider(start: Dp = 18.dp) {
    Box(Modifier.padding(start = start, end = 18.dp).fillMaxWidth().height(1.dp).background(Hawk.colors.line))
}

/** 경보 목록 한 줄: 단계 아이콘 · 제목 · 요약 · 시각. muted = 처리된 사건. */
@Composable
fun AlertRow(
    level: String,
    title: String,
    subtitle: String,
    time: String,
    onClick: () -> Unit,
    mutedIcon: ImageVector? = null,
) {
    val c = Hawk.colors
    val muted = mutedIcon != null
    val t: ToneColors = if (muted) ToneColors(c.faint, c.fill, c.sub) else c.tone(levelTone(level))
    Row(
        Modifier.fillMaxWidth().clickable(onClick = onClick).padding(horizontal = 18.dp, vertical = 12.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Box(Modifier.size(36.dp).clip(RoundedCornerShape(12.dp)).background(t.tint), contentAlignment = Alignment.Center) {
            Icon(mutedIcon ?: levelIcon(level), null, tint = t.main, modifier = Modifier.size(20.dp))
        }
        Spacer(Modifier.width(12.dp))
        Column(Modifier.weight(1f)) {
            Text(title, style = if (muted) HawkType.body.copy(fontWeight = HawkType.label.fontWeight?.let { androidx.compose.ui.text.font.FontWeight.Medium })
                else HawkType.listTitle, color = if (muted) c.sub else c.text, maxLines = 1, overflow = Ellipsis)
            Text(subtitle, style = HawkType.caption, color = c.faint, maxLines = 1, overflow = Ellipsis)
        }
        Spacer(Modifier.width(8.dp))
        Text(time, style = HawkType.caption, color = c.faint, modifier = Modifier.align(Alignment.Top).padding(top = 2.dp))
    }
}

/** 묶음 위 작은 제목. */
@Composable
fun SectionLabel(text: String, modifier: Modifier = Modifier, trailing: String? = null) {
    Row(modifier.fillMaxWidth().padding(horizontal = 4.dp), verticalAlignment = Alignment.Bottom) {
        Text(text, style = HawkType.label, color = Hawk.colors.faint, modifier = Modifier.weight(1f))
        if (trailing != null) Text(trailing, style = HawkType.caption.copy(fontSize = HawkType.tiny.fontSize.times(1.1f)), color = Hawk.colors.faint)
    }
}

/** 누르면 다른 화면으로 가는 한 줄 ("경보 1분 전부터 지난 영상"). */
@Composable
fun LinkRow(icon: ImageVector, text: String, onClick: () -> Unit) {
    val c = Hawk.colors
    Row(
        Modifier.fillMaxWidth().clip(RoundedCornerShape(18.dp)).background(c.surf).clickable(onClick = onClick)
            .padding(horizontal = 16.dp, vertical = 15.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Icon(icon, null, tint = c.brand, modifier = Modifier.size(20.dp))
        Spacer(Modifier.width(10.dp))
        Text(text, style = HawkType.body.copy(fontWeight = androidx.compose.ui.text.font.FontWeight.Medium), color = c.text, modifier = Modifier.weight(1f))
        Icon(Icons.AutoMirrored.Rounded.KeyboardArrowRight, null, tint = c.fill2)
    }
}

/** 안내 상자 (옅은 색 바탕 + 아이콘). */
@Composable
fun Note(
    icon: ImageVector,
    main: Color,
    tint: Color,
    text: String,
    modifier: Modifier = Modifier,
    title: String? = null,
    actionLabel: String? = null,
    onAction: (() -> Unit)? = null,
) {
    val c = Hawk.colors
    Row(
        modifier.fillMaxWidth().clip(RoundedCornerShape(18.dp)).background(tint)
            .then(if (onAction != null) Modifier.clickable(onClick = onAction) else Modifier)
            .padding(horizontal = 16.dp, vertical = 14.dp),
        verticalAlignment = if (title == null) Alignment.CenterVertically else Alignment.Top,
    ) {
        Icon(icon, null, tint = main, modifier = Modifier.size(20.dp))
        Spacer(Modifier.width(10.dp))
        Column(Modifier.weight(1f)) {
            if (title != null) Text(title, style = HawkType.listTitle.copy(fontSize = HawkType.bodySmall.fontSize), color = c.text)
            Text(text, style = HawkType.caption, color = c.sub)
        }
        if (actionLabel != null) {
            Spacer(Modifier.width(8.dp))
            Text(actionLabel, style = HawkType.label, color = main, modifier = Modifier.align(Alignment.CenterVertically))
        }
    }
}

// ---- 숫자 · 단추 · 분할 단추 ----

/** 가져감 / 결제 / 미결제 3칸. */
@Composable
fun Facts(taken: Int, paid: Int, unpaid: Int, highlightUnpaid: Boolean = true) {
    val c = Hawk.colors
    Row(Modifier.fillMaxWidth().clip(RoundedCornerShape(20.dp)).background(c.surf).padding(vertical = 16.dp)) {
        listOf("가져감" to taken, "결제" to paid, "미결제" to unpaid).forEachIndexed { i, (label, v) ->
            if (i > 0) Box(Modifier.width(1.dp).height(44.dp).background(c.line))
            Column(Modifier.weight(1f).padding(horizontal = 16.dp)) {
                Text(label, style = HawkType.caption, color = c.faint)
                Text("$v", style = HawkType.number, color = if (i == 2 && highlightUnpaid && v > 0) c.danger else c.text)
            }
        }
    }
}

enum class ButtonKind { PRIMARY, SOFT, WHITE, BRAND }

@Composable
fun HawkButton(
    text: String,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
    kind: ButtonKind = ButtonKind.PRIMARY,
    icon: ImageVector? = null,
    enabled: Boolean = true,
    height: Dp = 52.dp,
) {
    val c = Hawk.colors
    val (bg, fg) = when (kind) {
        ButtonKind.PRIMARY -> c.text to Color.White
        ButtonKind.SOFT -> c.fill2 to c.text
        ButtonKind.WHITE -> c.surf to c.text
        ButtonKind.BRAND -> c.brand to Color.White
    }
    Row(
        modifier.height(height).clip(RoundedCornerShape(14.dp)).background(if (enabled) bg else bg.copy(alpha = 0.5f))
            .clickable(enabled = enabled, onClick = onClick).padding(horizontal = 14.dp),
        horizontalArrangement = Arrangement.Center,
        verticalAlignment = Alignment.CenterVertically,
    ) {
        if (icon != null) {
            Icon(icon, null, tint = fg, modifier = Modifier.size(20.dp))
            Spacer(Modifier.width(6.dp))
        }
        Text(text, style = HawkType.button, color = fg, maxLines = 1)
    }
}

/** 회색 홈 안의 분할 단추 (카메라 고르기, 확인 필요/전체). */
@Composable
fun Segmented(
    options: List<String>,
    selected: Int,
    onSelect: (Int) -> Unit,
    modifier: Modifier = Modifier,
    trailing: List<String?> = emptyList(),
    dots: List<Color?> = emptyList(),
) {
    val c = Hawk.colors
    Row(modifier.fillMaxWidth().clip(RoundedCornerShape(12.dp)).background(c.fill2).padding(3.dp)) {
        options.forEachIndexed { i, label ->
            val on = i == selected
            Row(
                Modifier.weight(1f).heightIn(min = 38.dp).clip(RoundedCornerShape(10.dp))
                    .background(if (on) c.surf else Color.Transparent).clickable { onSelect(i) }.padding(vertical = 8.dp),
                horizontalArrangement = Arrangement.Center,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                dots.getOrNull(i)?.let {
                    Box(Modifier.size(6.dp).background(it, CircleShape))
                    Spacer(Modifier.width(6.dp))
                }
                Text(label, style = if (on) HawkType.label.copy(fontSize = HawkType.bodySmall.fontSize) else HawkType.bodySmall.copy(fontWeight = androidx.compose.ui.text.font.FontWeight.Medium),
                    color = if (on) c.text else c.sub)
                trailing.getOrNull(i)?.let {
                    Spacer(Modifier.width(4.dp))
                    Text(it, style = HawkType.label.copy(fontSize = HawkType.bodySmall.fontSize), color = c.danger)
                }
            }
        }
    }
}

/** 둥근 칩 (사건 클립의 카메라 고르기). */
@Composable
fun Chip(text: String, selected: Boolean, onClick: () -> Unit) {
    val c = Hawk.colors
    Text(
        text,
        style = if (selected) HawkType.label else HawkType.caption.copy(fontWeight = androidx.compose.ui.text.font.FontWeight.Medium),
        color = if (selected) Color.White else c.sub,
        modifier = Modifier.clip(CircleShape).background(if (selected) c.text else c.surf).clickable(onClick = onClick)
            .padding(horizontal = 12.dp, vertical = 7.dp),
    )
}

/** 흰 줄 + 스위치 (AI 탐지 화면). */
@Composable
fun SwitchRow(title: String, subtitle: String, checked: Boolean, onChange: (Boolean) -> Unit) {
    val c = Hawk.colors
    Row(
        Modifier.fillMaxWidth().clip(RoundedCornerShape(20.dp)).background(c.surf).clickable { onChange(!checked) }
            .padding(horizontal = 18.dp, vertical = 14.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Column(Modifier.weight(1f)) {
            Text(title, style = HawkType.listTitle, color = c.text)
            Text(subtitle, style = HawkType.caption, color = c.faint)
        }
        Spacer(Modifier.width(12.dp))
        Switch(
            checked = checked, onCheckedChange = onChange,
            colors = SwitchDefaults.colors(
                checkedThumbColor = Color.White, checkedTrackColor = c.brand, checkedBorderColor = c.brand,
                uncheckedThumbColor = Color.White, uncheckedTrackColor = c.fill2, uncheckedBorderColor = c.fill2,
            ),
        )
    }
}

// ---- 화면 머리 ----

/** 탭 화면 제목 (경보는 로고와 함께). */
@Composable
fun ScreenTitle(title: String, modifier: Modifier = Modifier, brand: Boolean = false, actions: @Composable RowScope.() -> Unit = {}) {
    Row(modifier.fillMaxWidth().padding(start = 20.dp, end = 8.dp, top = 8.dp, bottom = 8.dp).heightIn(min = 40.dp),
        verticalAlignment = Alignment.CenterVertically) {
        if (brand) {
            BrandMark(26.dp)
            Spacer(Modifier.width(8.dp))
            Text(title, style = HawkType.screenTitle.copy(fontSize = HawkType.appBar.fontSize.times(1.06f)), color = Hawk.colors.text, modifier = Modifier.weight(1f))
        } else {
            Text(title, style = HawkType.screenTitle, color = Hawk.colors.text, modifier = Modifier.weight(1f))
        }
        actions()
    }
}

/** 상세 화면 앱 바: 뒤로 · 제목 · 오른쪽 표시. */
@Composable
fun AppBar(title: String, onBack: () -> Unit, trailing: @Composable RowScope.() -> Unit = {}) {
    Row(Modifier.fillMaxWidth().padding(start = 4.dp, end = 16.dp, top = 4.dp, bottom = 4.dp), verticalAlignment = Alignment.CenterVertically) {
        IconButton(onClick = onBack) { Icon(Icons.AutoMirrored.Rounded.ArrowBack, contentDescription = "뒤로", tint = Hawk.colors.text) }
        Text(title, style = HawkType.appBar, color = Hawk.colors.text, modifier = Modifier.weight(1f), maxLines = 1, overflow = Ellipsis)
        trailing()
    }
}

// ---- 영상 가림막 · 빈 상태 ----

/** 영상 위 어두운 가림막 (송출 멈춤, AI 화면 없음, 영상 준비 중). 영상 위라 색은 고정. */
@Composable
fun Veil(icon: ImageVector, title: String, text: String?) {
    Column(
        Modifier.fillMaxSize().background(Color(0xBC141A22)).padding(horizontal = 24.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Icon(icon, null, tint = Color.White, modifier = Modifier.size(26.dp))
        Spacer(Modifier.height(6.dp))
        Text(title, style = HawkType.listTitle, color = Color.White, textAlign = TextAlign.Center)
        if (text != null) Text(text, style = HawkType.caption, color = Color.White.copy(alpha = 0.75f), textAlign = TextAlign.Center)
    }
}

/** 영상 위 작은 표시 ("실시간", "AI"). */
@Composable
fun VideoTag(text: String, modifier: Modifier = Modifier, live: Boolean = false, light: Boolean = false) {
    Row(
        modifier.clip(CircleShape).background(if (light) Color(0xEBFFFFFF) else Color(0x80000000)).padding(horizontal = 9.dp, vertical = 5.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        if (live) {
            Box(Modifier.size(6.dp).background(Color(0xFFFF4D5E), CircleShape))
            Spacer(Modifier.width(6.dp))
        }
        Text(text, style = HawkType.label.copy(fontSize = HawkType.caption.fontSize.times(0.92f)), color = if (light) Color(0xFF191F28) else Color.White)
    }
}

@Composable
fun EmptyState(icon: ImageVector, title: String, text: String, modifier: Modifier = Modifier) {
    Column(modifier.fillMaxWidth().padding(horizontal = 24.dp, vertical = 30.dp), horizontalAlignment = Alignment.CenterHorizontally) {
        Icon(icon, null, tint = Hawk.colors.fill2, modifier = Modifier.size(32.dp))
        Spacer(Modifier.height(10.dp))
        Text(title, style = HawkType.listTitle, color = Hawk.colors.sub, textAlign = TextAlign.Center)
        Text(text, style = HawkType.bodySmall, color = Hawk.colors.faint, textAlign = TextAlign.Center)
    }
}

// ---- 막대 (지난 영상) ----

data class BarMark(val at: Float, val exact: Boolean, val color: Color)

/** 녹화 구간을 칠하고 경보 시각에 세로선을 긋는 막대. cursor = 재생 위치. */
@Composable
fun TimeBar(
    segments: List<Pair<Float, Float>>,
    marks: List<BarMark>,
    ticks: List<String>,
    modifier: Modifier = Modifier,
    cursor: Float? = null,
    barHeight: Dp = 28.dp,
) {
    val c = Hawk.colors
    Column(modifier.fillMaxWidth()) {
        BoxWithConstraints(Modifier.fillMaxWidth().height(barHeight + 12.dp), contentAlignment = Alignment.CenterStart) {
            val w = maxWidth
            Box(Modifier.fillMaxWidth().height(barHeight).clip(RoundedCornerShape(8.dp)).background(c.fill)) {
                segments.forEach { (s, e) ->
                    Box(Modifier.offset(x = w * s).width((w * (e - s)).coerceAtLeast(2.dp)).fillMaxHeight()
                        .clip(RoundedCornerShape(6.dp)).background(c.brand.copy(alpha = 0.45f)))
                }
            }
            marks.forEach { m ->
                val mod = Modifier.offset(x = w * m.at - 1.5.dp).width(3.dp).height(barHeight + 8.dp).clip(RoundedCornerShape(2.dp))
                if (m.exact) Box(mod.background(m.color)) else Box(mod.border(1.dp, m.color, RoundedCornerShape(2.dp)))
            }
            if (cursor != null) {
                Box(Modifier.offset(x = w * cursor - 1.dp).width(2.dp).height(barHeight + 12.dp).background(c.text))
                Box(Modifier.offset(x = w * cursor - 5.dp, y = -(barHeight / 2 + 6.dp)).size(10.dp).background(c.text, CircleShape))
            }
        }
        if (ticks.isNotEmpty()) {
            Row(Modifier.fillMaxWidth().padding(top = 4.dp), horizontalArrangement = Arrangement.SpaceBetween) {
                ticks.forEach { Text(it, style = HawkType.tiny, color = c.faint) }
            }
        }
    }
}

/** 처리 기록·경보 기록 타임라인 한 줄. */
@Composable
fun TimelineItem(color: Color, title: String, subtitle: String?, time: String, last: Boolean) {
    val c = Hawk.colors
    Row(Modifier.fillMaxWidth()) {
        Column(horizontalAlignment = Alignment.CenterHorizontally, modifier = Modifier.width(10.dp)) {
            Spacer(Modifier.height(15.dp))
            Box(Modifier.size(10.dp).background(color, CircleShape))
            if (!last) Box(Modifier.width(2.dp).height(34.dp).background(c.line))
        }
        Spacer(Modifier.width(12.dp))
        Column(Modifier.weight(1f).padding(vertical = 10.dp)) {
            Text(title, style = HawkType.listTitle, color = c.text)
            if (subtitle != null) Text(subtitle, style = HawkType.caption, color = c.faint)
        }
        Text(time, style = HawkType.caption, color = c.faint, modifier = Modifier.padding(top = 10.dp))
    }
}

/** 처리된 사건 아이콘 (목록에서 흐리게). */
val DoneIcon: ImageVector get() = Icons.Rounded.Done
