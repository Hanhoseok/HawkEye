package com.hawkeye.admin.ui.theme

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Typography
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.Immutable
import androidx.compose.runtime.staticCompositionLocalOf
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.sp
import com.hawkeye.admin.domain.Tone

/**
 * 디자인 기준 (시안 C 상태등 다듬은 판: .ai-review/admin-app-design/c2-screens.html '디자인 기준').
 * 밝은 회색 바탕 위에 흰 묶음, 색은 상태 카드와 단계 표시에만.
 */
@Immutable
data class HawkColors(
    val bg: Color = Color(0xFFF2F4F6),
    val surf: Color = Color(0xFFFFFFFF),
    val text: Color = Color(0xFF191F28),
    val sub: Color = Color(0xFF4E5968),
    val faint: Color = Color(0xFF8B95A1),
    val line: Color = Color(0xFFEEF0F3),
    val fill: Color = Color(0xFFF2F4F6),
    val fill2: Color = Color(0xFFE5E8EB),
    val brand: Color = Color(0xFF0B9E92),
    val brandTint: Color = Color(0xFFE3F6F3),
    val brandInk: Color = Color(0xFF087A70),
    val danger: Color = Color(0xFFF04452),
    val dangerTint: Color = Color(0xFFFFEEF0),
    val dangerInk: Color = Color(0xFFD32638),
    val review: Color = Color(0xFFF76808),
    val reviewTint: Color = Color(0xFFFFF1E6),
    val reviewInk: Color = Color(0xFFC24E00),
    val warn: Color = Color(0xFFE5A000),
    val warnTint: Color = Color(0xFFFFF7DE),
    val warnInk: Color = Color(0xFF946200),
    val ok: Color = Color(0xFF1DA45E),
    val okTint: Color = Color(0xFFE8F7EE),
    val okInk: Color = Color(0xFF137A44),
    val off: Color = Color(0xFF6B7684),
    val offTint: Color = Color(0xFFEEF0F3),
    val offInk: Color = Color(0xFF4E5968),
)

/** 한 단계의 세 가지 색: 진한 색(점·아이콘·단추), 옅은 바탕, 글자. */
data class ToneColors(val main: Color, val tint: Color, val ink: Color)

fun HawkColors.tone(t: Tone): ToneColors = when (t) {
    Tone.DANGER -> ToneColors(danger, dangerTint, dangerInk)
    Tone.REVIEW -> ToneColors(review, reviewTint, reviewInk)
    Tone.WARN -> ToneColors(warn, warnTint, warnInk)
    Tone.CALM -> ToneColors(brand, surf, brandInk)
    Tone.OFF -> ToneColors(off, offTint, offInk)
    Tone.OK -> ToneColors(ok, okTint, okInk)
}

/** 글자 단계 (dp 기준 시안과 같음). 굵기는 400·500·600·700 만. */
object HawkType {
    private val base = TextStyle(letterSpacing = (-0.2).sp)
    val screenTitle = base.copy(fontSize = 20.sp, fontWeight = FontWeight.Bold, lineHeight = 28.sp, letterSpacing = (-0.5).sp)
    val statusTitle = base.copy(fontSize = 20.sp, fontWeight = FontWeight.SemiBold, lineHeight = 28.sp, letterSpacing = (-0.5).sp)
    val appBar = base.copy(fontSize = 17.sp, fontWeight = FontWeight.SemiBold, lineHeight = 24.sp)
    val listTitle = base.copy(fontSize = 15.sp, fontWeight = FontWeight.SemiBold, lineHeight = 21.sp)
    val button = base.copy(fontSize = 15.sp, fontWeight = FontWeight.SemiBold, lineHeight = 20.sp)
    val body = base.copy(fontSize = 15.sp, fontWeight = FontWeight.Normal, lineHeight = 22.sp)
    val bodySmall = base.copy(fontSize = 14.sp, fontWeight = FontWeight.Normal, lineHeight = 21.sp)
    val caption = base.copy(fontSize = 13.sp, fontWeight = FontWeight.Normal, lineHeight = 19.sp)
    val label = base.copy(fontSize = 13.sp, fontWeight = FontWeight.SemiBold, lineHeight = 18.sp)
    val tiny = base.copy(fontSize = 11.sp, fontWeight = FontWeight.Medium, lineHeight = 14.sp, letterSpacing = 0.sp)
    val number = base.copy(fontSize = 22.sp, fontWeight = FontWeight.SemiBold, lineHeight = 28.sp, fontFeatureSettings = "tnum")
    val clock = base.copy(fontSize = 20.sp, fontWeight = FontWeight.SemiBold, lineHeight = 26.sp, fontFeatureSettings = "tnum")
}

val LocalHawkColors = staticCompositionLocalOf { HawkColors() }

object Hawk {
    val colors: HawkColors
        @Composable get() = LocalHawkColors.current
}

/** 한 줄로 자를 글자 */
val Ellipsis = TextOverflow.Ellipsis

@Composable
fun HawkEyeTheme(content: @Composable () -> Unit) {
    val c = HawkColors()
    val scheme = lightColorScheme(
        primary = c.brand,
        onPrimary = Color.White,
        primaryContainer = c.brandTint,
        onPrimaryContainer = c.brandInk,
        secondary = c.text,
        onSecondary = Color.White,
        background = c.bg,
        onBackground = c.text,
        surface = c.surf,
        onSurface = c.text,
        surfaceVariant = c.fill,
        onSurfaceVariant = c.sub,
        outline = c.fill2,
        outlineVariant = c.line,
        error = c.danger,
        onError = Color.White,
    )
    val typography = Typography(
        titleLarge = HawkType.screenTitle,
        titleMedium = HawkType.appBar,
        titleSmall = HawkType.listTitle,
        bodyLarge = HawkType.body,
        bodyMedium = HawkType.bodySmall,
        bodySmall = HawkType.caption,
        labelLarge = HawkType.button,
        labelMedium = HawkType.label,
        labelSmall = HawkType.tiny,
    )
    CompositionLocalProvider(LocalHawkColors provides c) {
        MaterialTheme(colorScheme = scheme, typography = typography, content = content)
    }
}
