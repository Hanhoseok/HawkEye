package com.hawkeye.admin.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import java.time.OffsetDateTime
import java.time.ZoneId
import java.time.format.DateTimeFormatter

private val Navy = Color(0xFF1F3864)

@Composable
fun HawkEyeTheme(content: @Composable () -> Unit) {
    MaterialTheme(colorScheme = lightColorScheme(primary = Navy), content = content)
}

fun levelColor(level: String): Color = when (level) {
    "HIGH_RISK" -> Color(0xFFC0392B)
    "REVIEW" -> Color(0xFFE67E22)
    "WARNING" -> Color(0xFFD4A017)
    else -> Color(0xFF7F8C8D)
}

fun levelLabel(level: String): String = when (level) {
    "HIGH_RISK" -> "위험"
    "REVIEW" -> "확인 필요"
    "WARNING" -> "주의"
    "CLEAR" -> "정상"
    else -> level
}

fun stateLabel(state: String, resolution: String?): String = when (state) {
    "active" -> "미처리"
    "resolved" -> "처리됨 · " + resolutionLabel(resolution)
    "auto_cleared" -> "자동 해제 (결제·복귀)"
    "pass" -> "통과"
    else -> state
}

fun resolutionLabel(resolution: String?): String = when (resolution) {
    "paid_confirmed" -> "결제 확인 완료"
    "false_alarm" -> "잘못된 알림"
    else -> "-"
}

private val clock = DateTimeFormatter.ofPattern("HH:mm:ss")

/** 서버 시각(ISO8601, UTC)을 폰 시간대의 시:분:초로. */
fun localTime(iso: String): String =
    runCatching { OffsetDateTime.parse(iso).atZoneSameInstant(ZoneId.systemDefault()).format(clock) }.getOrDefault(iso)

@Composable
fun LevelBadge(level: String, modifier: Modifier = Modifier) {
    Text(
        text = levelLabel(level),
        color = Color.White,
        fontWeight = FontWeight.Bold,
        fontSize = 13.sp,
        modifier = modifier
            .background(levelColor(level), RoundedCornerShape(6.dp))
            .padding(horizontal = 8.dp, vertical = 3.dp),
    )
}
