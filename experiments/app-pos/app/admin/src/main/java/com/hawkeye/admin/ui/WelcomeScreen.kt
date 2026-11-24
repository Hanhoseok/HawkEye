package com.hawkeye.admin.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.hawkeye.admin.ui.components.BrandMark
import com.hawkeye.admin.ui.components.ButtonKind
import com.hawkeye.admin.ui.components.HawkButton
import com.hawkeye.admin.ui.theme.Hawk
import com.hawkeye.admin.ui.theme.HawkType

/** 처음 켰을 때 (서버 설정 없음): 할 일 세 가지. 알림 권한은 이 안내와 서버 설정을 마친 뒤 묻는다. */
@Composable
fun WelcomeScreen(onStart: () -> Unit) {
    val c = Hawk.colors
    Column(Modifier.fillMaxSize().background(c.surf).safeDrawingPadding().padding(horizontal = 24.dp, vertical = 20.dp)) {
        Spacer(Modifier.height(48.dp))
        BrandMark(60.dp)
        Spacer(Modifier.height(22.dp))
        Text("매장 경보를\n폰으로 받아 보세요", style = HawkType.screenTitle.copy(fontSize = HawkType.screenTitle.fontSize * 1.2f, lineHeight = HawkType.screenTitle.lineHeight * 1.15f), color = c.text)
        Spacer(Modifier.height(8.dp))
        Text("결제 없이 나가는 손님을 카메라가 알아채면 바로 알려 드려요.", style = HawkType.body, color = c.sub)
        Spacer(Modifier.height(30.dp))
        Column(verticalArrangement = Arrangement.spacedBy(18.dp)) {
            Step(1, "Tailscale 켜기", "렌탈 서버에 닿으려면 필요해요")
            Step(2, "서버 주소와 API 키 넣기", "팀에서 받은 값을 넣어요")
            Step(3, "알림 허용하기", "앱이 꺼져 있어도 경보를 받아요")
        }
        Spacer(Modifier.weight(1f))
        HawkButton("서버 연결하기", onClick = onStart, kind = ButtonKind.BRAND, modifier = Modifier.fillMaxWidth())
    }
}

@Composable
private fun Step(n: Int, title: String, sub: String) {
    val c = Hawk.colors
    Row(verticalAlignment = Alignment.Top) {
        Box(Modifier.size(26.dp).background(c.brandTint, CircleShape), contentAlignment = Alignment.Center) {
            Text("$n", style = HawkType.label, color = c.brandInk)
        }
        Spacer(Modifier.width(14.dp))
        Column {
            Text(title, style = HawkType.listTitle, color = c.text)
            Text(sub, style = HawkType.caption, color = c.faint)
        }
    }
}
