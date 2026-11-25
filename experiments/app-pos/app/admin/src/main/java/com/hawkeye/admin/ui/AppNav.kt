package com.hawkeye.admin.ui

import android.Manifest
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.systemBarsPadding
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.History
import androidx.compose.material.icons.outlined.Notifications
import androidx.compose.material.icons.outlined.Settings
import androidx.compose.material.icons.outlined.Videocam
import androidx.compose.material.icons.rounded.History
import androidx.compose.material.icons.rounded.Notifications
import androidx.compose.material.icons.rounded.Settings
import androidx.compose.material.icons.rounded.Videocam
import androidx.compose.material3.Icon
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import androidx.navigation.NavType
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import androidx.navigation.navArgument
import com.hawkeye.admin.AppGraph
import com.hawkeye.admin.domain.CaseFilter
import com.hawkeye.admin.domain.StatusSummaries
import com.hawkeye.admin.ui.theme.Hawk
import com.hawkeye.admin.ui.theme.HawkType
import kotlinx.coroutines.flow.MutableStateFlow

private const val TAB_ALERTS = 0
private const val TAB_LIVE = 1
private const val TAB_RECORDINGS = 2
private const val TAB_SETTINGS = 3

/**
 * 화면 이동. 서버 설정이 없으면 시작 안내, 있으면 아래 탭 4개(경보·라이브·지난 영상·설정).
 * 사건 상세와 '경보 1분 전부터 지난 영상'은 탭 위로 쌓인다.
 */
@Composable
fun AppNav(openCase: MutableStateFlow<Int?>, openLive: MutableStateFlow<String?>) {
    val configured by AppGraph.settings.configured.collectAsState()
    if (!configured) {
        WelcomeFlow()
        return
    }

    val nav = rememberNavController()
    var tab by rememberSaveable { mutableIntStateOf(TAB_ALERTS) }
    var liveCamera by rememberSaveable { mutableStateOf("") }
    var liveRequest by rememberSaveable { mutableIntStateOf(0) }   // 같은 카메라를 다시 요청해도 바꾸게
    val pendingCase by openCase.collectAsState()
    val pendingLive by openLive.collectAsState()

    fun goLive(camera: String) {
        liveCamera = camera
        liveRequest++
        tab = TAB_LIVE
        nav.popBackStack("main", inclusive = false)
    }

    LaunchedEffect(pendingCase) {
        pendingCase?.let {
            nav.navigate("case/$it") { launchSingleTop = true }
            openCase.value = null
        }
    }
    LaunchedEffect(pendingLive) {
        pendingLive?.let {
            goLive(it)
            openLive.value = null
        }
    }

    NavHost(navController = nav, startDestination = "main") {
        composable("main") {
            MainTabs(
                tab = tab,
                onTab = { tab = it },
                liveCamera = liveCamera,
                liveRequest = liveRequest,
                onOpenCase = { nav.navigate("case/$it") },
                onLive = { goLive(it) },
                onOpenFolder = { nav.navigate("library/$it") },
            )
        }
        composable("library/{folderId}", arguments = listOf(navArgument("folderId") { type = NavType.IntType })) {
            val folderId = it.arguments!!.getInt("folderId")
            Box(Modifier.fillMaxSize().background(Hawk.colors.bg).systemBarsPadding()) {
                LibraryFolderScreen(folderId = folderId, onBack = { nav.popBackStack() },
                    onPlay = { scene -> nav.navigate("library/$folderId/$scene") })
            }
        }
        composable(
            "library/{folderId}/{sceneId}",
            arguments = listOf(navArgument("folderId") { type = NavType.IntType }, navArgument("sceneId") { type = NavType.IntType }),
        ) {
            val folderId = it.arguments!!.getInt("folderId")
            Box(Modifier.fillMaxSize().background(Hawk.colors.bg).systemBarsPadding()) {
                LibraryScenePlayerScreen(
                    folderId = folderId, sceneId = it.arguments!!.getInt("sceneId"), onBack = { nav.popBackStack() },
                    onOpenScene = { scene ->   // 이전·다음 장면: 뒤로 가면 폴더로 (장면끼리 쌓지 않음)
                        nav.navigate("library/$folderId/$scene") { popUpTo("library/{folderId}/{sceneId}") { inclusive = true } }
                    },
                )
            }
        }
        composable("case/{id}", arguments = listOf(navArgument("id") { type = NavType.IntType })) {
            CaseDetailScreen(
                caseId = it.arguments!!.getInt("id"),
                onBack = { nav.popBackStack() },
                onRecordings = { camera, start -> nav.navigate("recordings?camera=${Uri.encode(camera)}&start=${Uri.encode(start)}") },
            )
        }
        composable(
            "recordings?camera={camera}&start={start}",
            arguments = listOf(
                navArgument("camera") { defaultValue = "" },
                navArgument("start") { type = NavType.StringType; nullable = true; defaultValue = null },
            ),
        ) {
            Box(Modifier.fillMaxSize().background(Hawk.colors.bg).systemBarsPadding()) {
                RecordingsScreen(
                    cameraId = it.arguments?.getString("camera").orEmpty(),
                    startIso = it.arguments?.getString("start"),
                    onBack = { nav.popBackStack() },
                    onOpenCase = { id -> nav.navigate("case/$id") },
                )
            }
        }
    }
}

@Composable
private fun MainTabs(tab: Int, onTab: (Int) -> Unit, liveCamera: String, liveRequest: Int, onOpenCase: (Int) -> Unit, onLive: (String) -> Unit,
                     onOpenFolder: (Int) -> Unit) {
    val repo = AppGraph.alerts
    val cases by remember { repo.cases(CaseFilter.ALL) }.collectAsState(initial = emptyList())
    val sync by repo.sync.collectAsState()
    val badge = StatusSummaries.badgeText(StatusSummaries.activeCount(cases), sync)
    NotificationPermissionOnce()

    Scaffold(
        containerColor = Hawk.colors.bg,
        bottomBar = { BottomBar(tab, onTab, badge) },
    ) { pad ->
        Box(Modifier.padding(pad)) {
            when (tab) {
                TAB_ALERTS -> CaseListScreen(onOpen = onOpenCase, onLive = { onLive("") })
                TAB_LIVE -> LiveScreen(cameraId = liveCamera, request = liveRequest, onOpenCase = onOpenCase)
                TAB_RECORDINGS -> PastVideosTab(onOpenCase = onOpenCase, onOpenFolder = onOpenFolder)
                else -> SettingsScreen(onBack = null)
            }
        }
    }
}

private data class TabItem(val label: String, val icon: ImageVector, val selectedIcon: ImageVector)

private val TABS = listOf(
    TabItem("경보", Icons.Outlined.Notifications, Icons.Rounded.Notifications),
    TabItem("라이브", Icons.Outlined.Videocam, Icons.Rounded.Videocam),
    TabItem("지난 영상", Icons.Outlined.History, Icons.Rounded.History),
    TabItem("설정", Icons.Outlined.Settings, Icons.Rounded.Settings),
)

@Composable
private fun BottomBar(selected: Int, onSelect: (Int) -> Unit, alertBadge: String?) {
    val c = Hawk.colors
    Column(Modifier.fillMaxWidth().background(c.surf)) {
        Box(Modifier.fillMaxWidth().height(1.dp).background(c.line))
        Row(Modifier.fillMaxWidth().navigationBarsPadding().padding(top = 8.dp, bottom = 8.dp)) {
            TABS.forEachIndexed { i, t ->
                val on = i == selected
                Column(
                    Modifier.weight(1f).clickable { onSelect(i) }.padding(vertical = 4.dp),
                    horizontalAlignment = Alignment.CenterHorizontally,
                ) {
                    Box {
                        Icon(if (on) t.selectedIcon else t.icon, contentDescription = t.label,
                            tint = if (on) c.text else c.faint, modifier = Modifier.size(24.dp))
                        if (i == TAB_ALERTS && alertBadge != null) {
                            Text(
                                alertBadge, style = HawkType.tiny, color = Color.White,
                                modifier = Modifier.align(Alignment.TopEnd).offset(x = 12.dp, y = (-4).dp)
                                    .background(c.danger, CircleShape).padding(horizontal = 5.dp, vertical = 1.dp),
                            )
                        }
                    }
                    Spacer(Modifier.height(4.dp))
                    Text(t.label, style = if (on) HawkType.tiny.copy(fontWeight = HawkType.label.fontWeight) else HawkType.tiny,
                        color = if (on) c.text else c.faint)
                }
            }
        }
    }
}

/**
 * 알림 권한 (안드로이드 13+): 시작 안내와 서버 설정을 마친 뒤 경보 화면에서 한 번만 묻는다.
 * 거절했으면 다시 묻지 않고 설정 화면 '알림' 줄에서 안드로이드 설정으로 안내한다.
 */
@Composable
private fun NotificationPermissionOnce() {
    if (Build.VERSION.SDK_INT < 33) return
    val context = LocalContext.current
    val launcher = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { }
    LaunchedEffect(Unit) {
        val settings = AppGraph.settings
        val granted = ContextCompat.checkSelfPermission(context, Manifest.permission.POST_NOTIFICATIONS) == PackageManager.PERMISSION_GRANTED
        if (!granted && !settings.notificationAsked) {
            settings.notificationAsked = true
            launcher.launch(Manifest.permission.POST_NOTIFICATIONS)
        }
    }
}

/** 서버 설정이 없을 때: 시작 안내 → 서버 연결(설정 화면) → 저장하면 경보 화면으로. */
@Composable
private fun WelcomeFlow() {
    var setup by rememberSaveable { mutableStateOf(false) }
    if (setup) SettingsScreen(onBack = { setup = false }, firstRun = true)
    else WelcomeScreen(onStart = { setup = true })
}
