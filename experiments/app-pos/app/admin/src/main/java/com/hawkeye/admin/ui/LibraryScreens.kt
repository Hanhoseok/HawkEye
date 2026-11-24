package com.hawkeye.admin.ui

import androidx.activity.compose.BackHandler
import androidx.compose.foundation.ExperimentalFoundationApi
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.combinedClickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.outlined.DriveFileMove
import androidx.compose.material.icons.automirrored.rounded.KeyboardArrowLeft
import androidx.compose.material.icons.automirrored.rounded.KeyboardArrowRight
import androidx.compose.material.icons.outlined.CreateNewFolder
import androidx.compose.material.icons.outlined.Delete
import androidx.compose.material.icons.outlined.Edit
import androidx.compose.material.icons.rounded.CheckCircle
import androidx.compose.material.icons.rounded.Close
import androidx.compose.material.icons.rounded.ErrorOutline
import androidx.compose.material.icons.rounded.Folder
import androidx.compose.material.icons.rounded.Info
import androidx.compose.material.icons.rounded.MoreVert
import androidx.compose.material.icons.rounded.RadioButtonUnchecked
import androidx.compose.material.icons.rounded.Refresh
import androidx.compose.material.icons.rounded.VideoLibrary
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.rememberModalBottomSheetState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.key
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableLongStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import com.hawkeye.admin.AppGraph
import com.hawkeye.admin.data.FolderDetail
import com.hawkeye.admin.data.LibraryFolder
import com.hawkeye.admin.data.LibraryScene
import com.hawkeye.admin.data.LibraryVerdict
import com.hawkeye.admin.domain.CameraSwitch
import com.hawkeye.admin.domain.LibraryErrors
import com.hawkeye.admin.domain.LibraryFormat
import com.hawkeye.admin.domain.LibraryNames
import com.hawkeye.admin.domain.SceneNav
import com.hawkeye.admin.domain.VerdictNav
import com.hawkeye.admin.domain.Selection
import com.hawkeye.admin.ui.components.AppBar
import com.hawkeye.admin.ui.components.ButtonKind
import com.hawkeye.admin.ui.components.EmptyState
import com.hawkeye.admin.ui.components.Group
import com.hawkeye.admin.ui.components.HawkButton
import com.hawkeye.admin.ui.components.LevelChip
import com.hawkeye.admin.ui.components.Note
import com.hawkeye.admin.ui.components.RowDivider
import com.hawkeye.admin.ui.components.ScreenTitle
import com.hawkeye.admin.ui.components.SectionLabel
import com.hawkeye.admin.ui.components.Segmented
import com.hawkeye.admin.ui.components.SwitchRow
import com.hawkeye.admin.ui.components.TextTabs
import com.hawkeye.admin.ui.components.Veil
import com.hawkeye.admin.ui.components.VideoTag
import com.hawkeye.admin.ui.components.groupItem
import com.hawkeye.admin.ui.theme.Hawk
import com.hawkeye.admin.ui.theme.HawkType
import kotlinx.coroutines.launch

/** 지난 영상 탭: 제목 아래 '녹화 | 보관함'. 고른 쪽은 폴더·장면을 보고 돌아와도 남는다. */
@Composable
fun PastVideosTab(onOpenCase: (Int) -> Unit, onOpenFolder: (Int) -> Unit) {
    var mode by rememberSaveable { mutableIntStateOf(0) }
    val tabs: @Composable () -> Unit = { TextTabs(listOf("녹화", "보관함"), mode, { mode = it }) }
    if (mode == 0) RecordingsScreen(cameraId = "", startIso = null, onBack = null, onOpenCase = onOpenCase, tabs = tabs)
    else LibraryHome(tabs = tabs, onOpenFolder = onOpenFolder)
}

// ---------------------------------------------------------------------------------------------
// 보관함: 폴더 목록

@OptIn(ExperimentalFoundationApi::class)
@Composable
private fun LibraryHome(tabs: @Composable () -> Unit, onOpenFolder: (Int) -> Unit) {
    val repo = AppGraph.library
    val version by repo.version.collectAsState()
    val scope = rememberCoroutineScope()
    var folders by remember { mutableStateOf<List<LibraryFolder>?>(null) }
    var message by remember { mutableStateOf<String?>(null) }
    var reload by remember { mutableIntStateOf(0) }
    var creating by remember { mutableStateOf(false) }
    var menuFor by remember { mutableStateOf<LibraryFolder?>(null) }
    var renaming by remember { mutableStateOf<LibraryFolder?>(null) }
    var deleting by remember { mutableStateOf<LibraryFolder?>(null) }
    val c = Hawk.colors

    LaunchedEffect(version, reload) {
        runCatching { repo.folders() }
            .onSuccess { folders = it; message = null }
            .onFailure { message = LibraryErrors.message(it) }
    }

    Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState())) {
        ScreenTitle("지난 영상") {
            IconButton(onClick = { creating = true }) { Icon(Icons.Outlined.CreateNewFolder, "새 폴더", tint = c.sub) }
            IconButton(onClick = { reload++ }) { Icon(Icons.Rounded.Refresh, "새로고침", tint = c.sub) }
        }
        tabs()
        Column(Modifier.padding(horizontal = 16.dp).padding(bottom = 20.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
            message?.let { Note(Icons.Rounded.Info, c.faint, c.surf, text = it) }
            val list = folders
            when {
                list == null && message == null -> Text("불러오는 중…", style = HawkType.caption, color = c.faint, modifier = Modifier.padding(4.dp))
                list != null && list.isEmpty() -> EmptyState(Icons.Rounded.VideoLibrary, "보관함이 비어 있어요",
                    "서버에서 촬영 영상을 넣으면 회차별 폴더로 보여요. 오른쪽 위에서 새 폴더를 만들 수도 있어요.")
                list != null -> {
                    Group {
                        list.forEachIndexed { i, f ->
                            if (i > 0) RowDivider(start = 70.dp)
                            Row(
                                Modifier.fillMaxWidth().combinedClickable(onClick = { onOpenFolder(f.id) }, onLongClick = { menuFor = f })
                                    .padding(start = 16.dp, end = 4.dp, top = 12.dp, bottom = 12.dp),
                                verticalAlignment = Alignment.CenterVertically,
                            ) {
                                Box(Modifier.size(40.dp).clip(RoundedCornerShape(12.dp)).background(c.brandTint), contentAlignment = Alignment.Center) {
                                    Icon(Icons.Rounded.Folder, null, tint = c.brand, modifier = Modifier.size(22.dp))
                                }
                                Spacer(Modifier.width(14.dp))
                                Column(Modifier.weight(1f)) {
                                    Text(f.name, style = HawkType.listTitle, color = c.text, maxLines = 1, overflow = TextOverflow.Ellipsis)
                                    Text(folderLine(f), style = HawkType.caption, color = c.faint)
                                }
                                IconButton(onClick = { menuFor = f }) { Icon(Icons.Rounded.MoreVert, "폴더 메뉴", tint = c.faint) }
                            }
                        }
                    }
                }
            }
        }
    }

    menuFor?.let { f ->
        ActionSheet(f.name, onDismiss = { menuFor = null }, actions = listOf(
            SheetAction(Icons.Outlined.Edit, "이름 바꾸기") { renaming = f },
            SheetAction(Icons.Outlined.Delete, "폴더 지우기", danger = true) { deleting = f },
        ))
    }
    if (creating) {
        NameDialog("새 폴더", initial = "", confirm = "만들기", onDismiss = { creating = false }) { name ->
            runCatching { repo.createFolder(name) }.fold({ null }, { LibraryErrors.message(it) })
        }
    }
    renaming?.let { f ->
        NameDialog("폴더 이름 바꾸기", initial = f.name, confirm = "바꾸기", onDismiss = { renaming = null }) { name ->
            runCatching { repo.renameFolder(f.id, name) }.fold({ null }, { LibraryErrors.message(it) })
        }
    }
    deleting?.let { f ->
        ConfirmDialog(
            title = if (f.sceneCount > 0) "'${f.name}' 폴더와 장면 ${f.sceneCount}개를 지울까요?" else "'${f.name}' 폴더를 지울까요?",
            text = if (f.sceneCount > 0) "서버의 영상 파일도 지워져 되돌릴 수 없어요." else "빈 폴더예요.",
            confirm = "지우기",
            onDismiss = { deleting = null },
        ) {
            scope.launch {
                runCatching { repo.deleteFolder(f.id, withScenes = f.sceneCount > 0) }
                    .onFailure { message = LibraryErrors.message(it); reload++ }
            }
        }
    }
}

private fun folderLine(f: LibraryFolder): String =
    if (f.sceneCount == 0) "비어 있음" else "${f.sceneCount}장면 · ${LibraryFormat.total(f.totalSec)}"

// ---------------------------------------------------------------------------------------------
// 폴더 안: 장면 목록 · 고르기(옮기기 · 이름 바꾸기 · 지우기)

@OptIn(ExperimentalFoundationApi::class)
@Composable
fun LibraryFolderScreen(folderId: Int, onBack: () -> Unit, onPlay: (Int) -> Unit) {
    val repo = AppGraph.library
    val version by repo.version.collectAsState()
    val scope = rememberCoroutineScope()
    var detail by remember { mutableStateOf<FolderDetail?>(null) }
    var gone by remember { mutableStateOf(false) }
    var message by remember { mutableStateOf<String?>(null) }
    var reload by remember { mutableIntStateOf(0) }
    var selected by remember { mutableStateOf<Set<Int>>(emptySet()) }
    var folderMenu by remember { mutableStateOf(false) }
    var renamingFolder by remember { mutableStateOf(false) }
    var deletingFolder by remember { mutableStateOf(false) }
    var moving by remember { mutableStateOf(false) }
    var renamingScene by remember { mutableStateOf<LibraryScene?>(null) }
    var deletingScenes by remember { mutableStateOf(false) }
    val c = Hawk.colors

    LaunchedEffect(folderId, version, reload) {
        runCatching { repo.folder(folderId) }
            .onSuccess { d -> detail = d; message = null; selected = Selection.keepExisting(selected, d.scenes.map { it.id }) }
            .onFailure { if (LibraryErrors.isGone(it)) gone = true else message = LibraryErrors.message(it) }
    }
    val selecting = selected.isNotEmpty()
    BackHandler(enabled = selecting) { selected = emptySet() }

    val d = detail
    val scenes = d?.scenes.orEmpty()
    Column(Modifier.fillMaxSize()) {
        if (selecting) {
            Row(Modifier.fillMaxWidth().padding(start = 4.dp, end = 12.dp, top = 4.dp, bottom = 4.dp), verticalAlignment = Alignment.CenterVertically) {
                IconButton(onClick = { selected = emptySet() }) { Icon(Icons.Rounded.Close, "고르기 끝내기", tint = c.text) }
                Text("${selected.size}개 고름", style = HawkType.appBar, color = c.text, modifier = Modifier.weight(1f))
                TextButton(onClick = { selected = Selection.all(selected, scenes.map { it.id }) }) {
                    Text(if (selected.size == scenes.size) "모두 풀기" else "모두 고르기", style = HawkType.label, color = c.brandInk)
                }
            }
        } else {
            AppBar(d?.folder?.name ?: "보관함", onBack) {
                if (d != null) IconButton(onClick = { folderMenu = true }) { Icon(Icons.Rounded.MoreVert, "폴더 메뉴", tint = c.sub) }
            }
        }

        LazyColumn(Modifier.weight(1f), contentPadding = PaddingValues(start = 16.dp, end = 16.dp, bottom = 20.dp)) {
            item {
                Column(Modifier.padding(bottom = 12.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                    if (gone) Note(Icons.Rounded.Info, c.faint, c.surf, text = "이 폴더는 지워졌어요.", actionLabel = "돌아가기", onAction = onBack)
                    message?.let { Note(Icons.Rounded.Info, c.faint, c.surf, text = it) }
                    if (d != null && scenes.isNotEmpty()) {
                        Text("${scenes.size}장면 · ${LibraryFormat.total(d.folder.totalSec)} · 길게 누르면 여러 개를 골라 옮기거나 지울 수 있어요",
                            style = HawkType.caption, color = c.faint, modifier = Modifier.padding(horizontal = 4.dp))
                    }
                }
            }
            if (d != null && scenes.isEmpty()) {
                item { EmptyState(Icons.Rounded.VideoLibrary, "빈 폴더예요", "다른 폴더에서 장면을 길게 눌러 이 폴더로 옮길 수 있어요.") }
            }
            itemsIndexed(scenes, key = { _, s -> s.id }) { i, s ->
                val on = s.id in selected
                Column(Modifier.groupItem(i, scenes.size)) {
                    if (i > 0) RowDivider(start = if (selecting) 122.dp else 98.dp)
                    Row(
                        Modifier.fillMaxWidth().combinedClickable(
                            onClick = { if (selecting) selected = Selection.toggle(selected, s.id) else onPlay(s.id) },
                            onLongClick = { selected = Selection.toggle(selected, s.id) },
                        ).padding(horizontal = 14.dp, vertical = 10.dp),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        if (selecting) {
                            Icon(if (on) Icons.Rounded.CheckCircle else Icons.Rounded.RadioButtonUnchecked, null,
                                tint = if (on) c.brand else c.fill2, modifier = Modifier.size(22.dp))
                            Spacer(Modifier.width(10.dp))
                        }
                        SceneThumb(s, Modifier.width(72.dp))
                        Spacer(Modifier.width(12.dp))
                        Column(Modifier.weight(1f)) {
                            Text(s.name, style = HawkType.listTitle, color = c.text, maxLines = 1, overflow = TextOverflow.Ellipsis)
                            Text(LibraryFormat.sceneLine(s, d?.folder?.name.orEmpty()), style = HawkType.caption, color = c.faint,
                                maxLines = 1, overflow = TextOverflow.Ellipsis)
                        }
                    }
                }
            }
        }

        if (selecting) {
            Column(Modifier.fillMaxWidth().background(c.surf)) {
                Box(Modifier.fillMaxWidth().height(1.dp).background(c.line))
                Row(Modifier.fillMaxWidth().navigationBarsPadding().padding(vertical = 6.dp)) {
                    DockAction(Icons.AutoMirrored.Outlined.DriveFileMove, "옮기기", c.text) { moving = true }
                    DockAction(Icons.Outlined.Edit, "이름 바꾸기", c.text, enabled = selected.size == 1) {
                        renamingScene = scenes.firstOrNull { it.id in selected }
                    }
                    DockAction(Icons.Outlined.Delete, "지우기", c.danger) { deletingScenes = true }
                }
            }
        }
    }

    // ---- 시트 · 대화상자 ----
    if (folderMenu && d != null) {
        ActionSheet(d.folder.name, onDismiss = { folderMenu = false }, actions = listOf(
            SheetAction(Icons.Outlined.Edit, "폴더 이름 바꾸기") { renamingFolder = true },
            SheetAction(Icons.Outlined.Delete, "폴더 지우기", danger = true) { deletingFolder = true },
        ))
    }
    if (renamingFolder && d != null) {
        NameDialog("폴더 이름 바꾸기", initial = d.folder.name, confirm = "바꾸기", onDismiss = { renamingFolder = false }) { name ->
            runCatching { repo.renameFolder(folderId, name) }.fold({ null }, { LibraryErrors.message(it) })
        }
    }
    if (deletingFolder && d != null) {
        val n = scenes.size
        ConfirmDialog(
            title = if (n > 0) "'${d.folder.name}' 폴더와 장면 ${n}개를 지울까요?" else "'${d.folder.name}' 폴더를 지울까요?",
            text = if (n > 0) "서버의 영상 파일도 지워져 되돌릴 수 없어요." else "빈 폴더예요.",
            confirm = "지우기",
            onDismiss = { deletingFolder = false },
        ) {
            scope.launch {
                runCatching { repo.deleteFolder(folderId, withScenes = n > 0) }
                    .onSuccess { onBack() }
                    .onFailure { message = LibraryErrors.message(it); reload++ }
            }
        }
    }
    renamingScene?.let { s ->
        NameDialog("장면 이름 바꾸기", initial = s.name, confirm = "바꾸기", onDismiss = { renamingScene = null }) { name ->
            runCatching { repo.renameScene(s.id, name) }.fold({ selected = emptySet(); null }, { LibraryErrors.message(it) })
        }
    }
    if (deletingScenes) {
        ConfirmDialog(
            title = "장면 ${selected.size}개를 지울까요?",
            text = "서버의 영상 파일도 지워져 되돌릴 수 없어요.",
            confirm = "지우기",
            onDismiss = { deletingScenes = false },
        ) {
            val ids = selected.toList()
            scope.launch {
                runCatching { repo.deleteScenes(ids) }
                    .onSuccess { selected = emptySet() }
                    .onFailure { message = LibraryErrors.message(it); reload++ }
            }
        }
    }
    if (moving) {
        MoveSheet(currentFolderId = folderId, count = selected.size, onDismiss = { moving = false }) { target ->
            val ids = selected.toList()
            runCatching { repo.moveScenes(ids, target) }
                .fold({ selected = emptySet(); null }, { LibraryErrors.message(it) })
        }
    }
}

@Composable
private fun androidx.compose.foundation.layout.RowScope.DockAction(
    icon: ImageVector, label: String, tint: Color, enabled: Boolean = true, onClick: () -> Unit,
) {
    val color = if (enabled) tint else Hawk.colors.fill2
    Column(Modifier.weight(1f).clickable(enabled = enabled, onClick = onClick).padding(vertical = 8.dp),
        horizontalAlignment = Alignment.CenterHorizontally) {
        Icon(icon, null, tint = color, modifier = Modifier.size(22.dp))
        Spacer(Modifier.height(4.dp))
        Text(label, style = HawkType.tiny, color = color)
    }
}

/** 장면 썸네일 (16:9). 없으면 어두운 자리만. */
@Composable
private fun SceneThumb(scene: LibraryScene, modifier: Modifier = Modifier) {
    val repo = AppGraph.library
    var image by remember(scene.id, scene.mediaAt) { mutableStateOf<ImageBitmap?>(repo.cachedThumb(scene)) }
    LaunchedEffect(scene.id, scene.mediaAt) {
        if (image == null) image = repo.thumb(scene)
    }
    Box(modifier.aspectRatio(16f / 9f).clip(RoundedCornerShape(8.dp)).background(Color(0xFF1B2129))) {
        image?.let { Image(it, contentDescription = null, contentScale = ContentScale.Crop, modifier = Modifier.fillMaxSize()) }
    }
}

// ---------------------------------------------------------------------------------------------
// 장면 재생: 카메라 바꾸기(같은 순간 이어 보기) · 이전/다음 장면

@Composable
fun LibraryScenePlayerScreen(folderId: Int, sceneId: Int, onBack: () -> Unit, onOpenScene: (Int) -> Unit) {
    val repo = AppGraph.library
    val api = remember { repo.api() }
    val version by repo.version.collectAsState()
    var detail by remember { mutableStateOf<FolderDetail?>(null) }
    var message by remember { mutableStateOf<String?>(null) }
    var camera by rememberSaveable(sceneId) { mutableStateOf("") }
    var startMs by rememberSaveable(sceneId) { mutableLongStateOf(0L) }
    var position by remember(sceneId) { mutableLongStateOf(0L) }
    var note by remember(sceneId) { mutableStateOf<String?>(null) }
    var failed by remember(sceneId) { mutableStateOf<String?>(null) }
    var retry by remember(sceneId) { mutableIntStateOf(0) }
    var jump by remember(sceneId) { mutableIntStateOf(0) }       // 판정을 눌러 그 위치로 다시 열 때
    val aiOn by repo.aiView.collectAsState()
    val c = Hawk.colors

    LaunchedEffect(folderId, version) {
        runCatching { repo.folder(folderId) }
            .onSuccess { detail = it; message = null }
            .onFailure { message = LibraryErrors.message(it) }
    }
    val d = detail
    val scene = d?.scenes?.firstOrNull { it.id == sceneId }
    val videos = scene?.videos.orEmpty()
    LaunchedEffect(scene?.id, videos.map { it.cameraId }) {
        if (videos.isNotEmpty() && videos.none { it.cameraId == camera }) {
            camera = (videos.firstOrNull { it.cameraId == "cam1" } ?: videos.first()).cameraId
        }
    }
    val video = videos.firstOrNull { it.cameraId == camera }
    val showAi = aiOn && video?.hasAi == true

    Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState())) {
        AppBar(scene?.name ?: "장면", onBack)
        Column(Modifier.padding(horizontal = 16.dp).padding(bottom = 20.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
            message?.let { Note(Icons.Rounded.Info, c.faint, c.surf, text = it) }
            if (d != null && scene == null) {
                EmptyState(Icons.Rounded.VideoLibrary, "장면이 없어요", "다른 폴더로 옮겨졌거나 지워졌어요.")
                return@Column
            }
            VideoFrame {
                when {
                    video == null && scene != null && videos.isEmpty() ->
                        Veil(Icons.Rounded.ErrorOutline, "이 장면에는 영상이 없어요", "서버에서 다시 가져와 주세요.")
                    video == null -> Veil(Icons.Rounded.VideoLibrary, "영상을 불러오는 중", null)
                    failed != null -> Box(Modifier.fillMaxSize().clickable { failed = null; retry++ }) {
                        Veil(Icons.Rounded.ErrorOutline, "영상을 열지 못했어요", "${failed} · 눌러서 다시 시도")
                    }
                    else -> key(video.id, showAi, retry, jump) {
                        HttpVideoPlayer(
                            url = if (showAi) api.libraryAiVideoUrl(video.id) else api.libraryVideoUrl(video.id),
                            headers = api.authHeaders(), modifier = Modifier.fillMaxSize(),
                            startMs = startMs, onPosition = { position = it },
                            onError = { e -> failed = if (e.errorCode == androidx.media3.common.PlaybackException.ERROR_CODE_IO_BAD_HTTP_STATUS) "영상 파일이 없거나 키가 맞지 않아요" else "재생 오류 (${e.errorCodeName})" },
                        )
                    }
                }
                if (video != null && (videos.size > 1 || showAi)) {
                    val tag = (if (videos.size > 1) LibraryFormat.shortCamera(video.cameraId) + " 카메라" else "") +
                        (if (showAi) (if (videos.size > 1) " · AI" else "AI 탐지") else "")
                    VideoTag(tag, Modifier.align(Alignment.TopStart).padding(10.dp))
                }
            }
            if (videos.size > 1) {
                Segmented(
                    options = videos.map { LibraryFormat.shortCamera(it.cameraId) },
                    selected = videos.indexOfFirst { it.cameraId == camera }.coerceAtLeast(0),
                    onSelect = { i ->
                        val to = videos[i]
                        val from = video
                        if (from != null && to.cameraId != from.cameraId) {
                            val r = CameraSwitch.position(position, from.offsetMs, to.offsetMs, to.duration?.let { (it * 1000).toLong() })
                            startMs = r.positionMs
                            position = r.positionMs
                            note = when (r.clip) {
                                CameraSwitch.Clip.BEFORE_START -> "${LibraryFormat.shortCamera(to.cameraId)} 카메라는 이 순간 뒤부터 찍혔어요. 처음부터 보여 줘요."
                                CameraSwitch.Clip.AFTER_END -> "${LibraryFormat.shortCamera(to.cameraId)} 카메라는 이 순간 전에 끝났어요. 끝부분을 보여 줘요."
                                CameraSwitch.Clip.NONE -> null
                            }
                            failed = null
                            camera = to.cameraId
                        }
                    },
                )
            }
            if (videos.any { it.hasAi }) {
                SwitchRow("AI 탐지 화면", "사람 상자·번호 (저장된 추적 기록)", checked = aiOn, onChange = { on ->
                    startMs = position          // 같은 시간축 — 보던 순간에서 이어 봄
                    failed = null
                    repo.setAiView(on)
                })
                if (aiOn && video != null && !video.hasAi) {
                    Note(Icons.Rounded.Info, c.faint, c.surf, text = "${LibraryFormat.shortCamera(video.cameraId)} 카메라는 AI 화면이 없어요. 원본을 보여 줘요.")
                }
            }
            note?.let { Note(Icons.Rounded.Info, c.faint, c.surf, text = it) }
            if (scene != null && scene.verdicts.isNotEmpty()) {
                val offset = video?.offsetMs ?: 0.0
                SectionLabel("AI 판정", trailing = "누르면 3초 전부터")
                Group {
                    scene.verdicts.forEachIndexed { i, v ->
                        if (i > 0) RowDivider(start = 70.dp)
                        VerdictRow(v, passed = VerdictNav.passed(v.ms, position, offset)) {
                            val to = VerdictNav.seekMs(v.ms, offset, video?.duration?.let { (it * 1000).toLong() })
                            startMs = to
                            position = to
                            failed = null
                            jump++
                        }
                    }
                }
                scene.verdictSource?.let {
                    Text(it, style = HawkType.caption, color = c.faint, modifier = Modifier.padding(horizontal = 4.dp))
                }
            }
            if (scene != null) {
                Row(Modifier.fillMaxWidth().clip(RoundedCornerShape(20.dp)).background(c.surf).padding(horizontal = 18.dp, vertical = 14.dp)) {
                    InfoCell("길이", LibraryFormat.clipLength(scene.duration), Modifier.weight(1f))
                    InfoCell("회차", scene.origin.ifBlank { d?.folder?.name.orEmpty() }, Modifier.weight(1.4f))
                    InfoCell("카메라", if (videos.size > 1) "2대" else "1대", Modifier.weight(1f))
                }
                val (prev, next) = SceneNav.neighbors(d?.scenes.orEmpty(), scene.id)
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    HawkButton(prev?.let { "이전 · ${it.name}" } ?: "이전 장면", onClick = { prev?.let { onOpenScene(it.id) } },
                        kind = ButtonKind.WHITE, icon = Icons.AutoMirrored.Rounded.KeyboardArrowLeft, enabled = prev != null, modifier = Modifier.weight(1f))
                    HawkButton(next?.let { "다음 · ${it.name}" } ?: "다음 장면", onClick = { next?.let { onOpenScene(it.id) } },
                        kind = ButtonKind.WHITE, icon = Icons.AutoMirrored.Rounded.KeyboardArrowRight, enabled = next != null, modifier = Modifier.weight(1f), iconEnd = true)
                }
            }
        }
    }
}

/** 판정 한 줄: 시각 · 단계 · 손님 · 이유 · 개수. 아직 안 온 판정은 옅게. */
@Composable
private fun VerdictRow(v: LibraryVerdict, passed: Boolean, onClick: () -> Unit) {
    val c = Hawk.colors
    Row(
        Modifier.fillMaxWidth().clickable(onClick = onClick).alpha(if (passed) 1f else 0.5f)
            .padding(start = 16.dp, end = 16.dp, top = 12.dp, bottom = 12.dp),
        verticalAlignment = Alignment.Top,
    ) {
        Text(LibraryFormat.verdictTime(v.ms), style = HawkType.label.copy(fontFeatureSettings = "tnum"), color = c.text,
            modifier = Modifier.width(42.dp).padding(top = 5.dp))
        Spacer(Modifier.width(12.dp))
        Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(4.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                LevelChip(v.level)
                Spacer(Modifier.width(8.dp))
                Text("손님 ${v.person}", style = HawkType.listTitle, color = c.text)
            }
            val reason = LibraryFormat.verdictReason(v.reason)
            if (reason.isNotEmpty()) Text(reason, style = HawkType.caption, color = c.sub)
            LibraryFormat.verdictCounts(v).takeIf { it.isNotEmpty() }?.let { Text(it, style = HawkType.caption, color = c.faint) }
        }
    }
}

@Composable
private fun InfoCell(label: String, value: String, modifier: Modifier = Modifier) {
    Column(modifier) {
        Text(label, style = HawkType.caption, color = Hawk.colors.faint)
        Text(value, style = HawkType.listTitle, color = Hawk.colors.text, maxLines = 1, overflow = TextOverflow.Ellipsis)
    }
}

// ---------------------------------------------------------------------------------------------
// 공용: 아래 시트 · 이름 입력 · 확인 · 옮길 폴더 고르기

private data class SheetAction(val icon: ImageVector, val label: String, val danger: Boolean = false, val onClick: () -> Unit)

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun ActionSheet(title: String, onDismiss: () -> Unit, actions: List<SheetAction>) {
    val c = Hawk.colors
    ModalBottomSheet(onDismissRequest = onDismiss, sheetState = rememberModalBottomSheetState(skipPartiallyExpanded = true),
        containerColor = c.surf, shape = RoundedCornerShape(topStart = 24.dp, topEnd = 24.dp)) {
        Column(Modifier.navigationBarsPadding().padding(bottom = 12.dp)) {
            Text(title, style = HawkType.label, color = c.faint, modifier = Modifier.padding(horizontal = 24.dp, vertical = 6.dp),
                maxLines = 1, overflow = TextOverflow.Ellipsis)
            actions.forEach { a ->
                val tint = if (a.danger) c.danger else c.text
                Row(Modifier.fillMaxWidth().clickable { onDismiss(); a.onClick() }.padding(horizontal = 24.dp, vertical = 15.dp),
                    verticalAlignment = Alignment.CenterVertically) {
                    Icon(a.icon, null, tint = tint, modifier = Modifier.size(22.dp))
                    Spacer(Modifier.width(14.dp))
                    Text(a.label, style = HawkType.body, color = tint)
                }
            }
        }
    }
}

/** 이름 입력. onSubmit 은 성공하면 null, 실패하면 보여 줄 문장을 돌려준다 (성공하면 닫힘). */
@Composable
private fun NameDialog(title: String, initial: String, confirm: String, onDismiss: () -> Unit, onSubmit: suspend (String) -> String?) {
    val c = Hawk.colors
    val scope = rememberCoroutineScope()
    var text by remember { mutableStateOf(initial) }
    var error by remember { mutableStateOf<String?>(null) }
    var busy by remember { mutableStateOf(false) }
    fun submit() {
        LibraryNames.problem(text)?.let { error = it; return }
        if (text.trim() == initial.trim() && initial.isNotBlank()) { onDismiss(); return }
        busy = true
        scope.launch {
            val e = onSubmit(text.trim())
            busy = false
            if (e == null) onDismiss() else error = e
        }
    }
    AlertDialog(
        onDismissRequest = { if (!busy) onDismiss() },
        containerColor = c.surf,
        shape = RoundedCornerShape(20.dp),
        title = { Text(title, style = HawkType.appBar, color = c.text) },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Box(Modifier.fillMaxWidth().clip(RoundedCornerShape(12.dp)).background(c.fill).padding(horizontal = 14.dp, vertical = 12.dp)) {
                    if (text.isEmpty()) Text("이름", style = HawkType.body, color = c.faint)
                    BasicTextField(text, { text = it.take(LibraryNames.MAX + 10); error = null }, singleLine = true,
                        textStyle = HawkType.body.copy(color = c.text), cursorBrush = SolidColor(c.brand), modifier = Modifier.fillMaxWidth())
                }
                Text(error ?: "${text.trim().length}/${LibraryNames.MAX}", style = HawkType.caption, color = if (error != null) c.danger else c.faint)
            }
        },
        confirmButton = {
            TextButton(onClick = { submit() }, enabled = !busy) { Text(confirm, style = HawkType.button, color = c.brandInk) }
        },
        dismissButton = {
            TextButton(onClick = onDismiss, enabled = !busy) { Text("취소", style = HawkType.button, color = c.sub) }
        },
    )
}

@Composable
private fun ConfirmDialog(title: String, text: String, confirm: String, onDismiss: () -> Unit, onConfirm: () -> Unit) {
    val c = Hawk.colors
    AlertDialog(
        onDismissRequest = onDismiss,
        containerColor = c.surf,
        shape = RoundedCornerShape(20.dp),
        title = { Text(title, style = HawkType.appBar, color = c.text) },
        text = { Text(text, style = HawkType.body, color = c.sub) },
        confirmButton = { TextButton(onClick = { onDismiss(); onConfirm() }) { Text(confirm, style = HawkType.button, color = c.danger) } },
        dismissButton = { TextButton(onClick = onDismiss) { Text("취소", style = HawkType.button, color = c.sub) } },
    )
}

/** 옮길 폴더 고르기: 지금 폴더를 뺀 목록 + 새 폴더 만들어 옮기기. onMove 는 성공하면 null. */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun MoveSheet(currentFolderId: Int, count: Int, onDismiss: () -> Unit, onMove: suspend (Int) -> String?) {
    val repo = AppGraph.library
    val c = Hawk.colors
    val scope = rememberCoroutineScope()
    var folders by remember { mutableStateOf<List<LibraryFolder>?>(null) }
    var error by remember { mutableStateOf<String?>(null) }
    var busy by remember { mutableStateOf(false) }
    var creating by remember { mutableStateOf(false) }
    LaunchedEffect(Unit) {
        runCatching { repo.folders() }.onSuccess { folders = it.filter { f -> f.id != currentFolderId } }
            .onFailure { error = LibraryErrors.message(it) }
    }
    fun move(id: Int) {
        if (busy) return
        busy = true
        scope.launch {
            val e = onMove(id)
            busy = false
            if (e == null) onDismiss() else error = e
        }
    }
    ModalBottomSheet(onDismissRequest = { if (!busy) onDismiss() }, sheetState = rememberModalBottomSheetState(skipPartiallyExpanded = true),
        containerColor = c.surf, shape = RoundedCornerShape(topStart = 24.dp, topEnd = 24.dp)) {
        Column(Modifier.navigationBarsPadding().padding(bottom = 12.dp).heightIn(max = 480.dp).verticalScroll(rememberScrollState())) {
            Text("장면 ${count}개를 옮길 폴더", style = HawkType.label, color = c.faint, modifier = Modifier.padding(horizontal = 24.dp, vertical = 6.dp))
            error?.let { Text(it, style = HawkType.caption, color = c.danger, modifier = Modifier.padding(horizontal = 24.dp, vertical = 4.dp)) }
            Row(Modifier.fillMaxWidth().clickable(enabled = !busy) { creating = true }.padding(horizontal = 24.dp, vertical = 14.dp),
                verticalAlignment = Alignment.CenterVertically) {
                Icon(Icons.Outlined.CreateNewFolder, null, tint = c.brand, modifier = Modifier.size(22.dp))
                Spacer(Modifier.width(14.dp))
                Text("새 폴더 만들어 옮기기", style = HawkType.body, color = c.brandInk)
            }
            folders?.forEach { f ->
                Row(Modifier.fillMaxWidth().clickable(enabled = !busy) { move(f.id) }.padding(horizontal = 24.dp, vertical = 14.dp),
                    verticalAlignment = Alignment.CenterVertically) {
                    Icon(Icons.Rounded.Folder, null, tint = c.faint, modifier = Modifier.size(22.dp))
                    Spacer(Modifier.width(14.dp))
                    Text(f.name, style = HawkType.body, color = c.text, modifier = Modifier.weight(1f), maxLines = 1, overflow = TextOverflow.Ellipsis)
                    Text(folderLine(f), style = HawkType.caption, color = c.faint)
                }
            }
        }
    }
    if (creating) {
        NameDialog("새 폴더 만들어 옮기기", initial = "", confirm = "만들고 옮기기", onDismiss = { creating = false }) { name ->
            val made = runCatching { repo.createFolder(name) }
            made.fold({ f -> onMove(f.id).also { if (it == null) onDismiss() } }, { LibraryErrors.message(it) })
        }
    }
}
