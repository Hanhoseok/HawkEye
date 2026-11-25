package com.hawkeye.admin.data

import android.graphics.BitmapFactory
import android.util.LruCache
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.asImageBitmap
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.withContext

/**
 * 보관함 (촬영 영상 폴더). 바꾸는 호출 뒤에는 version 을 올려 열린 화면들이 다시 받게 한다.
 * 썸네일은 메모리에 (키 = 장면 id + 영상을 넣은 시각 — 영상을 바꾸면 새로 받음).
 */
class LibraryRepository(private val apiFactory: () -> ApiClient) {
    private val _version = MutableStateFlow(0)
    val version: StateFlow<Int> = _version.asStateFlow()
    private val thumbs = LruCache<String, ImageBitmap>(300)
    private val _aiView = MutableStateFlow(false)

    /** 보관함 재생에서 AI 화면을 켜 두었나 (앱을 켜 둔 동안 장면을 넘겨도 유지). */
    val aiView: StateFlow<Boolean> = _aiView.asStateFlow()

    fun setAiView(on: Boolean) {
        _aiView.value = on
    }

    fun api(): ApiClient = apiFactory()

    fun changed() {
        _version.value += 1
    }

    suspend fun folders(): List<LibraryFolder> = api().libraryFolders()

    suspend fun folder(id: Int): FolderDetail = api().libraryFolder(id)

    suspend fun createFolder(name: String): LibraryFolder = api().createFolder(name.trim()).also { changed() }

    suspend fun renameFolder(id: Int, name: String): LibraryFolder = api().renameFolder(id, name.trim()).also { changed() }

    suspend fun deleteFolder(id: Int, withScenes: Boolean): Int = api().deleteFolder(id, withScenes).also { changed() }

    suspend fun renameScene(id: Int, name: String): LibraryScene = api().renameScene(id, name.trim()).also { changed() }

    suspend fun moveScenes(ids: Collection<Int>, folderId: Int): Int = api().moveScenes(ids, folderId).also { changed() }

    suspend fun deleteScenes(ids: Collection<Int>): Int = api().deleteScenes(ids).also { changed() }

    fun cachedThumb(scene: LibraryScene): ImageBitmap? = thumbs.get(thumbKey(scene))

    /** 썸네일. 없거나 못 받으면 null (목록은 그대로 보인다). */
    suspend fun thumb(scene: LibraryScene): ImageBitmap? {
        if (!scene.hasThumb) return null
        val key = thumbKey(scene)
        thumbs.get(key)?.let { return it }
        val bytes = runCatching { api().sceneThumb(scene.id) }.getOrNull() ?: return null
        val image = withContext(Dispatchers.Default) {
            BitmapFactory.decodeByteArray(bytes, 0, bytes.size)?.asImageBitmap()
        } ?: return null
        thumbs.put(key, image)
        return image
    }

    private fun thumbKey(scene: LibraryScene) = "${scene.id}@${scene.mediaAt.orEmpty()}"
}
