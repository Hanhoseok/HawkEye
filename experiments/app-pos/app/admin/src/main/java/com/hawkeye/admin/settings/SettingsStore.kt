package com.hawkeye.admin.settings

import android.content.Context
import com.hawkeye.admin.data.ServerSettings
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

/** 경보 서버 주소·API 키. 서버를 노트북에서 AWS 로 옮겨도 여기만 바꾸면 된다. */
class SettingsStore(context: Context) {
    private val prefs = context.getSharedPreferences("hawkeye", Context.MODE_PRIVATE)
    private val _settings = MutableStateFlow(load())
    val settings: StateFlow<ServerSettings> = _settings.asStateFlow()

    fun save(baseUrl: String, apiKey: String) {
        prefs.edit().putString(KEY_URL, baseUrl.trim().trimEnd('/')).putString(KEY_API, apiKey.trim()).apply()
        _settings.value = load()
    }

    private fun load() = ServerSettings(
        baseUrl = prefs.getString(KEY_URL, null) ?: DEFAULT_URL,
        apiKey = prefs.getString(KEY_API, null)?.takeIf { it.isNotBlank() },
    )

    companion object {
        /** 에뮬레이터에서 개발 PC 를 가리키는 주소. 실제 폰은 노트북 IP(예: http://192.168.0.10:8000)로 바꾼다. */
        const val DEFAULT_URL = "http://10.0.2.2:8000"
        private const val KEY_URL = "base_url"
        private const val KEY_API = "api_key"
    }
}
