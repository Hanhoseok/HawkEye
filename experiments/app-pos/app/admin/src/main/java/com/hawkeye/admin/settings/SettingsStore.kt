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

    /** 라이브에서 AI 화면(상자·번호)을 볼지. 마지막 선택을 기억한다. */
    private val _aiOverlay = MutableStateFlow(prefs.getBoolean(KEY_AI, false))
    val aiOverlay: StateFlow<Boolean> = _aiOverlay.asStateFlow()

    fun setAiOverlay(on: Boolean) {
        prefs.edit().putBoolean(KEY_AI, on).apply()
        _aiOverlay.value = on
    }

    /** 서버 주소를 한 번이라도 저장했는가. 아니면 시작 안내를 보여 준다. */
    private val _configured = MutableStateFlow(prefs.contains(KEY_URL))
    val configured: StateFlow<Boolean> = _configured.asStateFlow()

    /** 알림 권한을 이미 물어봤는가 (안드로이드 13+). 거절했으면 다시 자동으로 묻지 않고 설정 화면에서 안내한다. */
    var notificationAsked: Boolean
        get() = prefs.getBoolean(KEY_NOTIF_ASKED, false)
        set(v) { prefs.edit().putBoolean(KEY_NOTIF_ASKED, v).apply() }

    fun save(baseUrl: String, apiKey: String) {
        prefs.edit().putString(KEY_URL, baseUrl.trim().trimEnd('/')).putString(KEY_API, apiKey.trim()).apply()
        _settings.value = load()
        _configured.value = true
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
        private const val KEY_AI = "ai_overlay"
        private const val KEY_NOTIF_ASKED = "notification_asked"
    }
}
