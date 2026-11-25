package com.hawkeye.admin

import android.content.Context
import com.hawkeye.admin.alerts.AlertRepository
import com.hawkeye.admin.data.ApiClient
import com.hawkeye.admin.data.LibraryRepository
import com.hawkeye.admin.settings.SettingsStore

/** 앱 전체에서 하나씩만 쓰는 객체. */
object AppGraph {
    lateinit var settings: SettingsStore
        private set
    lateinit var alerts: AlertRepository
        private set
    lateinit var library: LibraryRepository
        private set

    @Synchronized
    fun init(context: Context) {
        if (::settings.isInitialized) return
        settings = SettingsStore(context.applicationContext)
        val s = settings
        alerts = AlertRepository { ApiClient(s.settings.value) }
        library = LibraryRepository { ApiClient(s.settings.value) }
    }
}
