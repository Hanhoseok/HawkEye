package com.hawkeye.admin.domain

/**
 * 경보 서버와 끊긴 상태로 볼지. 처음 연결하는 중(한 번도 연결된 적 없음)은 끊김이 아니지만,
 * 한 번 연결됐다가 끊긴 뒤 다시 붙는 중(연결 시도 중)은 연결될 때까지 끊김으로 본다 — 그동안 경보를 못 받으므로.
 */
object Links {
    fun offline(connected: Boolean, disconnected: Boolean, everConnected: Boolean): Boolean =
        !connected && (disconnected || everConnected)
}
