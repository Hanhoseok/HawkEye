package com.hawkeye.admin.domain

/** 재연결 대기 시간: base, 2배씩 늘리고 max 에서 멈춘다. 연결되면 reset. */
class Backoff(private val baseMs: Long = 1_000, private val maxMs: Long = 30_000) {
    private var current = baseMs

    fun next(): Long = current.also { current = (current * 2).coerceAtMost(maxMs) }

    fun reset() {
        current = baseMs
    }
}
