package com.hawkeye.admin

/** 실제 경보 서버 응답을 저장한 JSON (src/test/resources/fixtures). 서버 형식이 바뀌면 다시 만든다. */
object Fixtures {
    fun read(name: String): String =
        requireNotNull(javaClass.classLoader!!.getResource("fixtures/$name")) { "fixture 없음: $name" }.readText()
}
