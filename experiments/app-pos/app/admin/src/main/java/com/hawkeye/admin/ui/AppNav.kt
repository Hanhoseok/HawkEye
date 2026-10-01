package com.hawkeye.admin.ui

import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.navigation.NavType
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import androidx.navigation.navArgument
import kotlinx.coroutines.flow.MutableStateFlow

@Composable
fun AppNav(openCase: MutableStateFlow<Int?>) {
    val nav = rememberNavController()
    val pending by openCase.collectAsState()
    LaunchedEffect(pending) {
        pending?.let {
            nav.navigate("case/$it") { launchSingleTop = true }
            openCase.value = null
        }
    }

    NavHost(navController = nav, startDestination = "cases") {
        composable("cases") {
            CaseListScreen(
                onOpen = { nav.navigate("case/$it") },
                onLive = { nav.navigate("live/") },
                onSettings = { nav.navigate("settings") },
            )
        }
        composable("case/{id}", arguments = listOf(navArgument("id") { type = NavType.IntType })) {
            CaseDetailScreen(
                caseId = it.arguments!!.getInt("id"),
                onBack = { nav.popBackStack() },
                onLive = { camera -> nav.navigate("live/$camera") },
            )
        }
        composable("live/{camera}", arguments = listOf(navArgument("camera") { defaultValue = "" })) {
            LiveScreen(cameraId = it.arguments?.getString("camera").orEmpty(), onBack = { nav.popBackStack() })
        }
        composable("settings") { SettingsScreen(onBack = { nav.popBackStack() }) }
    }
}
