return {
    LrSdkVersion = 13.0,
    LrSdkMinimumVersion = 6.0,
    LrToolkitIdentifier = "com.filmcolor.lightroom",
    LrPluginName = "FilmColor v1.0",
    LrPluginInfoUrl = "https://github.com/filmcolor-ai",
    LrPluginInfoProvider = "PluginInfoProvider.lua",
    LrShutdownPlugin = "Shutdown.lua",

    LrLibraryMenuItems = {
        {
            title = "FilmColor 负片转换…",
            file = "ConvertDialog.lua",
            enabledWhen = "photosSelected",
        },
        {
            title = "FilmColor 快速一键转换",
            file = "QuickConvert.lua",
            enabledWhen = "photosSelected",
        },
        {
            title = "FilmColor 停止后台引擎",
            file = "StopBridge.lua",
        },
    },

    LrExportMenuItems = {
        {
            title = "FilmColor 负片转换…",
            file = "ConvertDialog.lua",
            enabledWhen = "photosSelected",
        },
        {
            title = "FilmColor 快速一键转换",
            file = "QuickConvert.lua",
            enabledWhen = "photosSelected",
        },
    },

    VERSION = { major = 1, minor = 0, revision = 0, build = 1 },
}
