return {
    LrSdkVersion = 13.0,
    LrSdkMinimumVersion = 6.0,
    LrToolkitIdentifier = "com.filmcolor.lightroom",
    LrPluginName = "FilmColor v1.0",
    LrPluginInfoUrl = "https://github.com/Guuuu233/FilmColor-AI",
    LrPluginInfoProvider = "PluginInfoProvider.lua",
    LrShutdownPlugin = "Shutdown.lua",

    LrLibraryMenuItems = {
        {
            title = LOC "$$$/FilmColor/Menu/Convert=FilmColor Negative Conversion...",
            file = "ConvertDialog.lua",
            enabledWhen = "photosSelected",
        },
        {
            title = LOC "$$$/FilmColor/Menu/Quick=FilmColor Quick Convert",
            file = "QuickConvert.lua",
            enabledWhen = "photosSelected",
        },
        {
            title = LOC "$$$/FilmColor/Menu/Stop=FilmColor Stop Background Engine",
            file = "StopBridge.lua",
        },
    },

    LrExportMenuItems = {
        {
            title = LOC "$$$/FilmColor/Menu/Convert=FilmColor Negative Conversion...",
            file = "ConvertDialog.lua",
            enabledWhen = "photosSelected",
        },
        {
            title = LOC "$$$/FilmColor/Menu/Quick=FilmColor Quick Convert",
            file = "QuickConvert.lua",
            enabledWhen = "photosSelected",
        },
    },

    VERSION = { major = 1, minor = 0, revision = 0, build = 1 },
}
