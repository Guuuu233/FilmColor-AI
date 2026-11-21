local LrPrefs = import "LrPrefs"
local LrFileUtils = import "LrFileUtils"

local Config = {}

Config.DEFAULT_REPO = "/Volumes/waizhiyingpan/FilmColor-AI"
Config.DEFAULT_PYTHON = "/Volumes/waizhiyingpan/FilmColor-AI/.venv/bin/python"
Config.FALLBACK_PYTHON = "/Users/davidliu/ClaudeScienceProjects/FilmColor-AI/.venv/bin/python"
Config.DEFAULT_PORT = 18765

function Config.prefs()
    local p = LrPrefs.prefsForPlugin()
    -- Auto-heal if a prior run stored a symlink-resolved base interpreter outside .venv
    if not p.pythonPath or p.pythonPath == "" or not string.find(p.pythonPath, "%.venv") then
        if LrFileUtils.exists(Config.DEFAULT_PYTHON) then
            p.pythonPath = Config.DEFAULT_PYTHON
        else
            p.pythonPath = Config.FALLBACK_PYTHON
        end
    end
    if not p.port then p.port = Config.DEFAULT_PORT end
    if not p.engine then p.engine = "t24" end
    if not p.film then p.film = "color" end
    if not p.preset then p.preset = "frontier" end
    if not p.baseMode then p.baseMode = "auto" end
    if not p.format then p.format = "tiff16" end
    if not p.encoding then p.encoding = "srgb" end
    if not p.destMode then p.destMode = "same_folder" end
    if p.stackWithSource == nil then p.stackWithSource = true end
    if p.rollSync == nil then p.rollSync = true end
    if p.useLrCrop == nil then p.useLrCrop = true end
    if p.autoCrop == nil then p.autoCrop = false end
    if p.writeSidecar == nil then p.writeSidecar = false end
    return p
end

return Config
