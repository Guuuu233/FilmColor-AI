local LrPrefs = import "LrPrefs"
local LrFileUtils = import "LrFileUtils"
local LrPathUtils = import "LrPathUtils"

local Config = {}

local IS_WIN = (WIN_ENV == true)
Config.IS_WIN = IS_WIN
Config.DEFAULT_PORT = 18765

-- Optional generated override when installed via copy on Windows
local okOv, override = pcall(require, "LocalPaths")
if not okOv or type(override) ~= "table" then override = {} end

local function detectRepoRoot()
    if override.repoRoot and override.repoRoot ~= "" and LrFileUtils.exists(override.repoRoot) then
        return override.repoRoot
    end
    if _PLUGIN and _PLUGIN.path then
        -- <repo>/plugins/lightroom/FilmColor.lrplugin -> 3 levels up
        local p1 = LrPathUtils.parent(_PLUGIN.path)
        local p2 = p1 and LrPathUtils.parent(p1)
        local p3 = p2 and LrPathUtils.parent(p2)
        if p3 and LrFileUtils.exists(LrPathUtils.child(p3, "pyproject.toml")) then
            return p3
        end
    end
    if IS_WIN then
        return "C:\\FilmColor-AI"
    end
    return "/Volumes/waizhiyingpan/FilmColor-AI"
end

Config.DEFAULT_REPO = detectRepoRoot()

local function defaultPythonCandidates()
    local sep = IS_WIN and "\\" or "/"
    local list = {}
    if override.pythonPath and override.pythonPath ~= "" then
        list[#list + 1] = override.pythonPath
    end
    if IS_WIN then
        list[#list + 1] = Config.DEFAULT_REPO .. "\\.venv\\Scripts\\pythonw.exe"
        list[#list + 1] = Config.DEFAULT_REPO .. "\\.venv\\Scripts\\python.exe"
    else
        list[#list + 1] = Config.DEFAULT_REPO .. "/.venv/bin/python"
        list[#list + 1] = "/Volumes/waizhiyingpan/FilmColor-AI/.venv/bin/python"
        list[#list + 1] = "/Users/davidliu/ClaudeScienceProjects/FilmColor-AI/.venv/bin/python"
    end
    return list
end

local function pickDefaultPython()
    local cands = defaultPythonCandidates()
    for _, c in ipairs(cands) do
        if LrFileUtils.exists(c) then return c end
    end
    return cands[1]
end

Config.DEFAULT_PYTHON = pickDefaultPython()

function Config.prefs()
    local p = LrPrefs.prefsForPlugin()
    -- Auto-heal if unset, missing on disk, or pointing outside .venv
    if not p.pythonPath or p.pythonPath == "" or not string.find(p.pythonPath, "%.venv") or not LrFileUtils.exists(p.pythonPath) then
        p.pythonPath = pickDefaultPython()
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
