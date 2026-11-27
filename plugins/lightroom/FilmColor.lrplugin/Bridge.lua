local LrHttp = import "LrHttp"
local LrTasks = import "LrTasks"
local LrFileUtils = import "LrFileUtils"
local LrPathUtils = import "LrPathUtils"

local json = require "json"
local Config = require "Config"
local I18n = require "I18n"

local T = I18n.t
local Bridge = {}
local bridgeStarting = false

-- Lightroom Classic runs Lua 5.1: raw pcall() creates a C-call boundary that
-- aborts any yielding SDK call (getDevelopSettings, getRawMetadata, LrHttp, etc.)
-- with 'attempt to yield across metamethod/C-call boundary'. Always use LrTasks.pcall
-- inside async tasks.
local function safeCall(fn, ...)
    if LrTasks.canYield() then
        return LrTasks.pcall(fn, ...)
    end
    return pcall(fn, ...)
end

local SUPPORTED_EXT = {
    nef = true, nrw = true, raf = true, cr2 = true, cr3 = true,
    arw = true, rw2 = true, orf = true, pef = true, srw = true,
    dng = true, tif = true, tiff = true,
}

function Bridge.filterNegativePhotos(photos)
    local valid, skipped = {}, 0
    for _, ph in ipairs(photos or {}) do
        local p = ph:getRawMetadata("path") or ""
        local leaf = LrPathUtils.leafName(p) or ""
        local ext = string.lower(LrPathUtils.extension(p) or "")
        local isAlreadyFC = (string.find(leaf, "_FilmColor", 1, true) ~= nil)
        if SUPPORTED_EXT[ext] and not isAlreadyFC then
            valid[#valid + 1] = ph
        else
            skipped = skipped + 1
        end
    end
    return valid, skipped
end

function Bridge.baseUrl()
    local p = Config.prefs()
    return "http://127.0.0.1:" .. tostring(p.port or Config.DEFAULT_PORT)
end

function Bridge.health()
    local body, hdrs = LrHttp.get(Bridge.baseUrl() .. "/health", nil, 2)
    if body and hdrs and hdrs.status == 200 then
        local data = json.decode(body)
        if type(data) == "table" and data.ok then return true, data end
    end
    return false, nil
end

local function psQuote(s)
    -- Escape single quotes for PowerShell single-quoted string and strip control chars
    local clean = tostring(s or ""):gsub('[%c"]', "")
    return "'" .. clean:gsub("'", "''") .. "'"
end

local function shQuote(s)
    local clean = tostring(s or ""):gsub("[%c]", "")
    return "'" .. clean:gsub("'", "'\''") .. "'"
end

local function launchWinProcess(workDir, exePath, argList, logPath)
    local errLog = tostring(logPath) .. ".err"
    local psArgs = {}
    for i, a in ipairs(argList) do
        psArgs[i] = psQuote(a)
    end
    local psCmd = string.format(
        "Start-Process -WindowStyle Hidden -WorkingDirectory %s -FilePath %s -ArgumentList @(%s) -RedirectStandardOutput %s -RedirectStandardError %s",
        psQuote(workDir), psQuote(exePath), table.concat(psArgs, ","), psQuote(logPath), psQuote(errLog)
    )
    local cmd = string.format('powershell -NoProfile -NonInteractive -ExecutionPolicy Bypass -Command "%s"', psCmd)
    LrTasks.execute(cmd)
end

local function getLogPath()
    local tmpDir = LrPathUtils.getStandardFilePath("temp")
    if tmpDir and tmpDir ~= "" then
        return LrPathUtils.child(tmpDir, "filmcolor_lr_bridge.log")
    end
    return Config.IS_WIN and "filmcolor_lr_bridge.log" or "/tmp/filmcolor_lr_bridge.log"
end

function Bridge.ensureRunning()
    local ok, info = Bridge.health()
    if ok then return true, info end

    -- Serialize concurrent start attempts so two tasks never spawn duplicate daemons
    local waited = 0
    while bridgeStarting and waited < 80 do
        LrTasks.sleep(0.25)
        waited = waited + 1
        local alive, hinfo = Bridge.health()
        if alive then return true, hinfo end
    end
    if bridgeStarting then
        return false, T("err_start_timeout", tostring(getLogPath()))
    end

    bridgeStarting = true
    local startOk, resA, resB = safeCall(function()
        local aliveNow, infoNow = Bridge.health()
        if aliveNow then return true, infoNow end

        local p = Config.prefs()
        local port = tostring(math.floor(tonumber(p.port) or Config.DEFAULT_PORT))
        local bundled = Config.bundledBridgePath()
        local logPath = getLogPath()

        if bundled then
            -- Out-of-the-box standalone engine bundled inside FilmColor.lrplugin (zero Python required)
            if Config.IS_WIN then
                local tmpDir = LrPathUtils.getStandardFilePath("temp") or "."
                local binDir = LrPathUtils.parent(bundled) or tmpDir
                launchWinProcess(binDir, bundled, { "--port", port }, logPath)
            else
                local inner = string.format("nohup %s --port %s >%s 2>&1 </dev/null &", shQuote(bundled), port, shQuote(logPath))
                LrTasks.execute("/bin/sh -c " .. shQuote(inner))
            end
        else
            local py = p.pythonPath or Config.DEFAULT_PYTHON
            if not LrFileUtils.exists(py) then
                local hint = Config.IS_WIN and ".venv\\Scripts\\pythonw.exe" or ".venv/bin/python"
                return false, T("err_missing_py", tostring(py), hint)
            end
            local repoDir = py:match("^(.-)[/\\]%.venv[/\\]") or Config.DEFAULT_REPO
            if Config.IS_WIN then
                local pyw = py:gsub("[Pp][Yy][Tt][Hh][Oo][Nn]%.exe$", "pythonw.exe")
                if LrFileUtils.exists(pyw) then py = pyw end
                local tmpDir = LrPathUtils.getStandardFilePath("temp") or "."
                if not LrFileUtils.exists(repoDir) then
                    repoDir = LrFileUtils.exists(Config.DEFAULT_REPO) and Config.DEFAULT_REPO or tmpDir
                end
                launchWinProcess(repoDir, py, { "-m", "filmcolor.plugin_bridge", "--port", port }, logPath)
            else
                if not LrFileUtils.exists(repoDir) then
                    repoDir = LrFileUtils.exists(Config.DEFAULT_REPO) and Config.DEFAULT_REPO or "/tmp"
                end
                local inner = string.format(
                    "cd %s && nohup %s -m filmcolor.plugin_bridge --port %s >%s 2>&1 </dev/null &",
                    shQuote(repoDir), shQuote(py), port, shQuote(logPath)
                )
                LrTasks.execute("/bin/sh -c " .. shQuote(inner))
            end
        end

        for _ = 1, 60 do
            LrTasks.sleep(0.25)
            local alive, hinfo = Bridge.health()
            if alive then return true, hinfo end
        end
        return false, T("err_start_timeout", tostring(logPath))
    end)
    bridgeStarting = false
    if not startOk then
        return false, tostring(resA)
    end
    return resA, resB
end

function Bridge.post(endpoint, payload, timeoutSec)
    local bodyStr = json.encode(payload or {})
    local headers = {
        { field = "Content-Type", value = "application/json; charset=utf-8" },
    }
    local resp, hdrs = LrHttp.post(Bridge.baseUrl() .. endpoint, bodyStr, headers, "POST", timeoutSec or 180)
    if not resp then
        return nil, T("err_cannot_connect")
    end
    local data, err = json.decode(resp)
    if type(data) ~= "table" then
        return nil, T("err_parse_resp", tostring(err or "non-object JSON"))
    end
    if hdrs and hdrs.status and hdrs.status >= 400 then
        return nil, data.error or ("HTTP " .. tostring(hdrs.status))
    end
    if data.error then
        return nil, data.error
    end
    return data, nil
end

function Bridge.shutdown()
    local ok = Bridge.health()
    if not ok then return true end
    Bridge.post("/shutdown", {}, 3)
    return true
end

local function copyPhotoMetadata(srcPhoto, newPhoto)
    if not newPhoto or not srcPhoto then return end
    safeCall(function()
        local rating = srcPhoto:getRawMetadata("rating")
        if rating then newPhoto:setRawMetadata("rating", rating) end
        local pick = srcPhoto:getRawMetadata("pickStatus")
        if pick then newPhoto:setRawMetadata("pickStatus", pick) end
        local label = srcPhoto:getRawMetadata("colorNameForLabel")
        if label and label ~= "" then newPhoto:setRawMetadata("colorNameForLabel", label) end
        local kws = srcPhoto:getRawMetadata("keywords")
        if kws then
            for _, kw in ipairs(kws) do
                newPhoto:addKeyword(kw)
            end
        end
    end)
end

function Bridge.importAndStackBatch(catalog, items, opts)
    local imported = {}
    if not items or #items == 0 then return imported end
    local wantStack = (opts and opts.stackWithSource and opts.destMode == "same_folder")
    local status = catalog:withWriteAccessDo(T("undo_import"), function()
        for _, item in ipairs(items) do
            local srcPhoto = item.srcPhoto
            local outPath = item.outPath
            local stackTarget = wantStack and srcPhoto or nil
            local stackPos = stackTarget and "above" or nil
            local addOk, newPhoto = safeCall(function()
                return catalog:addPhoto(outPath, stackTarget, stackPos)
            end)
            if addOk and newPhoto then
                copyPhotoMetadata(srcPhoto, newPhoto)
                imported[#imported + 1] = newPhoto
            end
        end
    end, { timeout = 30 })
    if status ~= "executed" then
        return {}
    end
    return imported
end

function Bridge.importAndStack(catalog, srcPhoto, outPath, opts)
    local batch = Bridge.importAndStackBatch(catalog, { { srcPhoto = srcPhoto, outPath = outPath } }, opts)
    return batch[1]
end

function Bridge.revealImportedPhotos(catalog, imported, firstOutPath)
    if not imported or #imported == 0 then return end
    local sources = catalog:getActiveSources() or {}
    local addedToCollection = false
    local inSpecialOrOtherView = false

    for _, src in ipairs(sources) do
        if type(src) == "string" then
            -- Special source like kLastImport ("上一次导入"): newly added photos do not appear there
            if src ~= catalog.kAllPhotos then
                inSpecialOrOtherView = true
            end
        elseif type(src) == "table" and src.type then
            local stype = src:type()
            if stype == "LrCollection" and not src:isSmartCollection() then
                catalog:withWriteAccessDo(T("undo_add_coll"), function()
                    src:addPhotos(imported)
                end, { timeout = 10 })
                addedToCollection = true
            end
        end
    end

    if inSpecialOrOtherView and not addedToCollection and firstOutPath then
        local parentDir = LrPathUtils.parent(firstOutPath)
        local folder = parentDir and catalog:getFolderByPath(parentDir) or nil
        if folder then
            safeCall(function() catalog:setActiveSources({ folder }) end)
            LrTasks.sleep(0.15)
        end
    end

    safeCall(function() catalog:setSelectedPhotos(imported[1], imported) end)
end

function Bridge.readLrGeometry(photo)
    local out = { lr_crop = nil, lr_orientation = nil, lr_angle = 0, has_crop = false, geom_err = nil }
    if not photo then return out end
    local ok, err = safeCall(function()
        local ds = photo:getDevelopSettings()
        if type(ds) == "table" then
            local ori = ds.Orientation or ds.orientation
            if ori and ori ~= "" then
                out.lr_orientation = ori
            end
            -- Lightroom Classic getDevelopSettings() often omits HasCrop (nil) while directly exposing CropTop/Left/Bottom/Right.
            local explicitNoCrop = (ds.HasCrop == false or ds.HasCrop == 0 or ds.HasCrop == "False" or ds.HasCrop == "false")
            local ct = tonumber(ds.CropTop or ds.cropTop)
            local cl = tonumber(ds.CropLeft or ds.cropLeft)
            local cb = tonumber(ds.CropBottom or ds.cropBottom)
            local cr = tonumber(ds.CropRight or ds.cropRight)
            if (not explicitNoCrop) and (ct ~= nil or cl ~= nil or cb ~= nil or cr ~= nil) then
                local t = math.max(0, math.min(1, ct or 0))
                local l = math.max(0, math.min(1, cl or 0))
                local b = math.max(0, math.min(1, cb or 1))
                local r = math.max(0, math.min(1, cr or 1))
                if t > b then t, b = b, t end
                if l > r then l, r = r, l end
                if (b - t) > 0.02 and (r - l) > 0.02 and (t > 0.001 or l > 0.001 or b < 0.999 or r < 0.999) then
                    out.lr_crop = { t, l, b, r }
                    out.has_crop = true
                end
            end
            if not explicitNoCrop then
                local ang = tonumber(ds.CropAngle or ds.cropAngle)
                if ang and math.abs(ang) >= 0.01 then
                    out.lr_angle = ang
                end
            end
        end
        if not out.lr_orientation then
            local rawOri = photo:getRawMetadata("orientation")
            if rawOri and rawOri ~= "" then
                out.lr_orientation = rawOri
            end
        end
    end)
    if not ok then
        out.geom_err = tostring(err)
    end
    return out
end

return Bridge
