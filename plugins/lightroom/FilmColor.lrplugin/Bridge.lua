local LrHttp = import "LrHttp"
local LrTasks = import "LrTasks"
local LrFileUtils = import "LrFileUtils"
local LrPathUtils = import "LrPathUtils"

local json = require "json"
local Config = require "Config"

local Bridge = {}

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
        if data and data.ok then return true, data end
    end
    return false, nil
end

function Bridge.ensureRunning()
    local ok, info = Bridge.health()
    if ok then return true, info end

    local p = Config.prefs()
    local py = p.pythonPath or Config.DEFAULT_PYTHON
    if not LrFileUtils.exists(py) then
        local hint = Config.IS_WIN and ".venv\\Scripts\\pythonw.exe" or ".venv/bin/python"
        return false, "找不到 FilmColor Python 环境：" .. tostring(py) .. "。请在「文件 → 增效工具管理器 → FilmColor v1.0」中设置 " .. hint .. " 路径。"
    end
    local port = tostring(math.floor(tonumber(p.port) or Config.DEFAULT_PORT))
    local repoDir = py:match("^(.-)[/\\]%.venv[/\\]") or Config.DEFAULT_REPO
    local logPath

    if Config.IS_WIN then
        -- Prefer pythonw.exe to avoid flashing a console window on Windows
        local pyw = py:gsub("[Pp][Yy][Tt][Hh][Oo][Nn]%.exe$", "pythonw.exe")
        if LrFileUtils.exists(pyw) then py = pyw end
        local tmpDir = LrPathUtils.getStandardFilePath("temp") or "."
        logPath = LrPathUtils.child(tmpDir, "filmcolor_lr_bridge.log")
        if not LrFileUtils.exists(repoDir) then
            repoDir = LrFileUtils.exists(Config.DEFAULT_REPO) and Config.DEFAULT_REPO or tmpDir
        end
        local cmd = string.format(
            'cmd /c start "" /B /D "%s" "%s" -m filmcolor.plugin_bridge --port %s',
            repoDir, py, port
        )
        LrTasks.execute(cmd)
    else
        logPath = "/tmp/filmcolor_lr_bridge.log"
        if not LrFileUtils.exists(repoDir) then
            repoDir = LrFileUtils.exists(Config.DEFAULT_REPO) and Config.DEFAULT_REPO or "/tmp"
        end
        local function shQuote(s) return "'" .. tostring(s):gsub("'", "'\''") .. "'" end
        local inner = string.format(
            "cd %s && nohup %s -m filmcolor.plugin_bridge --port %s >/tmp/filmcolor_lr_bridge.log 2>&1 </dev/null &",
            shQuote(repoDir), shQuote(py), port
        )
        local cmd = "/bin/sh -c " .. shQuote(inner)
        LrTasks.execute(cmd)
    end

    for _ = 1, 40 do
        LrTasks.sleep(0.25)
        local alive, hinfo = Bridge.health()
        if alive then return true, hinfo end
    end
    return false, "FilmColor 后台引擎启动超时，请检查日志：" .. tostring(logPath)
end

function Bridge.post(endpoint, payload, timeoutSec)
    local bodyStr = json.encode(payload or {})
    local headers = {
        { field = "Content-Type", value = "application/json; charset=utf-8" },
    }
    local resp, hdrs = LrHttp.post(Bridge.baseUrl() .. endpoint, bodyStr, headers, "POST", timeoutSec or 180)
    if not resp then
        return nil, "无法连接 FilmColor 本地引擎（请重试或检查插件管理器状态）"
    end
    local data, err = json.decode(resp)
    if not data then
        return nil, "解析引擎返回失败：" .. tostring(err)
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

function Bridge.importAndStack(catalog, srcPhoto, outPath, opts)
    local newPhoto = nil
    local stackTarget = (opts and opts.stackWithSource and opts.destMode == "same_folder") and srcPhoto or nil
    local stackPos = stackTarget and "above" or nil

    catalog:withWriteAccessDo("FilmColor v1.0 导入正片", function()
        newPhoto = catalog:addPhoto(outPath, stackTarget, stackPos)
        if newPhoto and srcPhoto then
            pcall(function()
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
    end, { timeout = 15 })
    return newPhoto
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
                catalog:withWriteAccessDo("FilmColor 添加至当前收藏夹", function()
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
            pcall(function() catalog:setActiveSources({ folder }) end)
            LrTasks.sleep(0.15)
        end
    end

    pcall(function() catalog:setSelectedPhotos(imported[1], imported) end)
end

function Bridge.readLrGeometry(photo)
    local out = { lr_crop = nil, lr_orientation = nil, lr_angle = 0, has_crop = false }
    if not photo then return out end
    pcall(function()
        local ds = photo:getDevelopSettings()
        if type(ds) == "table" then
            local ori = ds.Orientation or ds.orientation
            if ori and ori ~= "" then
                out.lr_orientation = ori
            end
            if ds.HasCrop then
                local t = tonumber(ds.CropTop) or 0
                local l = tonumber(ds.CropLeft) or 0
                local b = tonumber(ds.CropBottom) or 1
                local r = tonumber(ds.CropRight) or 1
                if (b - t) > 0.05 and (r - l) > 0.05 and (t > 0.001 or l > 0.001 or b < 0.999 or r < 0.999) then
                    out.lr_crop = { t, l, b, r }
                    out.has_crop = true
                end
                out.lr_angle = tonumber(ds.CropAngle) or 0
            end
        end
        if not out.lr_orientation then
            local rawOri = photo:getRawMetadata("orientation")
            if rawOri and rawOri ~= "" then
                out.lr_orientation = rawOri
            end
        end
    end)
    return out
end

return Bridge
