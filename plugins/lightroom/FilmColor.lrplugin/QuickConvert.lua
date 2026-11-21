local LrApplication = import "LrApplication"
local LrDialogs = import "LrDialogs"
local LrFunctionContext = import "LrFunctionContext"
local LrProgressScope = import "LrProgressScope"

local Bridge = require "Bridge"
local Config = require "Config"

LrFunctionContext.postAsyncTaskWithContext("FilmColorQuickConvert", function(context)
    local catalog = LrApplication.activeCatalog()
    local rawPhotos = catalog:getTargetPhotos()
    if not rawPhotos or #rawPhotos == 0 then
        LrDialogs.message("FilmColor T24", "请先选中至少一张负片。", "info")
        return
    end
    local photos = Bridge.filterNegativePhotos(rawPhotos)
    if #photos == 0 then
        LrDialogs.message("FilmColor T24", "当前选中的文件均为已转换的正片（*_FilmColor），请选中原始负片。", "info")
        return
    end

    local ok, err = Bridge.ensureRunning()
    if not ok then
        LrDialogs.message("FilmColor 引擎未就绪", tostring(err), "critical")
        return
    end

    local prefs = Config.prefs()
    local progress = LrProgressScope {
        title = string.format("FilmColor 快速转换 (%d 张 · %s)", #photos, string.upper(prefs.engine)),
        functionContext = context,
    }
    progress:setCancelable(true)

    local rollId = nil
    local baseMode = prefs.baseMode or "auto"
    if #photos >= 2 and prefs.rollSync then
        progress:setCaption("正在分析整卷片基与 T21b 色彩一致性…")
        local paths = {}
        for i, ph in ipairs(photos) do paths[i] = ph:getRawMetadata("path") end
        local rres = Bridge.post("/plugin/roll_analyze", {
            paths = paths, film = prefs.film, engine = prefs.engine, roll_pool = true,
        }, 180)
        if rres and rres.roll_id then
            rollId = rres.roll_id
            if rres.has_roll_base then baseMode = "roll_nat" end
        end
    end

    local imported, errors, firstOutPath = {}, {}, nil
    for i, ph in ipairs(photos) do
        if progress:isCanceled() then break end
        local pPath = ph:getRawMetadata("path")
        local pName = ph:getFormattedMetadata("fileName") or pPath
        local pGeom = Bridge.readLrGeometry(ph)
        progress:setCaption(string.format("正在转换 (%d/%d)：%s", i, #photos, pName))
        progress:setPortionComplete(i - 1, #photos)
        local res, cerr = Bridge.post("/plugin/convert", {
            path = pPath,
            film = prefs.film,
            engine = prefs.engine,
            preset = prefs.preset,
            base_mode = baseMode,
            roll_id = rollId,
            roll_pool = (#photos >= 2 and prefs.rollSync),
            use_lr_crop = prefs.useLrCrop,
            lr_crop = pGeom.lr_crop,
            lr_orientation = pGeom.lr_orientation,
            lr_angle = pGeom.lr_angle,
            auto_crop = prefs.autoCrop,
            format = prefs.format,
            encoding = prefs.encoding or "srgb",
            quality = 95,
            subfolder = (prefs.destMode == "subfolder") and "FilmColor_Positive" or nil,
            write_sidecar = prefs.writeSidecar,
            copy_exif = true,
        }, 300)
        if res and res.output_path then
            local np = Bridge.importAndStack(catalog, ph, res.output_path, {
                stackWithSource = prefs.stackWithSource,
                destMode = prefs.destMode,
            })
            if not firstOutPath then firstOutPath = res.output_path end
            if np then imported[#imported + 1] = np end
        else
            errors[#errors + 1] = pName .. "：" .. tostring(cerr)
        end
        progress:setPortionComplete(i, #photos)
    end
    progress:done()

    if #imported > 0 then
        Bridge.revealImportedPhotos(catalog, imported, firstOutPath)
    end
    if #errors > 0 then
        LrDialogs.message(string.format("FilmColor 快速转换（成功 %d / 失败 %d）", #imported, #errors),
            table.concat(errors, "\n"), "warning")
    else
        LrDialogs.showBezel(string.format("FilmColor T24：已完成 %d 张正片转换", #imported), 3.0)
    end
end)
