local LrApplication = import "LrApplication"
local LrDialogs = import "LrDialogs"
local LrFunctionContext = import "LrFunctionContext"
local LrProgressScope = import "LrProgressScope"

local Bridge = require "Bridge"
local Config = require "Config"
local I18n = require "I18n"
local T = I18n.t

LrFunctionContext.postAsyncTaskWithContext("FilmColorQuickConvert", function(context)
    local catalog = LrApplication.activeCatalog()
    local rawPhotos = catalog:getTargetPhotos()
    if not rawPhotos or #rawPhotos == 0 then
        LrDialogs.message(T("app_name"), T("msg_select_neg"), "info")
        return
    end
    local photos = Bridge.filterNegativePhotos(rawPhotos)
    if #photos == 0 then
        LrDialogs.message(T("app_name"), T("msg_all_converted"), "info")
        return
    end

    local ok, err = Bridge.ensureRunning()
    if not ok then
        LrDialogs.message(T("msg_engine_err"), tostring(err), "critical")
        return
    end

    local prefs = Config.prefs()
    local progress = LrProgressScope {
        title = T("prog_quick_title", #photos),
        functionContext = context,
    }
    progress:setCancelable(true)

    local rollId = nil
    local baseMode = prefs.baseMode or "auto"
    if #photos >= 2 and prefs.rollSync then
        progress:setCaption(T("prog_roll"))
        local paths = {}
        local lrCrops = {}
        for i, ph in ipairs(photos) do
            local pth = ph:getRawMetadata("path")
            paths[i] = pth
            if prefs.useLrCrop then
                local pg = Bridge.readLrGeometry(ph)
                if pg.lr_crop then lrCrops[pth] = pg.lr_crop end
            end
        end
        local rres = Bridge.post("/plugin/roll_analyze", {
            paths = paths, lr_crops = lrCrops, film = prefs.film, engine = prefs.engine, roll_pool = true,
        }, 180)
        if rres and rres.roll_id then
            rollId = rres.roll_id
            if rres.has_roll_base then baseMode = "roll_nat" end
        end
    end

    local imported, errors, firstOutPath = {}, {}, nil
    local pendingImports = {}
    local importOpts = {
        stackWithSource = prefs.stackWithSource,
        destMode = prefs.destMode,
    }
    local function flushImports()
        if #pendingImports == 0 then return end
        local added = Bridge.importAndStackBatch(catalog, pendingImports, importOpts)
        for _, np in ipairs(added) do
            imported[#imported + 1] = np
        end
        pendingImports = {}
    end
    local canceled = false
    for i, ph in ipairs(photos) do
        if progress:isCanceled() then canceled = true; break end
        local pPath = ph:getRawMetadata("path")
        local pName = ph:getFormattedMetadata("fileName") or pPath
        local pGeom = Bridge.readLrGeometry(ph)
        progress:setCaption(T("prog_item", i, #photos, pName))
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
            lr_geom_err = pGeom.geom_err,
            auto_crop = prefs.autoCrop,
            format = prefs.format,
            encoding = prefs.encoding or "srgb",
            quality = 95,
            subfolder = (prefs.destMode == "subfolder") and "FilmColor_Positive" or nil,
            write_sidecar = prefs.writeSidecar,
            copy_exif = true,
        }, 300)
        if res and res.output_path then
            if not firstOutPath then firstOutPath = res.output_path end
            pendingImports[#pendingImports + 1] = { srcPhoto = ph, outPath = res.output_path }
            if #pendingImports >= 20 then
                flushImports()
            end
        else
            errors[#errors + 1] = pName .. ": " .. tostring(cerr)
        end
        progress:setPortionComplete(i, #photos)
    end
    flushImports()
    progress:done()

    if #imported > 0 then
        Bridge.revealImportedPhotos(catalog, imported, firstOutPath)
    end
    if #errors > 0 then
        LrDialogs.message(T("done_err_title", #imported, #errors), table.concat(errors, "\n"), "warning")
    elseif canceled then
        LrDialogs.showBezel(T("done_canceled", #imported, #photos), 3.0)
    else
        LrDialogs.showBezel(T("done_quick_ok", #imported), 3.0)
    end
end)
