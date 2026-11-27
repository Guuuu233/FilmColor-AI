local LrApplication = import "LrApplication"
local LrBinding = import "LrBinding"
local LrDialogs = import "LrDialogs"
local LrFunctionContext = import "LrFunctionContext"
local LrProgressScope = import "LrProgressScope"
local LrTasks = import "LrTasks"
local LrFileUtils = import "LrFileUtils"
local LrView = import "LrView"

local Bridge = require "Bridge"
local Config = require "Config"
local I18n = require "I18n"
local T = I18n.t

local bind = LrView.bind

local function buildAdjustments(props)
    return {
        ev = props.ev,
        temp = props.temp,
        tint = props.tint,
        contrast = props.contrast,
        highlights = props.highlights,
        shadows = props.shadows,
        whites = props.whites,
        blacks = props.blacks,
        vibrance = props.vibrance,
        sat = props.sat,
        skin_protect = props.skin_protect,
    }
end

local function makeSliderRow(f, label, key, minVal, maxVal, precision)
    return f:row {
        spacing = f:control_spacing(),
        f:static_text {
            title = label,
            width = 72,
            alignment = "right",
        },
        f:slider {
            value = bind(key),
            min = minVal,
            max = maxVal,
            width = 160,
            immediate = false,
        },
        f:edit_field {
            value = bind(key),
            min = minVal,
            max = maxVal,
            precision = precision or 0,
            width_in_digits = 5,
            immediate = false,
        },
    }
end

LrFunctionContext.postAsyncTaskWithContext("FilmColorConvertDialog", function(context)
    local catalog = LrApplication.activeCatalog()
    local rawPhotos = catalog:getTargetPhotos()
    if not rawPhotos or #rawPhotos == 0 then
        LrDialogs.message(T("app_name"), T("msg_select_neg"), "info")
        return
    end
    local photos, skippedCount = Bridge.filterNegativePhotos(rawPhotos)
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
    local firstPhoto = photos[1]
    local firstPath = firstPhoto:getRawMetadata("path")
    local firstName = firstPhoto:getFormattedMetadata("fileName") or ""
    if not firstPath or firstPath == "" or not LrFileUtils.exists(firstPath) then
        LrDialogs.message(T("app_name"), T("err_file_offline", tostring(firstPath or firstName)), "warning")
        return
    end
    local firstGeom = Bridge.readLrGeometry(firstPhoto)

    local props = LrBinding.makePropertyTable(context)
    props.film = prefs.film or "color"
    props.engine = (prefs.engine == "t24" or not prefs.engine) and "t25" or prefs.engine
    props.preset = prefs.preset or "frontier"
    props.useLrCrop = prefs.useLrCrop
    props.baseMode = prefs.baseMode or "auto"
    props.format = prefs.format or "tiff16"
    props.encoding = prefs.encoding or "srgb"
    props.destMode = prefs.destMode or "same_folder"
    props.stackWithSource = prefs.stackWithSource
    props.rollSync = (#photos >= 2) and prefs.rollSync or false
    props.autoCrop = prefs.autoCrop
    props.writeSidecar = prefs.writeSidecar

    -- Optional fine-tuning sliders (0 = pure FilmColor v2.0 + preset)
    props.ev = 0.0
    props.temp = 0
    props.tint = 0
    props.contrast = 0
    props.highlights = 0
    props.shadows = 0
    props.whites = 0
    props.blacks = 0
    props.vibrance = 0
    props.sat = 0
    props.skin_protect = 0

    props.viewMode = "after"
    props.rotDelta = 0
    props.previewPath = nil
    props.statusLine = T("st_decoding", firstName)

    local reqSeq = 0
    local closed = false
    local suppressPreview = false
    local rollCache = { key = nil, rollId = nil, hasRollBase = false, inFlight = false, inFlightKey = nil }

    local function ensureRollAnalyzed(film, engine, useLrCrop)
        local key = tostring(film) .. "|" .. tostring(engine) .. "|" .. tostring(useLrCrop)
        -- Serialize concurrent preview clicks so /plugin/roll_analyze is never triggered twice simultaneously
        while rollCache.inFlight and not closed do
            LrTasks.sleep(0.1)
        end
        if closed then return nil, false end
        if rollCache.key == key and rollCache.rollId then
            return rollCache.rollId, rollCache.hasRollBase
        end
        rollCache.inFlight = true
        rollCache.inFlightKey = key
        local okCall, rres = LrTasks.pcall(function()
            local paths = {}
            local lrCrops = {}
            for i, ph in ipairs(photos) do
                local pth = ph:getRawMetadata("path")
                paths[i] = pth
                if useLrCrop then
                    local pg = Bridge.readLrGeometry(ph)
                    if pg.lr_crop then lrCrops[pth] = pg.lr_crop end
                end
            end
            return Bridge.post("/plugin/roll_analyze", {
                paths = paths,
                lr_crops = lrCrops,
                film = film,
                engine = engine,
                roll_pool = true,
            }, 180)
        end)
        rollCache.inFlight = false
        rollCache.inFlightKey = nil
        if okCall and type(rres) == "table" and rres.roll_id then
            rollCache.key = key
            rollCache.rollId = rres.roll_id
            rollCache.hasRollBase = (rres.has_roll_base == true)
            return rollCache.rollId, rollCache.hasRollBase
        end
        return nil, false
    end

    local function requestPreview()
        if closed or suppressPreview then return end
        reqSeq = reqSeq + 1
        local mySeq = reqSeq
        props.statusLine = T("st_updating")
        LrTasks.startAsyncTask(function()
            -- Small debounce for rapid slider nudges
            LrTasks.sleep(0.08)
            if closed or mySeq ~= reqSeq then return end
            local activeRollId = nil
            local activeRollPool = (#photos >= 2 and props.rollSync == true)
            local effBase = props.baseMode
            if activeRollPool then
                local rkey = tostring(props.film) .. "|" .. tostring(props.engine) .. "|" .. tostring(props.useLrCrop)
                if rollCache.key ~= rkey or not rollCache.rollId then
                    props.statusLine = T("prog_roll")
                end
                local rid, hasRb = ensureRollAnalyzed(props.film, props.engine, props.useLrCrop)
                if closed or mySeq ~= reqSeq then return end
                activeRollId = rid
                if hasRb and effBase == "auto" then
                    effBase = "roll_nat"
                end
            end
            local res, perr = Bridge.post("/plugin/preview", {
                path = firstPath,
                film = props.film,
                engine = props.engine,
                preset = props.preset,
                base_mode = effBase,
                roll_id = activeRollId,
                roll_pool = activeRollPool,
                auto_crop = props.autoCrop,
                use_lr_crop = props.useLrCrop,
                lr_crop = firstGeom.lr_crop,
                lr_orientation = firstGeom.lr_orientation,
                lr_angle = firstGeom.lr_angle,
                lr_geom_err = firstGeom.geom_err,
                rot_delta = props.rotDelta,
                view = props.viewMode,
                max_width = 520,
                max_height = 390,
                adjustments = buildAdjustments(props),
            }, 90)
            if closed or mySeq ~= reqSeq then return end
            if not res then
                props.statusLine = T("st_prev_fail", tostring(perr))
                return
            end
            props.previewPath = res.preview_path
            local clipTxt = ""
            if res.above_white then
                clipTxt = T("st_clip", res.above_white * 100)
            end
            props.statusLine = T(
                "st_ready",
                #photos, firstName, (((props.engine == "t25" or props.engine == "t24") and T("e_t25")) or (props.engine == "t23" and T("e_t23")) or T("e_v3")), clipTxt
            )
        end)
    end

    -- Trigger preview when any parameter changes
    local watchKeys = {
        "film", "engine", "preset", "baseMode", "rollSync", "useLrCrop", "autoCrop", "viewMode", "rotDelta",
        "ev", "temp", "tint", "contrast", "highlights", "shadows",
        "whites", "blacks", "vibrance", "sat", "skin_protect",
    }
    for _, k in ipairs(watchKeys) do
        props:addObserver(k, requestPreview)
    end

    -- Load initial preview asynchronously right as dialog opens
    requestPreview()

    local f = LrView.osFactory()
    local contents = f:row {
        bind_to_object = props,
        spacing = f:dialog_spacing(),

        -- Left: Native Live Preview Stage
        f:column {
            spacing = f:control_spacing(),
            f:group_box {
                title = T("preview_group"),
                font = "<system/small>",
                f:column {
                    spacing = 6,
                    f:picture {
                        value = bind "previewPath",
                        width = 520,
                        height = 390,
                    },
                    f:row {
                        spacing = f:control_spacing(),
                        f:radio_button { title = T("view_after"), value = bind "viewMode", checked_value = "after" },
                        f:radio_button { title = T("view_auto"), value = bind "viewMode", checked_value = "auto" },
                        f:radio_button { title = T("view_neg"), value = bind "viewMode", checked_value = "negative" },
                        f:spacer { fill_horizontal = 1 },
                        f:push_button {
                            title = T("rot_l"),
                            action = function()
                                props.rotDelta = (props.rotDelta + 1) % 4
                            end,
                        },
                        f:push_button {
                            title = T("rot_r"),
                            action = function()
                                props.rotDelta = (props.rotDelta + 3) % 4
                            end,
                        },
                        f:push_button {
                            title = T("reset_adj"),
                            action = function()
                                suppressPreview = true
                                props.ev = 0.0; props.temp = 0; props.tint = 0
                                props.contrast = 0; props.highlights = 0; props.shadows = 0
                                props.whites = 0; props.blacks = 0
                                props.vibrance = 0; props.sat = 0; props.skin_protect = 0
                                props.rotDelta = 0
                                suppressPreview = false
                                requestPreview()
                            end,
                        },
                    },
                    f:static_text {
                        title = bind "statusLine",
                        width = 520,
                        font = "<system/small>",
                    },
                },
            },
        },

        -- Right: Clean Native Controls
        f:column {
            spacing = f:control_spacing(),
            width = 330,

            f:group_box {
                title = T("film_group"),
                fill_horizontal = 1,
                spacing = 5,
                f:row {
                    f:static_text { title = T("film_type"), width = 68, alignment = "right" },
                    f:popup_menu {
                        value = bind "film",
                        width = 215,
                        items = {
                            { title = T("film_color"), value = "color" },
                            { title = T("film_bw"), value = "bw" },
                        },
                    },
                },
                f:row {
                    f:static_text { title = T("preset_lbl"), width = 68, alignment = "right" },
                    f:popup_menu {
                        value = bind "preset",
                        width = 215,
                        items = {
                            { title = T("p_frontier"), value = "frontier" },
                            { title = T("p_leica"), value = "leica" },
                            { title = T("p_noritsu"), value = "noritsu" },
                            { title = T("p_standard"), value = "standard" },
                            { title = T("p_linear"), value = "linear" },
                            { title = T("p_natural"), value = "natural" },
                            { title = T("p_portrait"), value = "portrait" },
                            { title = T("p_vivid"), value = "vivid" },
                            { title = T("p_print_classic"), value = "print_classic" },
                            { title = T("p_cinema"), value = "cinema" },
                        },
                    },
                },
                f:row {
                    f:static_text { title = T("engine_lbl"), width = 68, alignment = "right" },
                    f:popup_menu {
                        value = bind "engine",
                        width = 215,
                        items = {
                            { title = T("e_t25"), value = "t25" },
                            { title = T("e_t23"), value = "t23" },
                            { title = T("e_v3"), value = "v3" },
                        },
                    },
                },
            },

            f:group_box {
                title = T("adj_group"),
                fill_horizontal = 1,
                spacing = 3,
                makeSliderRow(f, T("ev"), "ev", -2.0, 2.0, 2),
                makeSliderRow(f, T("temp"), "temp", -100, 100, 0),
                makeSliderRow(f, T("tint"), "tint", -100, 100, 0),
                makeSliderRow(f, T("contrast"), "contrast", -100, 100, 0),
                makeSliderRow(f, T("highlights"), "highlights", -100, 100, 0),
                makeSliderRow(f, T("shadows"), "shadows", -100, 100, 0),
                makeSliderRow(f, T("whites"), "whites", -100, 100, 0),
                makeSliderRow(f, T("blacks"), "blacks", -100, 100, 0),
                makeSliderRow(f, T("vibrance"), "vibrance", -100, 100, 0),
                makeSliderRow(f, T("sat"), "sat", -100, 100, 0),
                makeSliderRow(f, T("skin_protect"), "skin_protect", 0, 100, 0),
            },

            f:group_box {
                title = T("out_group"),
                fill_horizontal = 1,
                spacing = 5,
                f:row {
                    f:static_text { title = T("fmt_lbl"), width = 68, alignment = "right" },
                    f:popup_menu {
                        value = bind "format",
                        width = 215,
                        items = {
                            { title = T("fmt_tiff16"), value = "tiff16" },
                            { title = T("fmt_jpeg"), value = "jpeg" },
                            { title = T("fmt_png"), value = "png" },
                        },
                    },
                },
                f:row {
                    f:static_text { title = T("dest_lbl"), width = 68, alignment = "right" },
                    f:popup_menu {
                        value = bind "destMode",
                        width = 215,
                        items = {
                            { title = T("dest_same"), value = "same_folder" },
                            { title = T("dest_sub"), value = "subfolder" },
                        },
                    },
                },
                f:checkbox {
                    title = T("chk_stack"),
                    value = bind "stackWithSource",
                },
                f:checkbox {
                    title = T("chk_roll"),
                    value = bind "rollSync",
                    enabled = (#photos >= 2),
                },
                f:checkbox {
                    title = T("chk_lr_crop"),
                    value = bind "useLrCrop",
                },
                f:checkbox {
                    title = T("chk_auto_crop"),
                    value = bind "autoCrop",
                },
                f:checkbox {
                    title = T("chk_sidecar"),
                    value = bind "writeSidecar",
                },
            },
        },
    }

    local actionVerb = (#photos == 1)
        and T("btn_convert_one")
        or T("btn_convert_many", #photos)

    local result = LrDialogs.presentModalDialog {
        title = T("dialog_title"),
        contents = contents,
        actionVerb = actionVerb,
        cancelVerb = T("btn_cancel"),
    }
    closed = true
    if result ~= "ok" then return end

    -- Remember user settings
    prefs.film = props.film
    prefs.engine = props.engine
    prefs.preset = props.preset
    prefs.baseMode = props.baseMode
    prefs.format = props.format
    prefs.destMode = props.destMode
    prefs.stackWithSource = props.stackWithSource
    if #photos >= 2 then prefs.rollSync = props.rollSync end
    prefs.useLrCrop = props.useLrCrop
    prefs.autoCrop = props.autoCrop
    prefs.writeSidecar = props.writeSidecar

    -- Run full-resolution sequential conversion with native Lightroom progress bar
    local progress = LrProgressScope {
        title = T("prog_title", #photos),
        functionContext = context,
    }
    progress:setCancelable(true)

    local rollId = nil
    local effectiveBase = props.baseMode
    if #photos >= 2 and props.rollSync then
        progress:setCaption(T("prog_roll"))
        local rid, hasRb = ensureRollAnalyzed(props.film, props.engine, props.useLrCrop)
        if rid then
            rollId = rid
            if hasRb and effectiveBase == "auto" then effectiveBase = "roll_nat" end
        end
    end

    local imported = {}
    local firstOutPath = nil
    local errors = {}
    local adj = buildAdjustments(props)

    local canceled = false
    local pendingImports = {}
    local importOpts = {
        stackWithSource = props.stackWithSource,
        destMode = props.destMode,
    }
    local function flushImports()
        if #pendingImports == 0 then return end
        local added = Bridge.importAndStackBatch(catalog, pendingImports, importOpts)
        for _, np in ipairs(added) do
            imported[#imported + 1] = np
        end
        pendingImports = {}
    end

    for i, ph in ipairs(photos) do
        if progress:isCanceled() then canceled = true; break end
        local pPath = ph:getRawMetadata("path")
        local pName = ph:getFormattedMetadata("fileName") or pPath
        local pGeom = Bridge.readLrGeometry(ph)
        progress:setCaption(T("prog_item", i, #photos, pName))
        progress:setPortionComplete(i - 1, #photos)

        local res, cerr = Bridge.post("/plugin/convert", {
            path = pPath,
            film = props.film,
            engine = props.engine,
            preset = props.preset,
            base_mode = effectiveBase,
            roll_id = rollId,
            roll_pool = (#photos >= 2 and props.rollSync),
            use_lr_crop = props.useLrCrop,
            lr_crop = pGeom.lr_crop,
            lr_orientation = pGeom.lr_orientation,
            lr_angle = pGeom.lr_angle,
            lr_geom_err = pGeom.geom_err,
            rot_delta = (#photos == 1) and props.rotDelta or 0,
            auto_crop = props.autoCrop,
            format = props.format,
            encoding = props.encoding or "srgb",
            quality = 95,
            subfolder = (props.destMode == "subfolder") and "FilmColor_Positive" or nil,
            write_sidecar = props.writeSidecar,
            copy_exif = true,
            adjustments = adj,
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
        LrDialogs.message(
            T("done_err_title", #imported, #errors),
            table.concat(errors, "\n"),
            "warning"
        )
    elseif canceled then
        LrDialogs.showBezel(T("done_canceled", #imported, #photos), 3.0)
    else
        LrDialogs.showBezel(T("done_ok", #imported), 3.0)
    end
end)
