local LrApplication = import "LrApplication"
local LrBinding = import "LrBinding"
local LrDialogs = import "LrDialogs"
local LrFunctionContext = import "LrFunctionContext"
local LrProgressScope = import "LrProgressScope"
local LrTasks = import "LrTasks"
local LrView = import "LrView"

local Bridge = require "Bridge"
local Config = require "Config"

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
        LrDialogs.message("FilmColor v1.0", "请先在图库或胶片窗格中选中至少一张负片。", "info")
        return
    end
    local photos, skippedCount = Bridge.filterNegativePhotos(rawPhotos)
    if #photos == 0 then
        LrDialogs.message("FilmColor v1.0", "当前选中的文件均为已转换的正片（*_FilmColor）或不支持格式，请选中原始负片 RAW / DNG / 负片 TIFF。", "info")
        return
    end

    local ok, err = Bridge.ensureRunning()
    if not ok then
        LrDialogs.message("FilmColor 引擎未就绪", tostring(err), "critical")
        return
    end

    local prefs = Config.prefs()
    local firstPhoto = photos[1]
    local firstPath = firstPhoto:getRawMetadata("path")
    local firstName = firstPhoto:getFormattedMetadata("fileName") or ""
    local firstGeom = Bridge.readLrGeometry(firstPhoto)

    local props = LrBinding.makePropertyTable(context)
    props.film = prefs.film or "color"
    props.engine = prefs.engine or "t24"
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

    -- Optional fine-tuning sliders (0 = pure FilmColor v1.0 + preset)
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
    props.statusLine = "正在解码并运行 FilmColor v1.0 预览：" .. firstName .. " …"

    local reqSeq = 0
    local closed = false
    local suppressPreview = false

    local function requestPreview()
        if closed or suppressPreview then return end
        reqSeq = reqSeq + 1
        local mySeq = reqSeq
        props.statusLine = "正在更新预览…"
        LrTasks.startAsyncTask(function()
            -- Small debounce for rapid slider nudges
            LrTasks.sleep(0.08)
            if closed or mySeq ~= reqSeq then return end
            local res, perr = Bridge.post("/plugin/preview", {
                path = firstPath,
                film = props.film,
                engine = props.engine,
                preset = props.preset,
                base_mode = props.baseMode,
                auto_crop = props.autoCrop,
                use_lr_crop = props.useLrCrop,
                lr_crop = firstGeom.lr_crop,
                lr_orientation = firstGeom.lr_orientation,
                lr_angle = firstGeom.lr_angle,
                rot_delta = props.rotDelta,
                view = props.viewMode,
                max_width = 520,
                max_height = 390,
                adjustments = buildAdjustments(props),
            }, 90)
            if closed or mySeq ~= reqSeq then return end
            if not res then
                props.statusLine = "预览失败：" .. tostring(perr)
                return
            end
            props.previewPath = res.preview_path
            local clipTxt = ""
            if res.above_white then
                clipTxt = string.format(" · 高光溢出 %.2f%%", res.above_white * 100)
            end
            props.statusLine = string.format(
                "已选 %d 张 · 预览 %s · %s%s",
                #photos, firstName, ((props.engine == "t24" and "FilmColor v1.0") or (props.engine == "t23" and "物理直出") or "经典引擎"), clipTxt
            )
        end)
    end

    -- Trigger preview when any parameter changes
    local watchKeys = {
        "film", "engine", "preset", "baseMode", "useLrCrop", "autoCrop", "viewMode", "rotDelta",
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
                title = "实时暗房预览（转换后导入为正像 16 位 TIFF，可在 Lr 正向精修）",
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
                        f:radio_button { title = "当前效果", value = bind "viewMode", checked_value = "after" },
                        f:radio_button { title = "自动基准", value = bind "viewMode", checked_value = "auto" },
                        f:radio_button { title = "原始负片", value = bind "viewMode", checked_value = "negative" },
                        f:spacer { fill_horizontal = 1 },
                        f:push_button {
                            title = "↺ 左转",
                            action = function()
                                props.rotDelta = (props.rotDelta + 1) % 4
                            end,
                        },
                        f:push_button {
                            title = "↻ 右转",
                            action = function()
                                props.rotDelta = (props.rotDelta + 3) % 4
                            end,
                        },
                        f:push_button {
                            title = "重置微调",
                            action = function()
                                suppressPreview = true
                                props.ev = 0.0; props.temp = 0; props.tint = 0
                                props.contrast = 0; props.highlights = 0; props.shadows = 0
                                props.whites = 0; props.blacks = 0
                                props.vibrance = 0; props.sat = 0; props.skin_protect = 0
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
                title = "胶片与风格",
                fill_horizontal = 1,
                spacing = 5,
                f:row {
                    f:static_text { title = "胶片类型", width = 68, alignment = "right" },
                    f:popup_menu {
                        value = bind "film",
                        width = 215,
                        items = {
                            { title = "彩色负片 (C-41 / ECN-2)", value = "color" },
                            { title = "黑白负片 (B&W)", value = "bw" },
                        },
                    },
                },
                f:row {
                    f:static_text { title = "风格预设", width = 68, alignment = "right" },
                    f:popup_menu {
                        value = bind "preset",
                        width = 215,
                        items = {
                            { title = "富士 Frontier 鲜润（推荐直出）", value = "frontier" },
                            { title = "德味徕卡 Leica Look（立体油润）", value = "leica" },
                            { title = "诺日士 Noritsu 通透（自然高宽容）", value = "noritsu" },
                            { title = "标准 Standard（保守色度基准）", value = "standard" },
                            { title = "线性 Linear（高宽容后期自调底稿）", value = "linear" },
                            { title = "自然 Natural（柔和暗部层次）", value = "natural" },
                            { title = "人像 Portrait（肤色保护通透）", value = "portrait" },
                            { title = "鲜明 Vivid（通透浓郁色彩）", value = "vivid" },
                            { title = "经典印相 Print（传统暗房相纸）", value = "print_classic" },
                            { title = "电影质感 Cinema（青暗暖亮）", value = "cinema" },
                        },
                    },
                },
                f:row {
                    f:static_text { title = "算法引擎", width = 68, alignment = "right" },
                    f:popup_menu {
                        value = bind "engine",
                        width = 215,
                        items = {
                            { title = "FilmColor v1.0（智能推荐）", value = "t24" },
                            { title = "物理直出引擎", value = "t23" },
                            { title = "经典引擎", value = "v3" },
                        },
                    },
                },
            },

            f:group_box {
                title = "可选预调（0 = 纯自动）",
                fill_horizontal = 1,
                spacing = 3,
                makeSliderRow(f, "曝光 EV", "ev", -2.0, 2.0, 2),
                makeSliderRow(f, "色温", "temp", -100, 100, 0),
                makeSliderRow(f, "色调", "tint", -100, 100, 0),
                makeSliderRow(f, "对比度", "contrast", -100, 100, 0),
                makeSliderRow(f, "高光", "highlights", -100, 100, 0),
                makeSliderRow(f, "阴影", "shadows", -100, 100, 0),
                makeSliderRow(f, "白色色阶", "whites", -100, 100, 0),
                makeSliderRow(f, "黑色色阶", "blacks", -100, 100, 0),
                makeSliderRow(f, "自然饱和", "vibrance", -100, 100, 0),
                makeSliderRow(f, "饱和度", "sat", -100, 100, 0),
                makeSliderRow(f, "肤色保护", "skin_protect", 0, 100, 0),
            },

            f:group_box {
                title = "输出与目录导入",
                fill_horizontal = 1,
                spacing = 5,
                f:row {
                    f:static_text { title = "输出格式", width = 68, alignment = "right" },
                    f:popup_menu {
                        value = bind "format",
                        width = 215,
                        items = {
                            { title = "16 位 TIFF (sRGB · 推荐)", value = "tiff16" },
                            { title = "JPEG (高品质 95%)", value = "jpeg" },
                            { title = "PNG (8 位无损)", value = "png" },
                        },
                    },
                },
                f:row {
                    f:static_text { title = "保存位置", width = 68, alignment = "right" },
                    f:popup_menu {
                        value = bind "destMode",
                        width = 215,
                        items = {
                            { title = "与原负片同目录（可自动堆叠）", value = "same_folder" },
                            { title = "子文件夹 FilmColor_Positive", value = "subfolder" },
                        },
                    },
                },
                f:checkbox {
                    title = "自动导入当前目录并堆叠在原负片上方",
                    value = bind "stackWithSource",
                },
                f:checkbox {
                    title = "多帧选中时启用整卷片基与色彩协同",
                    value = bind "rollSync",
                    enabled = (#photos >= 2),
                },
                f:checkbox {
                    title = "继承 Lr 裁切与旋转（内框精准测光 + 裁切输出）",
                    value = bind "useLrCrop",
                },
                f:checkbox {
                    title = "无 Lr 裁切时自动裁去片夹黑边",
                    value = bind "autoCrop",
                },
                f:checkbox {
                    title = "同时写入参数 Sidecar (.json)",
                    value = bind "writeSidecar",
                },
            },
        },
    }

    local actionVerb = (#photos == 1)
        and "转换并导入正片"
        or string.format("批量转换并导入 (%d 张)", #photos)

    local result = LrDialogs.presentModalDialog {
        title = "FilmColor v1.0 负片暗房",
        contents = contents,
        actionVerb = actionVerb,
        cancelVerb = "取消",
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
        title = string.format("FilmColor v1.0 正在转换 %d 张负片", #photos),
        functionContext = context,
    }
    progress:setCancelable(true)

    local rollId = nil
    local effectiveBase = props.baseMode
    if #photos >= 2 and props.rollSync then
        progress:setCaption("正在分析整卷片基与色彩一致性…")
        local paths = {}
        for i, ph in ipairs(photos) do paths[i] = ph:getRawMetadata("path") end
        local rres = Bridge.post("/plugin/roll_analyze", {
            paths = paths,
            film = props.film,
            engine = props.engine,
            roll_pool = true,
        }, 180)
        if rres and rres.roll_id then
            rollId = rres.roll_id
            if rres.has_roll_base then effectiveBase = "roll_nat" end
        end
    end

    local imported = {}
    local firstOutPath = nil
    local errors = {}
    local adj = buildAdjustments(props)

    local canceled = false
    for i, ph in ipairs(photos) do
        if progress:isCanceled() then canceled = true; break end
        local pPath = ph:getRawMetadata("path")
        local pName = ph:getFormattedMetadata("fileName") or pPath
        local pGeom = Bridge.readLrGeometry(ph)
        progress:setCaption(string.format("正在转换 (%d/%d)：%s", i, #photos, pName))
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
            rot_delta = props.rotDelta,
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
            local np = Bridge.importAndStack(catalog, ph, res.output_path, {
                stackWithSource = props.stackWithSource,
                destMode = props.destMode,
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
        LrDialogs.message(
            string.format("FilmColor 转换完成（成功 %d / 失败 %d）", #imported, #errors),
            table.concat(errors, "\n"),
            "warning"
        )
    elseif canceled then
        LrDialogs.showBezel(string.format("FilmColor v1.0：已取消（已完成 %d / %d 张）", #imported, #photos), 3.0)
    else
        LrDialogs.showBezel(string.format("FilmColor v1.0：已完成 %d 张正片转换并导入堆叠", #imported), 3.0)
    end
end)
