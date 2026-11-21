local LrTasks = import "LrTasks"
local LrView = import "LrView"
local LrDialogs = import "LrDialogs"
local Bridge = require "Bridge"
local Config = require "Config"

local bind = LrView.bind

return {
    startDialog = function(propertyTable)
        local p = Config.prefs()
        propertyTable.pythonPath = p.pythonPath
        propertyTable.port = tostring(p.port)
        propertyTable.bridgeStatus = "点击「检测/启动引擎」查看状态"
    end,

    endDialog = function(propertyTable)
        local p = Config.prefs()
        p.pythonPath = propertyTable.pythonPath
        local pn = tonumber(propertyTable.port)
        if pn and pn >= 1024 and pn <= 65535 then p.port = math.floor(pn) end
    end,

    sectionsForTopOfDialog = function(f, propertyTable)
        return {
            {
                title = "FilmColor T24 引擎设置",
                bind_to_object = propertyTable,
                f:row {
                    f:static_text { title = "Python 路径", width = 90, alignment = "right" },
                    f:edit_field { value = bind "pythonPath", fill_horizontal = 1 },
                },
                f:row {
                    f:static_text { title = "本地端口", width = 90, alignment = "right" },
                    f:edit_field { value = bind "port", width_in_digits = 6 },
                    f:push_button {
                        title = "检测 / 启动引擎",
                        action = function()
                            local p = Config.prefs()
                            p.pythonPath = propertyTable.pythonPath
                            p.port = tonumber(propertyTable.port) or Config.DEFAULT_PORT
                            propertyTable.bridgeStatus = "正在连接本地引擎…"
                            LrTasks.startAsyncTask(function()
                                local ok, info = Bridge.ensureRunning()
                                if ok then
                                    propertyTable.bridgeStatus = "运行中（版本 " .. tostring(info.version) .. " · PID " .. tostring(info.pid) .. "）"
                                else
                                    propertyTable.bridgeStatus = "失败：" .. tostring(info)
                                end
                            end)
                        end,
                    },
                    f:push_button {
                        title = "停止引擎",
                        action = function()
                            LrTasks.startAsyncTask(function()
                                Bridge.shutdown()
                                propertyTable.bridgeStatus = "已停止"
                            end)
                        end,
                    },
                },
                f:row {
                    f:static_text { title = "当前状态", width = 90, alignment = "right" },
                    f:static_text { title = bind "bridgeStatus", fill_horizontal = 1 },
                },
                f:static_text {
                    title = "快捷键提示：在 macOS「系统设置 → 键盘 → 键盘快捷键 → App 快捷键」中为 Adobe Lightroom Classic 添加菜单「FilmColor T24 负片转换…」，即可绑定如 Ctrl+Option+F 一键唤出。",
                    fill_horizontal = 1,
                    height_in_lines = 2,
                    font = "<system/small>",
                },
            },
        }
    end,
}
