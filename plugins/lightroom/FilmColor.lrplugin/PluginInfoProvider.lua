local LrTasks = import "LrTasks"
local LrView = import "LrView"
local Bridge = require "Bridge"
local Config = require "Config"
local I18n = require "I18n"

local bind = LrView.bind
local T = I18n.t
local dialogOpen = false

return {
    startDialog = function(propertyTable)
        dialogOpen = true
        local p = Config.prefs()
        propertyTable.language = p.language or "auto"
        propertyTable.pythonPath = p.pythonPath
        propertyTable.port = tostring(p.port)
        propertyTable.bridgeStatus = Config.bundledBridgePath() and T("pi_bundled") or T("pi_click")
    end,

    endDialog = function(propertyTable)
        dialogOpen = false
        local p = Config.prefs()
        p.language = propertyTable.language or "auto"
        p.pythonPath = propertyTable.pythonPath
        local pn = tonumber(propertyTable.port)
        if pn and pn >= 1024 and pn <= 65535 then p.port = math.floor(pn) end
    end,

    sectionsForTopOfDialog = function(f, propertyTable)
        return {
            {
                title = T("pi_title"),
                bind_to_object = propertyTable,
                f:row {
                    f:static_text { title = T("pi_lang"), width = 95, alignment = "right" },
                    f:popup_menu {
                        value = bind "language",
                        width = 180,
                        items = I18n.LANG_ITEMS,
                    },
                },
                f:row {
                    f:static_text { title = T("pi_py"), width = 95, alignment = "right" },
                    f:edit_field { value = bind "pythonPath", fill_horizontal = 1 },
                },
                f:row {
                    f:static_text { title = T("pi_port"), width = 95, alignment = "right" },
                    f:edit_field { value = bind "port", width_in_digits = 6 },
                    f:push_button {
                        title = T("pi_check"),
                        action = function()
                            local p = Config.prefs()
                            p.language = propertyTable.language or "auto"
                            p.pythonPath = propertyTable.pythonPath
                            local pn = tonumber(propertyTable.port)
                            if pn and pn >= 1024 and pn <= 65535 then
                                p.port = math.floor(pn)
                            else
                                p.port = Config.DEFAULT_PORT
                                propertyTable.port = tostring(p.port)
                            end
                            propertyTable.bridgeStatus = T("pi_connecting")
                            LrTasks.startAsyncTask(function()
                                local ok, info = Bridge.ensureRunning()
                                if not dialogOpen then return end
                                if ok then
                                    propertyTable.bridgeStatus = T("pi_running", tostring(info.version), tostring(info.pid))
                                else
                                    propertyTable.bridgeStatus = T("pi_failed", tostring(info))
                                end
                            end)
                        end,
                    },
                    f:push_button {
                        title = T("pi_stop"),
                        action = function()
                            LrTasks.startAsyncTask(function()
                                Bridge.shutdown()
                                if dialogOpen then
                                    propertyTable.bridgeStatus = T("pi_stopped")
                                end
                            end)
                        end,
                    },
                },
                f:row {
                    f:static_text { title = T("pi_status"), width = 95, alignment = "right" },
                    f:static_text { title = bind "bridgeStatus", fill_horizontal = 1 },
                },
                f:static_text {
                    title = T("pi_shortcut"),
                    fill_horizontal = 1,
                    font = "<system/small>",
                },
            },
        }
    end,
}
