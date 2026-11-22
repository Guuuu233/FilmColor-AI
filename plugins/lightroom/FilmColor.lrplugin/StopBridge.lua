local LrDialogs = import "LrDialogs"
local LrTasks = import "LrTasks"
local Bridge = require "Bridge"
local I18n = require "I18n"

LrTasks.startAsyncTask(function()
    Bridge.shutdown()
    LrDialogs.showBezel(I18n.t("stop_bezel"), 2.0)
end)
