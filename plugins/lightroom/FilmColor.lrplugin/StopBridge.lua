local LrDialogs = import "LrDialogs"
local LrTasks = import "LrTasks"
local Bridge = require "Bridge"

LrTasks.startAsyncTask(function()
    Bridge.shutdown()
    LrDialogs.showBezel("FilmColor 后台引擎已停止", 2.0)
end)
