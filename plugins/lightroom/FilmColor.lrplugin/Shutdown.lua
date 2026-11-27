local LrTasks = import "LrTasks"
local Bridge = require "Bridge"

return {
    LrShutdownFunction = function(doneFunction, _progressFunction)
        LrTasks.startAsyncTask(function()
            LrTasks.pcall(Bridge.shutdown)
            if doneFunction then doneFunction() end
        end)
    end,
}
