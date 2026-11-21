local LrTasks = import "LrTasks"
local Bridge = require "Bridge"

return {
    LrShutdownFunction = function(doneFunction)
        LrTasks.startAsyncTask(function()
            pcall(Bridge.shutdown)
            doneFunction()
        end)
    end,
}
