-- Minimal self-contained JSON encoder/decoder for Lightroom Classic Lua SDK (Lua 5.1)
local json = {}

local escape_char_map = {
    ["\\"] = "\\\\", ["\""] = "\\\"", ["\b"] = "\\b",
    ["\f"] = "\\f", ["\n"] = "\\n", ["\r"] = "\\r", ["\t"] = "\\t",
}

local function escape_char(c)
    return escape_char_map[c] or string.format("\\u%04x", c:byte())
end

local function encode_str(val)
    return '"' .. val:gsub('[%z\1-\31\\"]', escape_char) .. '"'
end

local function is_array(t)
    local n = 0
    for k, _ in pairs(t) do
        if type(k) ~= "number" or k <= 0 or math.floor(k) ~= k then
            return false
        end
        if k > n then n = k end
    end
    return n > 0 and n == #t
end

local function encode_val(val)
    local tv = type(val)
    if val == nil then return "null" end
    if tv == "boolean" then return val and "true" or "false" end
    if tv == "number" then
        if val ~= val or val == math.huge or val == -math.huge then return "null" end
        return tostring(val)
    end
    if tv == "string" then return encode_str(val) end
    if tv == "table" then
        if is_array(val) then
            local parts = {}
            for i = 1, #val do parts[i] = encode_val(val[i]) end
            return "[" .. table.concat(parts, ",") .. "]"
        else
            local parts = {}
            for k, v in pairs(val) do
                if type(k) == "string" then
                    parts[#parts + 1] = encode_str(k) .. ":" .. encode_val(v)
                end
            end
            return "{" .. table.concat(parts, ",") .. "}"
        end
    end
    return "null"
end

function json.encode(val)
    return encode_val(val)
end

local function codepoint_to_utf8(n)
    if n <= 0x7f then return string.char(n)
    elseif n <= 0x7ff then return string.char(math.floor(n / 64) + 192, (n % 64) + 128)
    elseif n <= 0xffff then
        return string.char(math.floor(n / 4096) + 224, math.floor((n % 4096) / 64) + 128, (n % 64) + 128)
    else
        return string.char(
            math.floor(n / 262144) + 240,
            math.floor((n % 262144) / 4096) + 128,
            math.floor((n % 4096) / 64) + 128,
            (n % 64) + 128
        )
    end
end

function json.decode(str)
    if type(str) ~= "string" or str == "" then return nil, "empty json" end
    local pos = 1
    local len = #str

    local function skip_ws()
        while pos <= len do
            local b = str:byte(pos)
            if b == 32 or b == 9 or b == 10 or b == 13 then pos = pos + 1 else break end
        end
    end

    local parse_value

    local function parse_string()
        pos = pos + 1 -- skip opening quote
        local out = {}
        while pos <= len do
            local c = str:sub(pos, pos)
            if c == '"' then
                pos = pos + 1
                return table.concat(out)
            elseif c == "\\" then
                pos = pos + 1
                local esc = str:sub(pos, pos)
                if esc == '"' or esc == "\\" or esc == "/" then out[#out + 1] = esc; pos = pos + 1
                elseif esc == "b" then out[#out + 1] = "\b"; pos = pos + 1
                elseif esc == "f" then out[#out + 1] = "\f"; pos = pos + 1
                elseif esc == "n" then out[#out + 1] = "\n"; pos = pos + 1
                elseif esc == "r" then out[#out + 1] = "\r"; pos = pos + 1
                elseif esc == "t" then out[#out + 1] = "\t"; pos = pos + 1
                elseif esc == "u" then
                    local hex = str:sub(pos + 1, pos + 4)
                    local cp = tonumber(hex, 16) or 63
                    pos = pos + 5
                    if cp >= 0xD800 and cp <= 0xDBFF and str:sub(pos, pos + 1) == "\\u" then
                        local hex2 = str:sub(pos + 2, pos + 5)
                        local cp2 = tonumber(hex2, 16)
                        if cp2 and cp2 >= 0xDC00 and cp2 <= 0xDFFF then
                            cp = 0x10000 + (cp - 0xD800) * 1024 + (cp2 - 0xDC00)
                            pos = pos + 6
                        end
                    end
                    out[#out + 1] = codepoint_to_utf8(cp)
                else
                    out[#out + 1] = esc; pos = pos + 1
                end
            else
                local next_special = str:find('[\\"]', pos)
                if not next_special then error("unterminated string") end
                out[#out + 1] = str:sub(pos, next_special - 1)
                pos = next_special
            end
        end
        error("unterminated string")
    end

    local function parse_number()
        local s, e = str:find("^-?%d+%.?%d*[eE]?[+-]?%d*", pos)
        if not s then error("invalid number at " .. pos) end
        pos = e + 1
        return tonumber(str:sub(s, e))
    end

    local function parse_array()
        pos = pos + 1
        local arr = {}
        skip_ws()
        if str:sub(pos, pos) == "]" then pos = pos + 1; return arr end
        while true do
            skip_ws()
            arr[#arr + 1] = parse_value()
            skip_ws()
            local c = str:sub(pos, pos)
            if c == "]" then pos = pos + 1; return arr
            elseif c == "," then pos = pos + 1
            else error("expected , or ] at " .. pos) end
        end
    end

    local function parse_object()
        pos = pos + 1
        local obj = {}
        skip_ws()
        if str:sub(pos, pos) == "}" then pos = pos + 1; return obj end
        while true do
            skip_ws()
            if str:sub(pos, pos) ~= '"' then error("expected string key at " .. pos) end
            local k = parse_string()
            skip_ws()
            if str:sub(pos, pos) ~= ":" then error("expected : at " .. pos) end
            pos = pos + 1
            skip_ws()
            obj[k] = parse_value()
            skip_ws()
            local c = str:sub(pos, pos)
            if c == "}" then pos = pos + 1; return obj
            elseif c == "," then pos = pos + 1
            else error("expected , or } at " .. pos) end
        end
    end

    parse_value = function()
        skip_ws()
        local c = str:sub(pos, pos)
        if c == '"' then return parse_string()
        elseif c == "{" then return parse_object()
        elseif c == "[" then return parse_array()
        elseif c == "t" and str:sub(pos, pos + 3) == "true" then pos = pos + 4; return true
        elseif c == "f" and str:sub(pos, pos + 4) == "false" then pos = pos + 5; return false
        elseif c == "n" and str:sub(pos, pos + 3) == "null" then pos = pos + 4; return nil
        else return parse_number() end
    end

    local ok, res = pcall(parse_value)
    if not ok then return nil, res end
    return res
end

return json
