--
-- json.lua — Minimal JSON encoder/decoder for Aseprite Lua
-- Pure Lua implementation, no dependencies.
-- Supports: objects, arrays, strings, numbers, booleans, null.
--

local json = {}

-- ─── DECODE ──────────────────────────────────────────────────

local function skip_ws(s, i)
  local _, e = s:find("^%s*", i)
  return e + 1
end

local function decode_string(s, i)
  -- i points to opening "
  local j = i + 1
  local parts = {}
  while j <= #s do
    local c = s:sub(j, j)
    if c == '"' then
      return table.concat(parts), j + 1
    elseif c == '\\' then
      j = j + 1
      local esc = s:sub(j, j)
      if esc == 'n' then parts[#parts+1] = '\n'
      elseif esc == 't' then parts[#parts+1] = '\t'
      elseif esc == 'r' then parts[#parts+1] = '\r'
      elseif esc == 'b' then parts[#parts+1] = '\b'
      elseif esc == 'f' then parts[#parts+1] = '\f'
      elseif esc == '\\' then parts[#parts+1] = '\\'
      elseif esc == '"' then parts[#parts+1] = '"'
      elseif esc == '/' then parts[#parts+1] = '/'
      elseif esc == 'u' then
        local hex = s:sub(j+1, j+4)
        local cp = tonumber(hex, 16)
        j = j + 4
        -- Handle UTF-16 surrogate pairs (emoji / astral plane)
        if cp >= 0xD800 and cp <= 0xDBFF then
          -- High surrogate — expect \uXXXX low surrogate
          if s:sub(j+1, j+2) == '\\u' then
            local low_hex = s:sub(j+3, j+6)
            local low = tonumber(low_hex, 16)
            if low and low >= 0xDC00 and low <= 0xDFFF then
              cp = 0x10000 + (cp - 0xD800) * 0x400 + (low - 0xDC00)
              j = j + 6
            end
          end
        end
        -- Encode codepoint as UTF-8
        if cp < 0x80 then
          parts[#parts+1] = string.char(cp)
        elseif cp < 0x800 then
          parts[#parts+1] = string.char(
            0xC0 + math.floor(cp / 64),
            0x80 + (cp % 64)
          )
        elseif cp < 0x10000 then
          parts[#parts+1] = string.char(
            0xE0 + math.floor(cp / 4096),
            0x80 + math.floor((cp % 4096) / 64),
            0x80 + (cp % 64)
          )
        else
          -- 4-byte UTF-8 (astral plane)
          parts[#parts+1] = string.char(
            0xF0 + math.floor(cp / 262144),
            0x80 + math.floor((cp % 262144) / 4096),
            0x80 + math.floor((cp % 4096) / 64),
            0x80 + (cp % 64)
          )
        end
      end
    else
      parts[#parts+1] = c
    end
    j = j + 1
  end
  error("Unterminated string")
end

local decode_value -- forward declaration

local function decode_array(s, i)
  local arr = {}
  i = skip_ws(s, i + 1)  -- skip [
  if s:sub(i, i) == ']' then return arr, i + 1 end
  while true do
    local val
    val, i = decode_value(s, i)
    arr[#arr+1] = val
    i = skip_ws(s, i)
    local c = s:sub(i, i)
    if c == ']' then return arr, i + 1 end
    if c ~= ',' then error("Expected ',' in array at " .. i) end
    i = skip_ws(s, i + 1)
  end
end

local function decode_object(s, i)
  local obj = {}
  i = skip_ws(s, i + 1)  -- skip {
  if s:sub(i, i) == '}' then return obj, i + 1 end
  while true do
    -- key
    i = skip_ws(s, i)
    if s:sub(i, i) ~= '"' then error("Expected key string at " .. i) end
    local key
    key, i = decode_string(s, i)
    i = skip_ws(s, i)
    if s:sub(i, i) ~= ':' then error("Expected ':' at " .. i) end
    i = skip_ws(s, i + 1)
    -- value
    local val
    val, i = decode_value(s, i)
    obj[key] = val
    i = skip_ws(s, i)
    local c = s:sub(i, i)
    if c == '}' then return obj, i + 1 end
    if c ~= ',' then error("Expected ',' in object at " .. i) end
    i = skip_ws(s, i + 1)
  end
end

decode_value = function(s, i)
  i = skip_ws(s, i)
  local c = s:sub(i, i)
  if c == '"' then return decode_string(s, i)
  elseif c == '{' then return decode_object(s, i)
  elseif c == '[' then return decode_array(s, i)
  elseif c == 't' then
    assert(s:sub(i, i+3) == "true", "Invalid literal at " .. i)
    return true, i + 4
  elseif c == 'f' then
    assert(s:sub(i, i+4) == "false", "Invalid literal at " .. i)
    return false, i + 5
  elseif c == 'n' then
    assert(s:sub(i, i+3) == "null", "Invalid literal at " .. i)
    return nil, i + 4
  else
    -- number
    local _, e, num = s:find("^(-?%d+%.?%d*)", i)
    if num then
      -- Check for exponent part (requires digits after e/E)
      local _, e2, exp = s:find("^([eE][+-]?%d+)", e + 1)
      if exp then
        num = num .. exp
        e = e2
      end
    end
    if not num then error("Invalid number at " .. i) end
    return tonumber(num), e + 1
  end
end

function json.decode(s)
  local val, _ = decode_value(s, 1)
  return val
end

-- ─── ENCODE ──────────────────────────────────────────────────

local function encode_string(s)
  s = s:gsub('\\', '\\\\')
  s = s:gsub('"', '\\"')
  s = s:gsub('\n', '\\n')
  s = s:gsub('\r', '\\r')
  s = s:gsub('\t', '\\t')
  s = s:gsub('[%z\1-\31]', function(c)
    return string.format('\\u%04x', c:byte())
  end)
  return '"' .. s .. '"'
end

local encode_value -- forward

local function is_array(t)
  local count = 0
  for _ in pairs(t) do
    count = count + 1
  end
  if count == 0 then return false end  -- empty table → encode as {}
  -- A table is an array if it has contiguous integer keys 1..count
  return t[1] ~= nil and t[count] ~= nil
end

local function encode_array(arr)
  local parts = {}
  for i, v in ipairs(arr) do
    parts[i] = encode_value(v)
  end
  return '[' .. table.concat(parts, ',') .. ']'
end

local function encode_object(obj)
  local parts = {}
  for k, v in pairs(obj) do
    parts[#parts+1] = encode_string(tostring(k)) .. ':' .. encode_value(v)
  end
  return '{' .. table.concat(parts, ',') .. '}'
end

encode_value = function(v)
  local t = type(v)
  if v == nil then return 'null'
  elseif t == 'boolean' then return v and 'true' or 'false'
  elseif t == 'number' then
    -- Guard against NaN and Inf (invalid JSON)
    if v ~= v then return 'null' end
    if v == math.huge or v == -math.huge then return 'null' end
    return tostring(v)
  elseif t == 'string' then return encode_string(v)
  elseif t == 'table' then
    if is_array(v) then return encode_array(v)
    else return encode_object(v) end
  else
    return 'null'
  end
end

function json.encode(v)
  return encode_value(v)
end

return json
