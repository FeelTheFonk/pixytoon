--
-- json.lua — Minimal JSON encoder/decoder for Aseprite Lua
-- Pure Lua implementation, no dependencies.
-- Supports: objects, arrays, strings, numbers, booleans, null.
-- Hardened: depth limits + cycle detection prevent stack overflow.
--

local json = {}

-- Sentinel: assign to a table value to force encoding as JSON [] instead of {}.
-- Usage: local slots = json.EMPTY_ARRAY   →   encodes as []
-- Normal empty tables {} still encode as {} (safe for dict-typed fields).
local _EMPTY_ARRAY_MT = {}
json.EMPTY_ARRAY = setmetatable({}, _EMPTY_ARRAY_MT)

-- Hard limits — prevent stack overflow on circular or adversarial data.
local _MAX_ENCODE_DEPTH  = 128
local _MAX_DECODE_DEPTH  = 128
local _MAX_STRING_LENGTH = 50 * 1024 * 1024  -- 50 MB max decoded string

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
    -- Batch plain characters up to next special (O(n) instead of per-char)
    local next_special = s:find('[\\"]', j)
    if not next_special then error("Unterminated string") end
    if next_special > j then
      parts[#parts + 1] = s:sub(j, next_special - 1)
    end
    local c = s:sub(next_special, next_special)
    j = next_special
    if c == '"' then
      local result = table.concat(parts)
      if #result > _MAX_STRING_LENGTH then error("JSON string too long") end
      return result, j + 1
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
      j = j + 1
    end
  end
  error("Unterminated string")
end

local decode_value -- forward declaration

local function decode_array(s, i, depth)
  if depth > _MAX_DECODE_DEPTH then error("JSON nesting too deep") end
  local arr = {}
  i = skip_ws(s, i + 1)  -- skip [
  if s:sub(i, i) == ']' then return arr, i + 1 end
  while true do
    local val
    val, i = decode_value(s, i, depth)
    arr[#arr+1] = val
    i = skip_ws(s, i)
    local c = s:sub(i, i)
    if c == ']' then return arr, i + 1 end
    if c ~= ',' then error("Expected ',' in array at " .. i) end
    i = skip_ws(s, i + 1)
  end
end

local function decode_object(s, i, depth)
  if depth > _MAX_DECODE_DEPTH then error("JSON nesting too deep") end
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
    val, i = decode_value(s, i, depth)
    obj[key] = val
    i = skip_ws(s, i)
    local c = s:sub(i, i)
    if c == '}' then return obj, i + 1 end
    if c ~= ',' then error("Expected ',' in object at " .. i) end
    i = skip_ws(s, i + 1)
  end
end

decode_value = function(s, i, depth)
  depth = (depth or 0) + 1
  if depth > _MAX_DECODE_DEPTH then error("JSON nesting too deep") end
  i = skip_ws(s, i)
  local c = s:sub(i, i)
  if c == '"' then return decode_string(s, i)
  elseif c == '{' then return decode_object(s, i, depth)
  elseif c == '[' then return decode_array(s, i, depth)
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
  local val, _ = decode_value(s, 1, 0)
  return val
end

-- ─── ENCODE ──────────────────────────────────────────────────

local _ESCAPE_MAP = {
  ['"']  = '\\"',
  ['\\'] = '\\\\',
  ['\n'] = '\\n',
  ['\r'] = '\\r',
  ['\t'] = '\\t',
}
local function encode_string(s)
  -- Single-pass: escape known chars via lookup, then catch remaining control chars
  s = s:gsub('["\\\n\r\t]', _ESCAPE_MAP)
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

local function encode_array(arr, seen, depth)
  local parts = {}
  for i, v in ipairs(arr) do
    parts[i] = encode_value(v, seen, depth)
  end
  return '[' .. table.concat(parts, ',') .. ']'
end

local function encode_object(obj, seen, depth)
  local parts = {}
  for k, v in pairs(obj) do
    parts[#parts+1] = encode_string(tostring(k)) .. ':' .. encode_value(v, seen, depth)
  end
  return '{' .. table.concat(parts, ',') .. '}'
end

encode_value = function(v, seen, depth)
  depth = (depth or 0) + 1
  if depth > _MAX_ENCODE_DEPTH then return 'null' end

  local t = type(v)
  if v == nil then return 'null'
  elseif t == 'boolean' then return v and 'true' or 'false'
  elseif t == 'number' then
    -- Guard against NaN and Inf (invalid JSON)
    if v ~= v then return 'null' end
    if v == math.huge or v == -math.huge then return 'null' end
    return string.format("%.17g", v)
  elseif t == 'string' then return encode_string(v)
  elseif t == 'table' then
    -- Cycle detection: if we've already visited this exact table, emit null
    seen = seen or {}
    if seen[v] then return 'null' end
    seen[v] = true

    local result
    -- Explicit empty-array sentinel: always encode as []
    if getmetatable(v) == _EMPTY_ARRAY_MT then
      result = '[]'
    elseif is_array(v) then
      result = encode_array(v, seen, depth)
    else
      result = encode_object(v, seen, depth)
    end

    seen[v] = nil  -- allow same table in sibling branches (DAG-safe)
    return result
  else
    return 'null'
  end
end

function json.encode(v)
  return encode_value(v, nil, 0)
end

return json
