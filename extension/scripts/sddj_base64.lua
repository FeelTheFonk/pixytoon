--
-- SDDj — Base64 Codec (optimized)
--

return function(PT)

local b64chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"

-- ─── Lookup Tables (O(1) encode/decode) ────────────────────

local _b64_lut = {}
for i = 1, #b64chars do
  _b64_lut[b64chars:byte(i)] = i - 1
end

local _b64_chars = {}
for i = 1, #b64chars do _b64_chars[i] = b64chars:sub(i, i) end

-- ─── Encode (O(n) via table.concat + LUT + bitwise ops) ────

function PT.base64_encode(data)
  local t, n = {}, 0
  local acc, bits = 0, 0
  for i = 1, #data do
    acc = (acc << 8) | data:byte(i)
    bits = bits + 8
    while bits >= 6 do
      bits = bits - 6
      n = n + 1
      t[n] = _b64_chars[((acc >> bits) & 63) + 1]
      acc = acc & ((1 << bits) - 1)
    end
  end
  if bits > 0 then
    n = n + 1
    local shifted = acc << (6 - bits)
    t[n] = _b64_chars[(shifted & 63) + 1]
  end
  local pad = (3 - #data % 3) % 3
  for _ = 1, pad do n = n + 1; t[n] = "=" end
  return table.concat(t)
end

-- ─── Decode (LUT + bitwise ops — O(1) per character) ────────
-- byte-indexed lookup table = O(1) per lookup
-- + table.concat (reduces GC pressure)
-- + bitwise shift/mask for O(1) byte extraction (no FPU)

function PT.base64_decode(data)
  if #data > PT.cfg.MAX_BASE64_SIZE then
    return nil
  end
  local t, n = {}, 0
  local acc, bits = 0, 0
  for i = 1, #data do
    local v = _b64_lut[data:byte(i)]
    if v then
      acc = (acc << 6) | v
      bits = bits + 6
      if bits >= 8 then
        bits = bits - 8
        n = n + 1
        t[n] = string.char((acc >> bits) & 255)
        acc = acc & ((1 << bits) - 1)
      end
    end
  end
  return table.concat(t)
end

end
