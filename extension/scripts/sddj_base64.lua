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

-- Pre-computed power-of-two table to avoid repeated 2^n calls
local _pow2 = {}
for i = 0, 24 do _pow2[i] = 2 ^ i end

-- ─── Encode (O(n) via table.concat + LUT) ──────────────────

function PT.base64_encode(data)
  local t, n = {}, 0
  local acc, bits = 0, 0
  for i = 1, #data do
    acc = acc * 256 + data:byte(i)
    bits = bits + 8
    while bits >= 6 do
      bits = bits - 6
      n = n + 1
      t[n] = b64chars:sub(math.floor(acc / _pow2[bits]) % 64 + 1, math.floor(acc / _pow2[bits]) % 64 + 1)
      acc = acc % _pow2[bits]
    end
  end
  if bits > 0 then
    n = n + 1
    local shifted = acc * _pow2[6 - bits]
    t[n] = b64chars:sub(shifted % 64 + 1, shifted % 64 + 1)
  end
  local pad = (3 - #data % 3) % 3
  for _ = 1, pad do n = n + 1; t[n] = "=" end
  return table.concat(t)
end

-- ─── Decode (lookup table — O(1) per character) ─────────────
-- Previous: string:find() per character = O(32) per lookup
-- Now: byte-indexed lookup table = O(1) per lookup
-- + table.concat instead of string concatenation (reduces GC pressure)
-- + math.floor(acc / 2^bits) for O(1) byte extraction (no inner loop)

function PT.base64_decode(data)
  if #data > (PT.cfg and PT.cfg.MAX_BASE64_SIZE or 104857600) then
    return nil
  end
  local t, n = {}, 0
  local acc, bits = 0, 0
  for i = 1, #data do
    local v = _b64_lut[data:byte(i)]
    if v then
      acc = acc * 64 + v
      bits = bits + 6
      if bits >= 8 then
        bits = bits - 8
        n = n + 1
        t[n] = string.char(math.floor(acc / _pow2[bits]) % 256)
        acc = acc % _pow2[bits]
      end
    end
  end
  return table.concat(t)
end

end
