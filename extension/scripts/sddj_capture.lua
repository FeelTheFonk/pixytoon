--
-- SDDj — Image Capture Functions
--

return function(PT)

function PT.capture_active_layer()
  local spr = app.sprite
  if spr == nil then return nil end
  local max_cap = PT.cfg.MAX_CAPTURE_SIZE or 4096
  if spr.width > max_cap or spr.height > max_cap then
    PT.update_status("Capture rejected: sprite exceeds " .. max_cap .. "px limit")
    return nil
  end
  if spr.width > 2048 or spr.height > 2048 then
    PT.update_status("Warning: large sprite — capture may be slow")
  end
  local cel = app.cel
  if cel == nil or cel.image == nil then return nil end
  local full = Image(spr.spec)
  full:clear()
  full:drawImage(cel.image, cel.position)
  return PT.image_to_base64(full)
end

function PT.capture_flattened()
  local spr = app.sprite
  if spr == nil or app.frame == nil then return nil end
  local flat_img = Image(spr.spec)
  flat_img:drawSprite(spr, app.frame)
  return PT.image_to_base64(flat_img)
end

-- Capture inpainting mask with priority:
--   1) Active selection (marquee/lasso) → white where selected
--   2) Layer named "Mask"/"mask"        → grayscale content
--   3) Active layer alpha channel       → white where alpha > 0
function PT.capture_mask()
  local spr = app.sprite
  if spr == nil or app.frame == nil then return nil end

  -- Strategy A: active selection (bytes-based for performance)
  local sel = spr.selection
  if sel and not sel.isEmpty then
    local mw, mh = spr.width, spr.height
    local mask_img = Image(mw, mh, ColorMode.GRAY)
    mask_img:clear(Color{ gray = 0 })
    local mask_buf = {}
    local white = string.char(255, 255)  -- GrayA: gray=255, alpha=255
    local black = string.char(0, 255)    -- GrayA: gray=0,   alpha=255
    for y = 0, mh - 1 do
      local in_sel_y = (y >= sel.bounds.y and y < sel.bounds.y + sel.bounds.height)
      for x = 0, mw - 1 do
        if in_sel_y and x >= sel.bounds.x and x < sel.bounds.x + sel.bounds.width and sel:contains(x, y) then
          mask_buf[#mask_buf + 1] = white
        else
          mask_buf[#mask_buf + 1] = black
        end
      end
    end
    mask_img.bytes = table.concat(mask_buf)
    return PT.image_to_base64(mask_img)
  end

  -- Strategy B: "Mask" layer (recursive group search, depth-limited)
  local function find_mask_layer(layers, depth)
    if depth > 16 then return nil end
    for _, layer in ipairs(layers) do
      if layer.name == "Mask" or layer.name == "mask" then
        local cel = layer:cel(app.frame)
        if cel and cel.image then
          local full = Image(spr.width, spr.height, ColorMode.GRAY)
          full:clear(Color{ gray = 0 })
          full:drawImage(cel.image, cel.position)
          return PT.image_to_base64(full)
        end
      end
      if layer.isGroup and layer.layers then
        local result = find_mask_layer(layer.layers, depth + 1)
        if result then return result end
      end
    end
    return nil
  end
  local mask_b64 = find_mask_layer(spr.layers, 0)
  if mask_b64 then return mask_b64 end

  -- Strategy C: auto from active layer alpha
  local cel = app.cel
  if cel and cel.image then
    local img = cel.image
    local mask_w, mask_h = spr.width, spr.height
    local mask_img = Image(mask_w, mask_h, ColorMode.GRAY)
    mask_img:clear(Color{ gray = 0 })
    local ox, oy = cel.position.x, cel.position.y
    if spr.colorMode == ColorMode.RGB then
      -- Bulk alpha extraction via Image.bytes (RGBA src → GRAY mask)
      local ok_bytes, src_bytes = pcall(function() return img.bytes end)
      if ok_bytes and src_bytes then
        local w, h = img.width, img.height
        local byte = string.byte
        -- Pre-fill mask buffer with zeroes via string.rep (avoids 4M-entry table)
        local zero = string.char(0, 255)     -- GrayA: gray=0, alpha=255
        local white_ch = string.char(255, 255) -- GrayA: gray=255, alpha=255
        local mask_data = {}
        -- Collect only non-zero pixel positions, build mask row-by-row
        for my = 0, mask_h - 1 do
          local row_parts = {}
          local src_y = my - oy
          if src_y >= 0 and src_y < h then
            local row_offset = src_y * w * 4
            local last_x = 0
            for mx = 0, mask_w - 1 do
              local src_x = mx - ox
              if src_x >= 0 and src_x < w then
                local a = byte(src_bytes, row_offset + src_x * 4 + 4)
                if a and a > 0 then
                  if mx > last_x then
                    row_parts[#row_parts + 1] = string.rep(zero, mx - last_x)
                  end
                  row_parts[#row_parts + 1] = white_ch
                  last_x = mx + 1
                end
              end
            end
            if last_x < mask_w then
              row_parts[#row_parts + 1] = string.rep(zero, mask_w - last_x)
            end
            mask_data[#mask_data + 1] = table.concat(row_parts)
          else
            mask_data[#mask_data + 1] = string.rep(zero, mask_w)
          end
        end
        mask_img.bytes = table.concat(mask_data)
      else
        -- Fallback: Image.bytes not available, use pixel-by-pixel
        for y = 0, img.height - 1 do
          for x = 0, img.width - 1 do
            local px = img:getPixel(x, y)
            local a = app.pixelColor.rgbaA(px)
            if a > 0 then
              local sx, sy = ox + x, oy + y
              if sx >= 0 and sx < mask_w and sy >= 0 and sy < mask_h then
                mask_img:drawPixel(sx, sy, Color{ gray = 255 })
              end
            end
          end
        end
      end
    else
      -- Non-RGB modes: render through sprite compositing to RGB, then check alpha
      local tmp_img = Image(mask_w, mask_h, ColorMode.RGB)
      tmp_img:clear()
      tmp_img:drawImage(cel.image, cel.position)
      for y = 0, mask_h - 1 do
        for x = 0, mask_w - 1 do
          local px = tmp_img:getPixel(x, y)
          local a = app.pixelColor.rgbaA(px)
          if a > 0 then
            mask_img:drawPixel(x, y, Color{ gray = 255 })
          end
        end
      end
    end
    return PT.image_to_base64(mask_img)
  end

  return nil
end

end
