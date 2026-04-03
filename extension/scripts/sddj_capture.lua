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
  local large_source = (spr.width > 2048 or spr.height > 2048)
  if large_source then
    PT.update_status("Large source — server will resize to fit 2048x2048")
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
    local sel_bounds = sel.bounds
    local sb_x, sb_y = sel_bounds.x, sel_bounds.y
    local sb_w, sb_h = sel_bounds.width, sel_bounds.height
    local sb_x2, sb_y2 = sb_x + sb_w, sb_y + sb_h
    -- Fast-path: if selection is a simple rectangle, skip per-pixel contains() checks
    local is_rect_selection = (sb_w * sb_h == sel:count())
    if is_rect_selection then
      -- Pure bounds check — no sel:contains() needed
      local black_row = string.rep(black, mw)
      for y = 0, mh - 1 do
        if y >= sb_y and y < sb_y2 then
          local pre = sb_x > 0 and string.rep(black, sb_x) or ""
          local mid = string.rep(white, sb_w)
          local post_count = mw - sb_x2
          local post = post_count > 0 and string.rep(black, post_count) or ""
          mask_buf[#mask_buf + 1] = pre .. mid .. post
        else
          mask_buf[#mask_buf + 1] = black_row
        end
      end
    else
      -- Complex selection (lasso, magic wand, etc.) — per-pixel contains()
      for y = 0, mh - 1 do
        local in_sel_y = (y >= sb_y and y < sb_y2)
        for x = 0, mw - 1 do
          if in_sel_y and x >= sb_x and x < sb_x2 and sel:contains(x, y) then
            mask_buf[#mask_buf + 1] = white
          else
            mask_buf[#mask_buf + 1] = black
          end
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
        local zero_row = string.rep(zero, mask_w)
        -- Build mask row-by-row using per-row string.byte range extraction
        for my = 0, mask_h - 1 do
          local src_y = my - oy
          if src_y >= 0 and src_y < h then
            -- Determine which source columns overlap this mask row
            local src_x_start = math.max(0, -ox)
            local src_x_end = math.min(w - 1, mask_w - 1 - ox)
            if src_x_start > src_x_end then
              mask_data[#mask_data + 1] = zero_row
            else
              local row_offset = src_y * w * 4
              -- Extract all RGBA bytes for this source row range at once
              -- We need alpha bytes at offsets +4, +8, +12... (every 4th byte)
              -- Extract the entire pixel range and pick alpha from the table
              local range_start = row_offset + src_x_start * 4 + 1
              local range_end = row_offset + src_x_end * 4 + 4
              local row_bytes = {byte(src_bytes, range_start, range_end)}
              local row_parts = {}
              local last_x = 0
              for src_x = src_x_start, src_x_end do
                local mx = src_x + ox
                -- Alpha is at position 4 within each 4-byte RGBA pixel
                local idx = (src_x - src_x_start) * 4 + 4
                local a = row_bytes[idx]
                if a and a > 0 then
                  if mx > last_x then
                    row_parts[#row_parts + 1] = string.rep(zero, mx - last_x)
                  end
                  row_parts[#row_parts + 1] = white_ch
                  last_x = mx + 1
                end
              end
              if last_x < mask_w then
                row_parts[#row_parts + 1] = string.rep(zero, mask_w - last_x)
              end
              mask_data[#mask_data + 1] = table.concat(row_parts)
            end
          else
            mask_data[#mask_data + 1] = zero_row
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
      -- Non-RGB modes: render through sprite compositing to RGB, then check alpha.
      -- Cold path: only hit for Indexed/Grayscale sprites which are rare in the
      -- SD workflow. Pixel-by-pixel getPixel/drawPixel is acceptable here since
      -- the bulk-bytes fast path above handles the common RGB case.
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
