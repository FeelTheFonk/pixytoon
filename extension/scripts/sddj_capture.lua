--
-- SDDj — Image Capture Functions
--

return function(PT)

function PT.capture_active_layer()
  local spr = app.sprite
  if spr == nil then return nil end
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

  -- Strategy A: active selection
  local sel = spr.selection
  if sel and not sel.isEmpty then
    local mask_img = Image(spr.width, spr.height, ColorMode.GRAY)
    mask_img:clear(Color{ gray = 0 })
    for y = sel.bounds.y, sel.bounds.y + sel.bounds.height - 1 do
      for x = sel.bounds.x, sel.bounds.x + sel.bounds.width - 1 do
        if sel:contains(x, y) then
          mask_img:drawPixel(x, y, Color{ gray = 255 })
        end
      end
    end
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
    local mask_img = Image(spr.width, spr.height, ColorMode.GRAY)
    mask_img:clear(Color{ gray = 0 })
    local ox, oy = cel.position.x, cel.position.y
    if spr.colorMode == ColorMode.RGB then
      for y = 0, img.height - 1 do
        for x = 0, img.width - 1 do
          local px = img:getPixel(x, y)
          local a = app.pixelColor.rgbaA(px)
          if a > 0 then
            local sx, sy = ox + x, oy + y
            if sx >= 0 and sx < spr.width and sy >= 0 and sy < spr.height then
              mask_img:drawPixel(sx, sy, Color{ gray = 255 })
            end
          end
        end
      end
    else
      -- Non-RGB modes: render through sprite compositing to RGB, then check alpha
      local tmp_img = Image(spr.width, spr.height, ColorMode.RGB)
      tmp_img:clear()
      tmp_img:drawImage(cel.image, cel.position)
      for y = 0, spr.height - 1 do
        for x = 0, spr.width - 1 do
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
