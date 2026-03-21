--
-- PixyToon — Image Import (result, animation frame, live preview)
--

return function(PT)

function PT.import_result(resp)
  local img_data = PT.base64_decode(resp.image)
  local tmp = PT.make_tmp_path("res")

  local ok, err = pcall(function()
    local f = io.open(tmp, "wb")
    if not f then error("Failed to create temp file") end
    f:write(img_data)
    f:close()

    local spr = app.sprite
    if spr == nil then
      spr = Sprite(resp.width or 512, resp.height or 512, ColorMode.RGB)
    end

    local img = Image{ fromFile = tmp }
    os.remove(tmp)
    tmp = nil  -- prevent double-remove in error handler

    if img then
      app.transaction("PixyToon Generate", function()
        local layer = spr:newLayer()
        layer.name = "PixyToon #" .. tostring(resp.seed or "?")
        spr:newCel(layer, app.frame, img, Point(0, 0))
      end)
    end

    app.refresh()
  end)
  if not ok then
    if tmp then pcall(os.remove, tmp) end
    PT.update_status("Import error: " .. tostring(err))
  end
end

function PT.import_animation_frame(resp)
  if not PT.state.animating then return end
  if resp.frame_index ~= 0 and PT.anim.layer == nil then return end

  local img_data = PT.base64_decode(resp.image)
  local tmp = PT.make_tmp_path("anim")

  local ok, err = pcall(function()
    local f = io.open(tmp, "wb")
    if not f then error("Failed to open temp file for writing") end
    f:write(img_data)
    f:close()

    local spr = app.sprite
    local created_sprite = false
    if spr == nil then
      spr = Sprite(resp.width or 512, resp.height or 512, ColorMode.RGB)
      created_sprite = true
    end

    local img = Image{ fromFile = tmp }
    os.remove(tmp)
    tmp = nil

    app.transaction("PixyToon Frame " .. (resp.frame_index + 1), function()
      -- First frame: create layer and anchor position
      if resp.frame_index == 0 then
        PT.anim.layer = spr:newLayer()
        PT.anim.layer.name = "PixyToon Anim #" .. tostring(resp.seed or "?")
        PT.anim.base_seed = resp.seed or 0
        PT.anim.frame_count = 0
        if created_sprite then
          PT.anim.start_frame = 1
        else
          PT.anim.start_frame = #spr.frames + 1
        end
      end

      -- Determine frame position
      local frame_num
      if resp.frame_index == 0 and created_sprite then
        frame_num = 1
      else
        local target_pos = PT.anim.start_frame + resp.frame_index
        target_pos = math.min(target_pos, #spr.frames + 1)
        local new_frame = spr:newEmptyFrame(target_pos)
        frame_num = new_frame.frameNumber
      end

      -- Validate layer still exists in sprite before creating cel
      local layer_valid = false
      if img and PT.anim.layer and spr.frames[frame_num] then
        for _, layer in ipairs(spr.layers) do
          if layer == PT.anim.layer then layer_valid = true; break end
        end
      end
      if layer_valid then
        spr:newCel(PT.anim.layer, spr.frames[frame_num], img, Point(0, 0))
      end
    end)

    PT.anim.frame_count = PT.anim.frame_count + 1
    app.refresh()

    if PT.dlg then
      PT.update_status("Frame " .. (resp.frame_index + 1) .. "/" .. tostring(resp.total_frames or "?")
        .. " (" .. tostring(resp.time_ms or "?") .. "ms)")
    end
  end)
  if not ok then
    if tmp then pcall(os.remove, tmp) end
    PT.update_status("Import error: " .. tostring(err))
  end
end

function PT.live_update_preview(resp)
  PT.live.importing = true  -- Guard: prevent sprite change event from re-triggering

  local spr = app.sprite
  if spr == nil or app.frame == nil then
    PT.live.importing = false
    return
  end

  -- Validate or find/create preview layer
  local need_new = PT.live.preview_layer == nil
      or PT.live.preview_sprite ~= spr
      or not pcall(function() return PT.live.preview_layer.name end)
  if not need_new then
    -- Verify the layer still exists in the sprite
    local found = false
    for _, layer in ipairs(spr.layers) do
      if layer == PT.live.preview_layer then found = true; break end
    end
    if not found then need_new = true end
  end

  if need_new then
    PT.live.preview_layer = nil
    for _, layer in ipairs(spr.layers) do
      if layer.name == "_pixytoon_live" then
        PT.live.preview_layer = layer
        break
      end
    end
    if PT.live.preview_layer == nil then
      PT.live.preview_layer = spr:newLayer()
      PT.live.preview_layer.name = "_pixytoon_live"
    end
    PT.live.preview_sprite = spr
  end

  local img_data = PT.base64_decode(resp.image)
  local tmp = PT.make_tmp_path("live")
  local f = io.open(tmp, "wb")
  if not f then PT.live.importing = false; return end
  f:write(img_data)
  f:close()

  local ok_img, img = pcall(function() return Image{ fromFile = tmp } end)
  os.remove(tmp)
  if not ok_img or not img then
    PT.update_status("Preview load failed")
    PT.live.importing = false
    return
  end

  -- Update existing cel in-place (avoids layer churn)
  local ok_cel, cel = pcall(function() return PT.live.preview_layer:cel(app.frame) end)
  if not ok_cel then
    PT.live.preview_layer = nil
    PT.live.importing = false
    return
  end
  if cel then
    cel.image = img
    cel.position = Point(0, 0)
  else
    spr:newCel(PT.live.preview_layer, app.frame, img, Point(0, 0))
  end

  -- Apply opacity
  if PT.dlg then
    PT.live.preview_layer.opacity = math.floor(PT.dlg.data.live_opacity * 255 / 100)
  end

  app.refresh()

  PT.live.importing = false

  -- If a change was pending during import, send now
  if PT.live.pending_send and not PT.live.request_inflight then
    PT.live_send_now()
  end
end

end
