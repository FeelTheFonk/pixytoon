--
-- SDDj — Image Import (result, animation frame)
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
      app.activeSprite = spr
    end

    local img = Image{ fromFile = tmp }
    os.remove(tmp)
    tmp = nil  -- prevent double-remove in error handler

    if img then
      app.transaction("SDDj Generate", function()
        local layer = spr:newLayer()
        layer.name = "SDDj #" .. tostring(resp.seed or "?")
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

-- ─── Sequence Mode: place each result as a new frame ───────

function PT.reset_sequence()
  PT.seq.layer = nil
  PT.seq.start_frame = 0
  PT.seq.frame_count = 0
  PT.seq.active = false
end

function PT.finalize_sequence()
  if not PT.seq.active or PT.seq.frame_count == 0 then
    PT.reset_sequence()
    return
  end
  local spr = app.sprite
  if spr and PT.seq.frame_count > 0 then
    pcall(function()
      app.transaction("SDDj Sequence Finalize", function()
        local dur = 0.1  -- 100ms default
        for i = 0, PT.seq.frame_count - 1 do
          local fn = PT.seq.start_frame + i
          if spr.frames[fn] then spr.frames[fn].duration = dur end
        end
      end)
    end)
  end
  PT.reset_sequence()
end

function PT.import_result_as_frame(resp)
  local img_data = PT.base64_decode(resp.image)
  local tmp = PT.make_tmp_path("seq")

  local ok, err = pcall(function()
    local f = io.open(tmp, "wb")
    if not f then error("Failed to create temp file") end
    f:write(img_data)
    f:close()

    local spr = app.sprite
    local created_sprite = false
    if spr == nil then
      spr = Sprite(resp.width or 512, resp.height or 512, ColorMode.RGB)
      app.activeSprite = spr
      created_sprite = true
    end

    local img = Image{ fromFile = tmp }
    os.remove(tmp)
    tmp = nil

    app.transaction("SDDj Seq Frame " .. (PT.seq.frame_count + 1), function()
      -- First frame in sequence: create layer and anchor
      if PT.seq.layer == nil then
        PT.seq.layer = spr:newLayer()
        PT.seq.layer.name = "SDDj Seq #" .. tostring(resp.seed or "?")
        PT.seq.active = true
        PT.seq.frame_count = 0
        if created_sprite then
          PT.seq.start_frame = 1
        else
          PT.seq.start_frame = #spr.frames + 1
        end
      end

      -- Determine frame position
      local frame_num
      if PT.seq.frame_count == 0 and created_sprite then
        frame_num = 1
      elseif PT.seq.frame_count == 0 and not created_sprite then
        -- First result: use current frame position
        local target_pos = PT.seq.start_frame
        target_pos = math.min(target_pos, #spr.frames + 1)
        local new_frame = spr:newEmptyFrame(target_pos)
        frame_num = new_frame.frameNumber
      else
        local target_pos = PT.seq.start_frame + PT.seq.frame_count
        target_pos = math.min(target_pos, #spr.frames + 1)
        local new_frame = spr:newEmptyFrame(target_pos)
        frame_num = new_frame.frameNumber
      end

      -- Validate layer still exists
      local layer_valid = false
      if img and PT.seq.layer and spr.frames[frame_num] then
        for _, layer in ipairs(spr.layers) do
          if layer == PT.seq.layer then layer_valid = true; break end
        end
      end
      if layer_valid then
        spr:newCel(PT.seq.layer, spr.frames[frame_num], img, Point(0, 0))
      end
    end)

    PT.seq.frame_count = PT.seq.frame_count + 1
    app.refresh()

    if PT.dlg then
      PT.update_status("Seq frame " .. PT.seq.frame_count
        .. " (seed=" .. tostring(resp.seed or "?") .. ", " .. tostring(resp.time_ms or "?") .. "ms)")
    end
  end)
  if not ok then
    if tmp then pcall(os.remove, tmp) end
    PT.update_status("Import error: " .. tostring(err))
  end
end

-- ─── Animation Frame Import ──────────────────────────────

function PT.import_animation_frame(resp)
  if not PT.state.animating then return end
  if resp.frame_index ~= 0 and PT.anim.layer == nil then return end

  local img_data = PT.base64_decode(resp.image)
  if not img_data or #img_data == 0 then
    PT.update_status("Frame " .. (resp.frame_index + 1) .. " decode failed — skipped")
    return
  end
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
      app.activeSprite = spr
      created_sprite = true
    end

    local img = Image{ fromFile = tmp }
    os.remove(tmp)
    tmp = nil

    app.transaction("SDDj Frame " .. (resp.frame_index + 1), function()
      -- First frame: create layer and anchor position
      if resp.frame_index == 0 then
        PT.anim.layer = spr:newLayer()
        PT.anim.layer.name = "SDDj Anim #" .. tostring(resp.seed or "?")
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
    PT.mark_frame_dirty()

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

end
