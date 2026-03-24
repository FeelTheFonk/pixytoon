--
-- SDDj — Image Import (result, animation frame)
-- Optimized: raw RGBA Image.bytes fast path, shared decode between import/save
--

return function(PT)

-- ─── Decode helper: raw RGBA → Image via Image.bytes, or PNG fallback ───

local function _decode_to_image(resp, decoded_bytes)
  local raw = decoded_bytes or PT.base64_decode(resp.image)
  if not raw or #raw == 0 then return nil, nil end

  if resp.encoding == "raw_rgba" and resp.width and resp.height then
    -- Fast path: raw RGBA bytes → Image.bytes (no temp file, no PNG decode)
    local img = Image(resp.width, resp.height, ColorMode.RGB)
    img.bytes = raw
    return img, raw
  else
    -- Legacy PNG path: temp file → Image{fromFile}
    local tmp = PT.make_tmp_path("dec")
    local f = io.open(tmp, "wb")
    if not f then return nil, raw end
    f:write(raw)
    f:close()
    local img = Image{ fromFile = tmp }
    os.remove(tmp)
    return img, raw
  end
end

-- ─── Single Result Import ────────────────────────────────────

function PT.import_result(resp)
  local ok, err = pcall(function()
    local img, _ = _decode_to_image(resp)
    if not img then
      PT.update_status("Decode failed — skipped")
      return
    end

    local spr = app.sprite
    if spr == nil then
      spr = Sprite(resp.width or 512, resp.height or 512, ColorMode.RGB)
      app.activeSprite = spr
    end

    app.transaction("SDDj Generate", function()
      local layer = spr:newLayer()
      layer.name = "SDDj #" .. tostring(resp.seed or "?")
      spr:newCel(layer, app.frame, img, Point(0, 0))
    end)

    app.refresh()
  end)
  if not ok then
    PT.update_status("Import error: " .. tostring(err))
  end
end

-- ─── Sequence Mode: place each result as a new frame ─────────

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
  local ok, err = pcall(function()
    local img, _ = _decode_to_image(resp)
    if not img then
      PT.update_status("Seq frame decode failed — skipped")
      return
    end

    local spr = app.sprite
    local created_sprite = false
    if spr == nil then
      spr = Sprite(resp.width or 512, resp.height or 512, ColorMode.RGB)
      app.activeSprite = spr
      created_sprite = true
    end

    app.transaction("SDDj Seq", function()
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

      -- Create or reuse frame
      local frame_num
      if PT.seq.frame_count == 0 and created_sprite then
        frame_num = 1
      else
        local target_pos = PT.seq.start_frame + PT.seq.frame_count
        target_pos = math.min(target_pos, #spr.frames + 1)
        local new_frame = spr:newEmptyFrame(target_pos)
        frame_num = new_frame.frameNumber
      end

      -- Validate layer exists before creating cel
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
    PT.update_status("Import error: " .. tostring(err))
  end
end

-- ─── Animation Frame Import (optimized: shared decode + Image.bytes) ──

function PT.import_animation_frame(resp)
  if not PT.state.animating then return end
  if resp.frame_index ~= 0 and PT.anim.layer == nil then return end

  -- Decode once — store for save_animation_frame to reuse (B3 fix)
  local img, decoded_bytes = _decode_to_image(resp)
  resp._decoded_bytes = decoded_bytes

  if not img then
    PT.anim.decode_failures = PT.anim.decode_failures + 1
    PT.update_status("Frame " .. (resp.frame_index + 1) .. " decode failed ("
      .. PT.anim.decode_failures .. " total) — skipped")
    return
  end

  local ok, err = pcall(function()
    local spr = app.sprite
    local created_sprite = false
    if spr == nil then
      spr = Sprite(resp.width or 512, resp.height or 512, ColorMode.RGB)
      app.activeSprite = spr
      created_sprite = true
    end

    app.transaction("SDDj Frame", function()
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
    PT.update_status("Import error: " .. tostring(err))
  end
end

end
