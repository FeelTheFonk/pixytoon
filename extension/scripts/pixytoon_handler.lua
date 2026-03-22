--
-- PixyToon — Response Handler (dispatch table)
--

return function(PT)

local handlers = {}

-- ─── Progress ───────────────────────────────────────────────

handlers.progress = function(resp)
  if not resp.total or resp.total <= 0 or not resp.step then return end
  if not PT.dlg then return end

  local pct = math.floor((resp.step / resp.total) * 100)
  local eta_str = ""
  local now = os.clock()
  if PT.state.gen_step_start and resp.step > 1 then
    local elapsed = now - PT.state.gen_step_start
    local steps_done = resp.step - 1
    if steps_done > 0 then
      local remaining = (elapsed / steps_done) * (resp.total - resp.step)
      if remaining < 60 then
        eta_str = string.format(" ~%.0fs", remaining)
      else
        eta_str = string.format(" ~%.1fmin", remaining / 60)
      end
    end
  end

  local frame_ctx = ""
  if resp.frame_index ~= nil and resp.total_frames ~= nil then
    frame_ctx = " [F" .. (resp.frame_index + 1) .. "/" .. resp.total_frames .. "]"
  end
  PT.update_status(resp.step .. "/" .. resp.total .. " (" .. pct .. "%)" .. frame_ctx .. eta_str)
end

-- ─── Generation Result ──────────────────────────────────────

handlers.result = function(resp)
  PT.state.generating = false
  PT.state.gen_step_start = nil
  PT.stop_gen_timeout()
  PT.state.cancel_pending = false
  PT.timers.cancel_safety = PT.stop_timer(PT.timers.cancel_safety)

  if not resp.image then
    PT.loop.mode = false
    PT.loop.random_mode = false
    PT.update_status("Error: missing image in result response")
    PT.reset_ui_buttons()
    return
  end
  if not resp.seed then resp.seed = 0 end

  -- Dispatch based on output mode: layer (default) or sequence (timeline)
  if PT.dlg and PT.dlg.data.output_mode == "sequence" then
    PT.import_result_as_frame(resp)
  else
    PT.import_result(resp)
  end

  -- Loop mode: schedule next generation
  if PT.loop.mode and PT.dlg then
    local seed_info = tostring(resp.seed or "?")
    PT.update_status("Loop #" .. PT.loop.counter .. " done (seed=" .. seed_info .. ") — next...")
    -- Adjust seed for next iteration
    if PT.loop.seed_mode == "increment" and resp.seed then
      PT.dlg:modify{ id = "seed", text = tostring(resp.seed + 1) }
    else
      PT.dlg:modify{ id = "seed", text = "-1" }
    end
    -- Small delay then trigger next generation (or random prompt first)
    PT.timers.loop = PT.stop_timer(PT.timers.loop)
    PT.timers.loop = Timer{
      interval = PT.cfg.LOOP_DELAY,
      ontick = function()
        PT.timers.loop = PT.stop_timer(PT.timers.loop)
        if not PT.loop.mode or not PT.dlg or not PT.state.connected or PT.state.generating then return end

        PT.loop.counter = PT.loop.counter + 1

        -- Random loop: generate new prompt first, then auto-generate in prompt_result handler
        if PT.loop.random_mode then
          PT.update_status("Random Loop #" .. PT.loop.counter .. " — Generating prompt...")
          PT.send({ action = "generate_prompt", locked_fields = PT.loop.locked_fields })
          return
        end

        -- Standard loop: generate directly
        local req = PT.build_generate_request()
        if not PT.attach_source_image(req) then
          PT.loop.mode = false
          PT.loop.random_mode = false
          PT.finalize_sequence()
          PT.reset_ui_buttons()
          PT.update_status("Loop stopped (no source image)")
          return
        end
        PT.state.generating = true
        PT.state.gen_step_start = os.clock()
        PT.start_gen_timeout()
        PT.dlg:modify{ id = "generate_btn", enabled = false }
        PT.dlg:modify{ id = "cancel_btn", enabled = true }
        PT.update_status("Loop #" .. PT.loop.counter .. " — Generating...")
        PT.send(req)
      end,
    }
    PT.timers.loop:start()
  elseif PT.dlg then
    -- Not looping: finalize any active sequence
    PT.finalize_sequence()
    PT.update_status("Done (" .. tostring(resp.time_ms or "?") .. "ms, seed=" .. tostring(resp.seed or "?") .. ")")
    PT.reset_ui_buttons()
  end
end

-- ─── Animation Frame ────────────────────────────────────────

handlers.animation_frame = function(resp)
  if not resp.image then
    PT.update_status("Error: missing image in animation_frame response")
    return
  end
  if resp.frame_index ~= nil then
    PT.import_animation_frame(resp)
  end
end

-- ─── Animation Complete ─────────────────────────────────────

handlers.animation_complete = function(resp)
  PT.state.animating = false
  PT.state.gen_step_start = nil
  PT.stop_gen_timeout()
  PT.state.cancel_pending = false
  PT.timers.cancel_safety = PT.stop_timer(PT.timers.cancel_safety)

  -- Validate frame count
  if resp.total_frames and PT.anim.frame_count ~= resp.total_frames then
    PT.update_status("Warning: received " .. PT.anim.frame_count .. "/" .. resp.total_frames .. " frames")
  end

  if PT.dlg then
    local tag_str = ""
    if resp.tag_name and resp.tag_name ~= "" then tag_str = ", tag=" .. resp.tag_name end
    PT.update_status("Animation done (" .. tostring(resp.total_frames or "?") .. " frames, "
      .. tostring(resp.total_time_ms or "?") .. "ms" .. tag_str .. ")")
    PT.reset_ui_buttons()
  end

  local spr = app.sprite
  if spr and PT.anim.frame_count > 0 then
    app.transaction("PixyToon Animation Finalize", function()
      local dur = (PT.dlg and PT.dlg.data.anim_duration or 100) / 1000.0
      for i = 0, PT.anim.frame_count - 1 do
        local fn = PT.anim.start_frame + i
        if spr.frames[fn] then spr.frames[fn].duration = dur end
      end
      local tag_start = PT.anim.start_frame
      local tag_end = PT.anim.start_frame + PT.anim.frame_count - 1
      if resp.tag_name and resp.tag_name ~= "" and spr.frames[tag_start] and spr.frames[tag_end] then
        local tag = spr:newTag(tag_start, tag_end)
        tag.name = resp.tag_name
      end
    end)
    app.refresh()
  end
  PT.anim.layer = nil
  PT.anim.start_frame = 0
  PT.anim.frame_count = 0
  PT.anim.base_seed = 0
end

-- ─── Error ──────────────────────────────────────────────────

handlers.error = function(resp)
  local was_animating = PT.state.animating
  local was_audio_gen = PT.audio.generating
  PT.state.generating = false
  PT.state.animating = false
  PT.audio.generating = false
  PT.audio.analyzing = false
  PT.loop.mode = false
  PT.loop.random_mode = false
  PT.timers.loop = PT.stop_timer(PT.timers.loop)
  PT.state.gen_step_start = nil
  PT.stop_gen_timeout()
  PT.state.cancel_pending = false
  PT.timers.cancel_safety = PT.stop_timer(PT.timers.cancel_safety)

  -- Finalize partial animation on error
  if was_animating and PT.anim.frame_count > 0 then
    local spr = app.sprite
    if spr then
      -- Use audio frame duration if this was an audio-reactive generation
      local dur_ms = was_audio_gen
        and (PT.dlg and PT.dlg.data.audio_frame_duration or 42)
        or (PT.dlg and PT.dlg.data.anim_duration or 100)
      local dur = dur_ms / 1000.0
      for i = 0, PT.anim.frame_count - 1 do
        local fn = PT.anim.start_frame + i
        if spr.frames[fn] then spr.frames[fn].duration = dur end
      end
    end
    PT.anim.layer = nil
    PT.anim.start_frame = 0
    PT.anim.frame_count = 0
    PT.anim.base_seed = 0
  end

  -- Finalize partial sequence on error
  PT.finalize_sequence()

  PT.live.request_inflight = false
  PT.live.inflight_time = nil
  -- If live mode was active, stop it cleanly before resetting UI
  if PT.live.mode then
    PT.stop_live_mode()
  end
  if PT.dlg then
    PT.update_status("Error: " .. tostring(resp.message or "Unknown"))
    PT.reset_ui_buttons()
    -- Re-enable analyze button (may have been disabled during analysis)
    PT.dlg:modify{ id = "audio_analyze_btn", enabled = PT.state.connected }
  end
  if resp.code ~= "CANCELLED" then
    app.alert("PixyToon: " .. tostring(resp.message or "Unknown error"))
  end
end

-- ─── Resource Lists ─────────────────────────────────────────

handlers.list = function(resp)
  local lt = resp.list_type or ""
  local items = resp.items or {}

  if lt == "palettes" then
    PT.res.palettes = items
    if PT.dlg and #items > 0 then
      local opts = {}
      for _, n in ipairs(items) do opts[#opts + 1] = n end
      PT.dlg:modify{ id = "palette_name", options = opts }
    end
  elseif lt == "loras" then
    PT.res.loras = items
    if PT.dlg then
      local opts = { "(default)" }
      for _, n in ipairs(items) do opts[#opts + 1] = n end
      PT.dlg:modify{ id = "lora_name", options = opts }
    end
  elseif lt == "embeddings" then
    PT.res.embeddings = items
  elseif lt == "presets" then
    PT.res.presets = items
    if PT.dlg then
      local opts = { "(none)" }
      for _, n in ipairs(items) do opts[#opts + 1] = n end
      PT.dlg:modify{ id = "preset_name", options = opts }
    end
  end

  local total = #PT.res.palettes + #PT.res.loras + #PT.res.embeddings
  if total > 0 then
    PT.update_status("Resources loaded (" .. #PT.res.loras .. " LoRAs, "
      .. #PT.res.palettes .. " palettes, " .. #PT.res.embeddings .. " embeddings)")
  else
    PT.update_status("Connected (no resources found)")
  end
end

-- ─── Realtime Mode ──────────────────────────────────────────

handlers.realtime_ready = function(resp)
  PT.live.mode = true
  PT.live.request_inflight = false
  PT.live.last_prompt = PT.dlg and PT.dlg.data.prompt or nil
  if PT.dlg then
    local mode_str = PT.live.auto_mode and "auto (stroke)" or "manual (F5)"
    PT.update_status("Live mode active — " .. mode_str)
    PT.dlg:modify{ id = "live_btn", text = "STOP LIVE" }
    PT.dlg:modify{ id = "live_accept_btn", visible = true }
    PT.dlg:modify{ id = "live_send_btn", visible = true }
    PT.dlg:modify{ id = "generate_btn", enabled = false }
    PT.dlg:modify{ id = "animate_btn", enabled = false }
  end
  PT.start_live_timer()
end

handlers.realtime_result = function(resp)
  PT.live.request_inflight = false
  PT.live.inflight_time = nil
  if not resp.image then
    PT.update_status("Error: missing image in realtime_result")
    return
  end
  -- Drop stale frames (latest-wins); use ~= nil because frame_id=0 is valid
  if resp.frame_id ~= nil and resp.frame_id < PT.live.frame_id then
    return
  end
  if PT.live.mode then
    PT.live_update_preview(resp)
    if PT.dlg then
      local mode_str = PT.live.auto_mode and "auto" or "manual"
      PT.update_status("Live " .. mode_str .. " (" .. tostring(resp.latency_ms or "?") .. "ms)")
    end
  end
end

handlers.realtime_stopped = function(resp)
  -- Clean up preview layer before stopping timers/listeners
  if PT.live.preview_layer then
    local spr = app.sprite
    if spr then
      PT.live.importing = true
      pcall(function()
        local cel = PT.live.preview_layer:cel(app.frame)
        if cel then spr:deleteCel(cel) end
        spr:deleteLayer(PT.live.preview_layer)
      end)
      PT.live.importing = false
    end
    pcall(app.refresh)
  end
  PT.stop_live_mode()
end

-- ─── Pong / Misc ────────────────────────────────────────────

handlers.pong = function(resp)
  PT.state.last_pong = os.clock()
  if not PT.state.connected then PT.set_connected(true) end
  if not PT.res.requested then PT.request_resources() end
  PT.update_status("Connected")
end

handlers.prompt_result = function(resp)
  if not PT.dlg or not resp.prompt then return end
  PT.dlg:modify{ id = "prompt", text = resp.prompt }
  if resp.negative_prompt and resp.negative_prompt ~= "" then
    PT.dlg:modify{ id = "negative_prompt", text = resp.negative_prompt }
  end
  -- Populate fixed_subject from generated components so Lock Subject works
  if resp.components and resp.components.subject then
    PT.dlg:modify{ id = "fixed_subject", text = resp.components.subject }
  end

  -- Random loop: auto-trigger generation after prompt is set
  if PT.loop.random_mode and PT.loop.mode and PT.dlg and PT.state.connected then
    local req = PT.build_generate_request()
    if not PT.attach_source_image(req) then
      PT.loop.mode = false
      PT.loop.random_mode = false
      PT.finalize_sequence()
      PT.reset_ui_buttons()
      PT.update_status("Random loop stopped (no source image)")
      return
    end
    PT.state.generating = true
    PT.state.gen_step_start = os.clock()
    PT.start_gen_timeout()
    PT.update_status("Random Loop #" .. PT.loop.counter .. " — Generating...")
    PT.send(req)
  else
    PT.update_status("Prompt generated")
  end
end

handlers.preset = function(resp)
  if not PT.dlg or not resp.data then return end
  local d = resp.data
  if d.prompt_prefix then PT.dlg:modify{ id = "prompt", text = d.prompt_prefix } end
  if d.negative_prompt then PT.dlg:modify{ id = "negative_prompt", text = d.negative_prompt } end
  if d.mode then PT.dlg:modify{ id = "mode", option = d.mode } end
  if d.width and d.height then
    PT.dlg:modify{ id = "output_size", option = d.width .. "x" .. d.height }
  end
  if d.steps then PT.dlg:modify{ id = "steps", value = d.steps } end
  if d.cfg_scale then
    local v = math.floor(d.cfg_scale * 10)
    PT.dlg:modify{ id = "cfg_scale", value = v }
    PT.dlg:modify{ id = "cfg_scale", label = string.format("CFG (%.1f)", v / 10.0) }
  end
  if d.clip_skip then PT.dlg:modify{ id = "clip_skip", value = d.clip_skip } end
  if d.denoise_strength then
    local v = math.floor(d.denoise_strength * 100)
    PT.dlg:modify{ id = "denoise", value = v }
    PT.dlg:modify{ id = "denoise", label = string.format("Strength (%.2f)", v / 100.0) }
  end
  if d.post_process then
    local pp = d.post_process
    if pp.pixelate ~= nil then
      local px = pp.pixelate
      if type(px) == "table" then
        if px.enabled ~= nil then PT.dlg:modify{ id = "pixelate", selected = px.enabled } end
        if px.target_size then PT.dlg:modify{ id = "pixel_size", value = px.target_size } end
      else
        PT.dlg:modify{ id = "pixelate", selected = px }
      end
    end
    if pp.quantize_colors then PT.dlg:modify{ id = "colors", value = pp.quantize_colors } end
    if pp.quantize_method then PT.dlg:modify{ id = "quantize_method", option = pp.quantize_method } end
    if pp.dither then PT.dlg:modify{ id = "dither", option = pp.dither } end
  end
  PT.update_status("Preset '" .. tostring(resp.name or "?") .. "' loaded")
end

handlers.preset_saved = function(resp)
  if PT.dlg then
    PT.update_status("Preset '" .. tostring(resp.name or "?") .. "' saved")
    PT.send({ action = "list_presets" })
  end
end

handlers.preset_deleted = function(resp)
  if PT.dlg then
    PT.update_status("Preset '" .. tostring(resp.name or "?") .. "' deleted")
    PT.dlg:modify{ id = "preset_name", option = "(none)" }
    PT.send({ action = "list_presets" })
  end
end

handlers.cleanup_done = function(resp)
  if PT.dlg then
    PT.update_status(tostring(resp.message or "Cleanup done")
      .. " (freed " .. string.format("%.1f", resp.freed_mb or 0) .. " MB)")
  end
end

-- ─── Audio Reactivity ─────────────────────────────────────────

handlers.audio_analysis = function(resp)
  PT.audio.analyzed = true
  PT.audio.analyzing = false
  PT.audio.duration = resp.duration or 0
  PT.audio.total_frames = resp.total_frames or 0
  PT.audio.features = resp.features or {}
  PT.audio.stems_available = resp.stems_available or false
  PT.audio.stems = resp.stems or {}

  if PT.dlg then
    local dur_str = string.format("%.1fs", PT.audio.duration)
    local stems_str = PT.audio.stems_available and " | Stems" or ""
    PT.dlg:modify{ id = "audio_status",
      text = dur_str .. " | " .. PT.audio.total_frames .. " frames | "
        .. #PT.audio.features .. " features" .. stems_str }
    PT.dlg:modify{ id = "audio_analyze_btn", enabled = true }

    -- Update source dropdowns with available features
    local src_opts = {}
    for _, f in ipairs(PT.audio.features) do
      src_opts[#src_opts + 1] = f
    end
    if #src_opts > 0 then
      for i = 1, 4 do
        PT.dlg:modify{ id = "mod" .. i .. "_source", options = src_opts }
      end
    end

    PT.update_status("Audio analyzed: " .. dur_str .. ", " .. PT.audio.total_frames .. " frames")
  end
end

handlers.audio_reactive_frame = function(resp)
  if not resp.image then
    PT.update_status("Error: missing image in audio_reactive_frame")
    return
  end
  if resp.frame_index ~= nil then
    -- Reuse the animation frame import mechanism
    PT.import_animation_frame(resp)
  end
end

handlers.audio_reactive_complete = function(resp)
  PT.state.animating = false
  PT.audio.generating = false
  PT.state.gen_step_start = nil
  PT.stop_gen_timeout()
  PT.state.cancel_pending = false
  PT.timers.cancel_safety = PT.stop_timer(PT.timers.cancel_safety)

  if PT.dlg then
    local tag_str = ""
    if resp.tag_name and resp.tag_name ~= "" then tag_str = ", tag=" .. resp.tag_name end
    PT.update_status("Audio animation done (" .. tostring(resp.total_frames or "?") .. " frames, "
      .. tostring(resp.total_time_ms or "?") .. "ms" .. tag_str .. ")")
    PT.reset_ui_buttons()
    PT.dlg:modify{ id = "audio_generate_btn", enabled = PT.state.connected }
  end

  -- Finalize frames (set durations + tag)
  local spr = app.sprite
  if spr and PT.anim.frame_count > 0 then
    app.transaction("PixyToon Audio Animation Finalize", function()
      local dur_ms = (PT.dlg and PT.dlg.data.audio_frame_duration or 42)
      local dur = dur_ms / 1000.0
      for i = 0, PT.anim.frame_count - 1 do
        local fn = PT.anim.start_frame + i
        if spr.frames[fn] then spr.frames[fn].duration = dur end
      end
      local tag_start = PT.anim.start_frame
      local tag_end = PT.anim.start_frame + PT.anim.frame_count - 1
      if resp.tag_name and resp.tag_name ~= "" and spr.frames[tag_start] and spr.frames[tag_end] then
        local tag = spr:newTag(tag_start, tag_end)
        tag.name = resp.tag_name
      end
    end)
    app.refresh()
  end
  PT.anim.layer = nil
  PT.anim.start_frame = 0
  PT.anim.frame_count = 0
  PT.anim.base_seed = 0
end

handlers.stems_available = function(resp)
  PT.audio.stems_available = resp.available
  if PT.dlg then
    PT.update_status(resp.available and "Stems: ready" or "Stems: " .. (resp.message or "not available"))
  end
end

handlers.modulation_presets = function(resp)
  PT.audio.mod_presets = resp.presets or {}
  if PT.dlg and #PT.audio.mod_presets > 0 then
    local opts = { "(custom)" }
    for _, p in ipairs(PT.audio.mod_presets) do
      opts[#opts + 1] = p
    end
    PT.dlg:modify{ id = "audio_mod_preset", options = opts }
  end
end

-- ─── Dispatch ───────────────────────────────────────────────

function PT.handle_response(resp)
  local handler = handlers[resp.type]
  if handler then handler(resp) end
end

end
