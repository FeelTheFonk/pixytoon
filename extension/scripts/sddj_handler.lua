--
-- SDDj — Response Handler (dispatch table)
--

return function(PT)

local handlers = {}

-- ─── Helper: Reset Animation State ────────────────────────
local function reset_anim_state()
  PT.anim.layer = nil
  PT.anim.start_frame = 0
  PT.anim.frame_count = 0
  PT.anim.base_seed = 0
  PT.anim.output_dir = nil
  PT.anim.output_count = 0
  PT.anim.last_saved_frame = nil
  PT.anim.last_frame_index = -1
  PT.anim.decode_failures = 0
end

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

  if not resp.image or resp.image == "" then
    PT.loop.mode = false
    PT.loop.random_mode = false
    PT.loop.target = nil
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

  -- Save to output directory + metadata
  local meta = PT.build_generation_meta(resp)
  PT.last_result_meta = meta
  if PT.dlg and PT.dlg.data.save_output then
    PT.save_to_output(resp, meta)
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
          PT.update_status("Random Loop #" .. PT.loop.counter .. " — Generating prompt, then image...")
          PT.send({ action = "generate_prompt", locked_fields = PT.loop.locked_fields, randomness = PT.dlg and PT.dlg.data.randomness or 0 })
          return
        end

        -- Standard loop: generate directly
        local req = PT.build_generate_request()
        if not req then
          PT.loop.mode = false
          PT.loop.random_mode = false
          PT.loop.target = nil
          PT.finalize_sequence()
          PT.reset_ui_buttons()
          PT.update_status("Loop stopped (dialog closed)")
          return
        end
        if not PT.attach_source_image(req) then
          PT.loop.mode = false
          PT.loop.random_mode = false
          PT.loop.target = nil
          PT.finalize_sequence()
          PT.reset_ui_buttons()
          PT.update_status("Loop stopped (no source image)")
          return
        end
        PT.state.generating = true
        PT.state.gen_step_start = os.clock()
        PT.start_gen_timeout()
        PT.dlg:modify{ id = "action_btn", enabled = false }
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
  if PT.state.cancel_pending then return end
  if not resp.image or resp.image == "" then
    PT.update_status("Error: missing image in animation_frame response")
    return
  end
  if resp.frame_index ~= nil then
    -- Start refresh timer on first frame
    if resp.frame_index == 0 then PT.start_refresh_timer() end
    -- Gap detection: check frame index continuity
    if PT.anim.last_frame_index >= 0 and resp.frame_index ~= PT.anim.last_frame_index + 1 then
      local gap = resp.frame_index - PT.anim.last_frame_index - 1
      PT.update_status("Warning: " .. gap .. " frame(s) dropped before F" .. (resp.frame_index + 1))
    end
    PT.anim.last_frame_index = resp.frame_index
    PT.import_animation_frame(resp)
    -- Write frame to output directory incrementally (no memory accumulation)
    PT.save_animation_frame(resp)
  end
end

-- ─── Helper: Chunked Finalize (Bresenham Sync) ────────────

local function chunked_finalize_durations(target_fps, resp, on_complete)
  if not PT.anim.start_frame or PT.anim.frame_count <= 0 then
    if on_complete then on_complete() end
    return
  end
  local spr = app.sprite
  if not spr then
    if on_complete then on_complete() end
    return
  end
  local total = PT.anim.frame_count
  local s_frame = PT.anim.start_frame
  local chunk_size = 200
  local ideal_inc = 1000.0 / target_fps
  local function process_chunk(s_idx, elapsed_ms)
    if s_idx >= total then
      if on_complete then on_complete() end
      return
    end
    -- SOTA Stability: Graceful abort if user closes the active sprite during the async yield interval
    if not app.sprite or app.sprite ~= spr then
      return
    end
    local e_idx = math.min(s_idx + chunk_size, total)
    app.transaction("SDDj Finalize " .. s_idx .. "-" .. (e_idx - 1), function()
      for i = s_idx, e_idx - 1 do
        local fn = s_frame + i
        local expected_ms = math.floor((i + 1) * ideal_inc + 0.5)
        local dur_ms = expected_ms - elapsed_ms
        elapsed_ms = elapsed_ms + dur_ms
        if spr.frames[fn] then spr.frames[fn].duration = dur_ms / 1000.0 end
      end
      if e_idx >= total and resp.tag_name and resp.tag_name ~= "" then
        local tag_end = s_frame + total - 1
        if spr.frames[s_frame] and spr.frames[tag_end] then
          local tag = spr:newTag(s_frame, tag_end)
          tag.name = resp.tag_name
        end
      end
    end)
    app.refresh()
    local t_ref = {}
    local ok_t, t = pcall(Timer, {
      interval = 0.01,
      ontick = function()
        if t_ref[1] then pcall(function() t_ref[1]:stop() end) end
        process_chunk(e_idx, elapsed_ms)
      end,
    })
    if ok_t and t then t_ref[1] = t; t:start() end
  end
  process_chunk(0, 0)
end

-- ─── Animation Complete ─────────────────────────────────────

handlers.animation_complete = function(resp)
  PT.state.animating = false
  PT.state.gen_step_start = nil
  PT.stop_gen_timeout()
  PT.state.cancel_pending = false
  PT.timers.cancel_safety = PT.stop_timer(PT.timers.cancel_safety)
  PT.stop_refresh_timer()

  -- Validate frame count
  if resp.total_frames and PT.anim.frame_count ~= resp.total_frames then
    PT.update_status("Warning: received " .. PT.anim.frame_count .. "/" .. resp.total_frames .. " frames")
  end
  local decode_note = PT.anim.decode_failures > 0
    and " (" .. PT.anim.decode_failures .. " decode failures)" or ""

  -- Determine loop continuation before finalize
  local should_loop = PT.loop.mode and PT.loop.target == "animate" and PT.dlg

  if should_loop then
    local seed_info = tostring(resp.seed or "?")
    PT.update_status("Animate Loop #" .. PT.loop.counter .. " done" .. decode_note .. " — next...")
    -- Seed handling for next iteration
    if PT.loop.seed_mode == "increment" and resp.seed then
      PT.dlg:modify{ id = "seed", text = tostring(resp.seed + 1) }
    else
      PT.dlg:modify{ id = "seed", text = "-1" }
    end
  elseif PT.dlg then
    local tag_str = ""
    if resp.tag_name and resp.tag_name ~= "" then tag_str = ", tag=" .. resp.tag_name end
    PT.update_status("Animation done (" .. tostring(resp.total_frames or "?") .. " frames, "
      .. tostring(resp.total_time_ms or "?") .. "ms" .. tag_str .. decode_note .. ")")
    PT.reset_ui_buttons()
  end

  -- Finalize frames (set durations + tag)
  local spr = app.sprite
  if spr and PT.anim.frame_count > 0 then
    local dur_ms = (PT.dlg and tonumber(PT.dlg.data.anim_duration) or 100)
    local fps = 1000.0 / math.max(1, dur_ms)
    chunked_finalize_durations(fps, resp, function()
      PT.save_animation_meta(resp)
      reset_anim_state()
    end)
  else
    PT.save_animation_meta(resp)
    reset_anim_state()
  end

  -- Schedule loop iteration via timer (after finalize starts processing)
  if should_loop then
    PT.timers.loop = PT.stop_timer(PT.timers.loop)
    PT.timers.loop = Timer{
      interval = PT.cfg.LOOP_DELAY,
      ontick = function()
        PT.timers.loop = PT.stop_timer(PT.timers.loop)
        if not PT.loop.mode or not PT.dlg or not PT.state.connected or PT.state.animating then return end

        PT.loop.counter = PT.loop.counter + 1

        -- Random loop: generate new prompt first
        if PT.loop.random_mode then
          PT.update_status("Random Loop #" .. PT.loop.counter .. " — Generating prompt, then animation...")
          PT.send({ action = "generate_prompt", locked_fields = PT.loop.locked_fields, randomness = PT.dlg and PT.dlg.data.randomness or 0 })
          return
        end

        -- Standard loop: trigger animate directly
        PT.trigger_animate()
      end,
    }
    PT.timers.loop:start()
  end
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
  PT.loop.target = nil
  PT.timers.loop = PT.stop_timer(PT.timers.loop)
  PT.state.gen_step_start = nil
  PT.state.pending_action = nil
  PT.stop_gen_timeout()
  PT.state.cancel_pending = false
  PT.timers.cancel_safety = PT.stop_timer(PT.timers.cancel_safety)
  PT.stop_refresh_timer()
  PT.clear_response_queue()

  -- Finalize partial animation on error (async chunked to avoid UI freeze)
  if was_animating and PT.anim.frame_count > 0 then
    local spr = app.sprite
    if spr then
      local fps = 24
      if was_audio_gen then
        fps = tonumber(PT.dlg and PT.dlg.data.audio_fps) or 24
        if fps <= 0 then fps = 24 end
      else
        local dur_ms = (PT.dlg and tonumber(PT.dlg.data.anim_duration) or 100)
        fps = 1000.0 / math.max(1, dur_ms)
      end
      chunked_finalize_durations(fps, resp, function()
        reset_anim_state()
      end)
    else
      reset_anim_state()
    end
  end

  -- Finalize partial sequence on error
  PT.finalize_sequence()

  if PT.dlg then
    PT.update_status("Error: " .. tostring(resp.message or "Unknown"))
    PT.reset_ui_buttons()
    -- Re-enable analyze button (may have been disabled during analysis)
    PT.dlg:modify{ id = "audio_analyze_btn", enabled = PT.state.connected }
  end
  if resp.code ~= "CANCELLED" then
    app.alert("SDDj: " .. tostring(resp.message or "Unknown error"))
  end
end

-- ─── Resource Lists ─────────────────────────────────────────

handlers.list = function(resp)
  local lt = resp.list_type or ""
  local items = resp.items or {}

  if lt == "palettes" then
    PT.res.palettes = items
    if PT.dlg and #items > 0 then
      local prev = PT.dlg.data.palette_name
      local opts = {}
      for _, n in ipairs(items) do opts[#opts + 1] = n end
      PT.dlg:modify{ id = "palette_name", options = opts }
      if prev then
        local found = false
        for _, o in ipairs(opts) do
          if o == prev then PT.dlg:modify{ id = "palette_name", option = prev }; found = true; break end
        end
        if not found then
          PT.update_status("Palette '" .. prev .. "' no longer available")
        end
      end
    end
  elseif lt == "loras" then
    PT.res.loras = items
    if PT.dlg then
      local prev = PT.dlg.data.lora_name
      local opts = { "(default)" }
      for _, n in ipairs(items) do opts[#opts + 1] = n end
      PT.dlg:modify{ id = "lora_name", options = opts }
      if prev then
        local found = false
        for _, o in ipairs(opts) do
          if o == prev then PT.dlg:modify{ id = "lora_name", option = prev }; found = true; break end
        end
        if not found and prev ~= "(default)" then
          PT.update_status("LoRA '" .. prev .. "' no longer available")
        end
      end
    end
  elseif lt == "embeddings" then
    PT.res.embeddings = items
  elseif lt == "presets" then
    PT.res.presets = items
    if PT.dlg then
      local prev = PT.dlg.data.preset_name
      local opts = { "(none)" }
      for _, n in ipairs(items) do opts[#opts + 1] = n end
      PT.dlg:modify{ id = "preset_name", options = opts }
      if prev then
        local found = false
        for _, o in ipairs(opts) do
          if o == prev then PT.dlg:modify{ id = "preset_name", option = prev }; found = true; break end
        end
        if not found and prev ~= "(none)" then
          PT.update_status("Preset '" .. prev .. "' no longer available")
        end
      end
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

  -- Dispatch via pending_action (universal random) or random loop
  local action = PT.state.pending_action
  PT.state.pending_action = nil

  if action == "generate" then
    PT.trigger_generate()
  elseif action == "animate" then
    PT.trigger_animate()
  elseif action == "qr_generate" then
    PT.trigger_qr_generate()
  elseif action == "audio" then
    PT.trigger_audio_generate()
  elseif PT.loop.random_mode and PT.loop.mode and PT.dlg and PT.state.connected then
    -- Random loop: auto-trigger based on stored loop target
    local target = PT.loop.target or "generate"
    if target == "animate" then
      PT.trigger_animate()
    elseif target == "audio" then
      PT.trigger_audio_generate()
    else
      -- Default: generate (existing behavior)
      local req = PT.build_generate_request()
      if not req then
        PT.loop.mode = false
        PT.loop.random_mode = false
        PT.loop.target = nil
        PT.finalize_sequence()
        PT.reset_ui_buttons()
        PT.update_status("Random loop stopped (dialog closed)")
        return
      end
      if not PT.attach_source_image(req) then
        PT.loop.mode = false
        PT.loop.random_mode = false
        PT.loop.target = nil
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
    end
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
  if d.steps then
    PT.dlg:modify{ id = "steps", value = d.steps }
    PT.dlg:modify{ id = "steps", label = "Steps (" .. d.steps .. ")" }
  end
  if d.cfg_scale then
    local v = math.floor(d.cfg_scale * 10)
    PT.dlg:modify{ id = "cfg_scale", value = v }
    PT.dlg:modify{ id = "cfg_scale", label = string.format("CFG (%.1f)", v / 10.0) }
  end
  if d.clip_skip then
    PT.dlg:modify{ id = "clip_skip", value = d.clip_skip }
    PT.dlg:modify{ id = "clip_skip", label = "CLIP Skip (" .. d.clip_skip .. ")" }
  end
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
        if px.target_size then
          PT.dlg:modify{ id = "pixel_size", value = px.target_size }
          PT.dlg:modify{ id = "pixel_size", label = "Target (" .. px.target_size .. "px)" }
        end
      else
        PT.dlg:modify{ id = "pixelate", selected = px }
      end
    end
    if pp.quantize_enabled ~= nil then PT.dlg:modify{ id = "quantize_enabled", selected = pp.quantize_enabled } end
    if pp.quantize_colors then
      PT.dlg:modify{ id = "colors", value = pp.quantize_colors }
      PT.dlg:modify{ id = "colors", label = "Colors (" .. pp.quantize_colors .. ")" }
    end
    if pp.quantize_method then PT.dlg:modify{ id = "quantize_method", option = pp.quantize_method } end
    if pp.dither then PT.dlg:modify{ id = "dither", option = pp.dither } end
    if pp.remove_bg ~= nil then PT.dlg:modify{ id = "remove_bg", selected = pp.remove_bg } end
    if pp.palette then
      if pp.palette.mode then PT.dlg:modify{ id = "palette_mode", option = pp.palette.mode } end
      if pp.palette.name and pp.palette.mode == "preset" then
        PT.dlg:modify{ id = "palette_name", option = pp.palette.name }
      end
    end
  end
  -- LoRA
  if d.lora then
    if d.lora.name then PT.dlg:modify{ id = "lora_name", option = d.lora.name } end
    if d.lora.weight then
      local w = math.floor(d.lora.weight * 100)
      PT.dlg:modify{ id = "lora_weight", value = w }
      PT.dlg:modify{ id = "lora_weight", label = string.format("LoRA (%.2f)", w / 100.0) }
    end
  end
  -- Randomness / Lock Subject / Randomize
  if d.randomness then
    PT.dlg:modify{ id = "randomness", value = d.randomness }
    local v = d.randomness
    local names = { [0]="Off", [5]="Subtle", [10]="Moderate", [15]="Wild", [20]="Chaos" }
    local name = names[v] or ""
    local suffix = name ~= "" and (" — " .. name) or ""
    PT.dlg:modify{ id = "randomness", label = "Randomness (" .. v .. suffix .. ")" }
  end
  if d.lock_subject ~= nil then PT.dlg:modify{ id = "lock_subject", selected = d.lock_subject } end
  if d.fixed_subject then PT.dlg:modify{ id = "fixed_subject", text = d.fixed_subject } end
  if d.randomize_before ~= nil then PT.dlg:modify{ id = "randomize_before", selected = d.randomize_before } end
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

handlers.palette_saved = function(resp)
  if PT.dlg then
    PT.update_status("Palette '" .. tostring(resp.name or "?") .. "' saved")
    PT.send({ action = "list_palettes" })
  end
end

handlers.palette_deleted = function(resp)
  if PT.dlg then
    PT.update_status("Palette '" .. tostring(resp.name or "?") .. "' deleted")
    PT.send({ action = "list_palettes" })
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
  PT.audio.bpm = resp.bpm or 0
  PT.audio.lufs = resp.lufs or -24
  PT.audio.sample_rate = resp.sample_rate or 44100
  PT.audio.hop_length = resp.hop_length or 256
  PT.audio.recommended_preset = resp.recommended_preset or ""
  PT.audio.waveform = resp.waveform or {}

  if PT.dlg then
    local dur_str = string.format("%.1fs", PT.audio.duration)
    local stems_str = PT.audio.stems_available and " | Stems" or ""
    local bpm_str = PT.audio.bpm > 0 and string.format(" | %.0f BPM", PT.audio.bpm) or ""
    local lufs_str = PT.audio.lufs > -90 and string.format(" | %.0f LUFS", PT.audio.lufs) or ""
    PT.dlg:modify{ id = "audio_status",
      text = dur_str .. " | " .. PT.audio.total_frames .. " frames | "
        .. #PT.audio.features .. " features" .. stems_str .. bpm_str .. lufs_str }
    PT.dlg:modify{ id = "audio_analyze_btn", enabled = true }

    -- Update source dropdowns with available features (preserve selection)
    local src_opts = {}
    for _, f in ipairs(PT.audio.features) do
      src_opts[#src_opts + 1] = f
    end
    if #src_opts > 0 then
      for i = 1, 6 do
        local prev_src = PT.dlg.data["mod" .. i .. "_source"]
        PT.dlg:modify{ id = "mod" .. i .. "_source", options = src_opts }
        -- Restore previous selection if it exists in new feature list
        if prev_src then
          for _, o in ipairs(src_opts) do
            if o == prev_src then
              PT.dlg:modify{ id = "mod" .. i .. "_source", option = prev_src }
              break
            end
          end
        end
      end
    end

    -- Auto-select recommended preset
    if PT.audio.recommended_preset ~= "" then
      PT.dlg:modify{ id = "audio_mod_preset", option = PT.audio.recommended_preset }
    end

    local rec_str = PT.audio.recommended_preset ~= ""
      and " — preset: " .. PT.audio.recommended_preset or ""
    PT.update_status("Audio analyzed: " .. dur_str .. ", " .. PT.audio.total_frames
      .. " frames" .. bpm_str .. lufs_str .. rec_str)
  end
end

handlers.audio_reactive_frame = function(resp)
  if PT.state.cancel_pending then return end
  if not resp.image or resp.image == "" then
    PT.update_status("Error: missing image in audio_reactive_frame")
    return
  end
  if resp.frame_index ~= nil then
    -- Start refresh timer on first frame
    if resp.frame_index == 0 then PT.start_refresh_timer() end
    -- Gap detection: check frame index continuity
    if PT.anim.last_frame_index >= 0 and resp.frame_index ~= PT.anim.last_frame_index + 1 then
      local gap = resp.frame_index - PT.anim.last_frame_index - 1
      PT.update_status("Warning: " .. gap .. " frame(s) dropped before F" .. (resp.frame_index + 1))
    end
    PT.anim.last_frame_index = resp.frame_index
    -- Status with percentage and ETA
    local total = resp.total_frames or 1
    local idx = resp.frame_index + 1
    local pct = math.floor((idx / total) * 100)
    local eta_str = ""
    if PT.state.gen_step_start and idx > 1 then
      local elapsed = os.clock() - PT.state.gen_step_start
      local remaining = (elapsed / (idx - 1)) * (total - idx)
      if remaining < 60 then
        eta_str = string.format(" ~%.0fs", remaining)
      else
        eta_str = string.format(" ~%.1fmin", remaining / 60)
      end
    end
    PT.update_status("Audio " .. idx .. "/" .. total .. " (" .. pct .. "%)" .. eta_str)
    -- Reuse the animation frame import mechanism
    PT.import_animation_frame(resp)
    -- Write frame to output directory incrementally
    PT.save_animation_frame(resp)
  end
end

handlers.audio_reactive_complete = function(resp)
  PT.state.animating = false
  PT.audio.generating = false
  PT.state.gen_step_start = nil
  PT.stop_gen_timeout()
  PT.state.cancel_pending = false
  PT.timers.cancel_safety = PT.stop_timer(PT.timers.cancel_safety)
  PT.stop_refresh_timer()

  -- Frame count validation (parity with animation_complete)
  if resp.total_frames and PT.anim.frame_count ~= resp.total_frames then
    PT.update_status("Warning: received " .. PT.anim.frame_count .. "/" .. resp.total_frames .. " frames")
  end
  local decode_note = PT.anim.decode_failures > 0
    and " (" .. PT.anim.decode_failures .. " decode failures)" or ""

  -- Determine loop continuation before finalize
  local should_loop = PT.loop.mode and PT.loop.target == "audio" and PT.dlg

  if should_loop then
    PT.update_status("Audio Loop #" .. PT.loop.counter .. " done" .. decode_note .. " — next...")
    -- Seed handling for next iteration
    if PT.loop.seed_mode == "increment" and resp.seed then
      PT.dlg:modify{ id = "seed", text = tostring(resp.seed + 1) }
    else
      PT.dlg:modify{ id = "seed", text = "-1" }
    end
  elseif PT.dlg then
    local tag_str = ""
    if resp.tag_name and resp.tag_name ~= "" then tag_str = ", tag=" .. resp.tag_name end
    PT.update_status("Audio animation done (" .. tostring(resp.total_frames or "?") .. " frames, "
      .. tostring(resp.total_time_ms or "?") .. "ms" .. tag_str .. decode_note .. ")")
    PT.reset_ui_buttons()
    PT.dlg:modify{ id = "action_btn", enabled = PT.state.connected }
    -- Enable MP4 export (output dir preserved in audio.last_output_dir after reset)
    PT.dlg:modify{ id = "export_mp4_btn", enabled = true }
  end

  -- Finalize frames (set durations + tag)
  local spr = app.sprite
  if spr and PT.anim.frame_count > 0 then
    local fps = (PT.dlg and tonumber(PT.dlg.data.audio_fps)) or 24
    if fps <= 0 then fps = 24 end
    chunked_finalize_durations(fps, resp, function()
      PT.save_animation_meta(resp)
      PT.audio.last_output_dir = PT.anim.output_dir
      reset_anim_state()
    end)
  else
    PT.save_animation_meta(resp)
    PT.audio.last_output_dir = PT.anim.output_dir
    reset_anim_state()
  end

  -- Schedule loop iteration via timer
  if should_loop then
    PT.timers.loop = PT.stop_timer(PT.timers.loop)
    PT.timers.loop = Timer{
      interval = PT.cfg.LOOP_DELAY,
      ontick = function()
        PT.timers.loop = PT.stop_timer(PT.timers.loop)
        if not PT.loop.mode or not PT.dlg or not PT.state.connected or PT.state.animating then return end

        PT.loop.counter = PT.loop.counter + 1

        -- Random loop: generate new prompt first
        if PT.loop.random_mode then
          PT.update_status("Random Loop #" .. PT.loop.counter .. " — Generating prompt, then audio...")
          PT.send({ action = "generate_prompt", locked_fields = PT.loop.locked_fields, randomness = PT.dlg and PT.dlg.data.randomness or 0 })
          return
        end

        -- Standard loop: trigger audio directly
        PT.trigger_audio_generate()
      end,
    }
    PT.timers.loop:start()
  end
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
    -- Preserve current selection before updating options
    local prev = PT.dlg.data.audio_mod_preset or "(custom)"
    local opts = { "(custom)" }
    for _, p in ipairs(PT.audio.mod_presets) do
      opts[#opts + 1] = p
    end
    PT.dlg:modify{ id = "audio_mod_preset", options = opts }
    -- Restore previous selection if it still exists in new options
    for _, o in ipairs(opts) do
      if o == prev then
        PT.dlg:modify{ id = "audio_mod_preset", option = prev }
        break
      end
    end
  end
end

handlers.modulation_preset_detail = function(resp)
  if not PT.dlg or not resp.slots then return end
  local slots = resp.slots
  local count = math.min(#slots, 6)  -- UI supports max 6 slots

  -- Inverse scaling: convert actual parameter values to slider % (0-100)
  local function to_pct(target, val)
    if target == "cfg_scale" then return val / 30.0 * 100
    elseif target == "seed_offset" then return val / 1000.0 * 100
    elseif target == "controlnet_scale" then return val / 2.0 * 100
    elseif target == "frame_cadence" then return (val - 1.0) / 7.0 * 100
    elseif target == "motion_x" or target == "motion_y" then return (val + 5.0) / 10.0 * 100
    elseif target == "motion_zoom" then return (val - 0.92) / 0.16 * 100
    elseif target == "motion_rotation" then return (val + 2.0) / 4.0 * 100
    elseif target == "motion_tilt_x" or target == "motion_tilt_y" then return (val + 3.0) / 6.0 * 100
    else return val * 100  -- denoise_strength, noise_amplitude, palette_shift
    end
  end

  -- Suppress auto-switch to (custom) during hydration
  PT.audio._hydrating_preset = true

  -- Update slot count
  PT.dlg:modify{ id = "mod_slot_count", value = count }
  PT.dlg:modify{ id = "mod_slot_count", label = "Slots (" .. count .. ")" }

  -- Populate each slot
  for i = 1, 6 do
    local prefix = "mod" .. i .. "_"
    if i <= count then
      local s = slots[i]
      PT.dlg:modify{ id = prefix .. "enable", selected = s.enabled ~= false }
      PT.dlg:modify{ id = prefix .. "invert", selected = s.invert == true }
      PT.dlg:modify{ id = prefix .. "source", option = s.source }
      PT.dlg:modify{ id = prefix .. "target", option = s.target }
      local mn = math.floor(to_pct(s.target, s.min_val) + 0.5)
      local mx = math.floor(to_pct(s.target, s.max_val) + 0.5)
      mn = math.max(0, math.min(100, mn))
      mx = math.max(0, math.min(100, mx))
      PT.dlg:modify{ id = prefix .. "min", value = mn }
      PT.dlg:modify{ id = prefix .. "max", value = mx }
      PT.dlg:modify{ id = prefix .. "attack", value = s.attack or 2 }
      PT.dlg:modify{ id = prefix .. "release", value = s.release or 8 }
    else
      -- Disable unused slots
      PT.dlg:modify{ id = prefix .. "enable", selected = false }
    end
  end

  PT.audio._hydrating_preset = false
  PT.update_status("Preset '" .. (resp.name or "?") .. "' loaded (" .. count .. " slots)")
end

handlers.expression_presets_list = function(resp)
  if not PT.dlg or not resp.presets then return end
  PT.audio.expression_presets = resp.presets
  -- Build flat options list: (manual) + all presets grouped by category
  local opts = { "(manual)" }
  for cat, presets in pairs(resp.presets) do
    for _, p in ipairs(presets) do
      opts[#opts + 1] = p.name
    end
  end
  PT.dlg:modify{ id = "audio_expr_preset", options = opts }
end

handlers.expression_preset_detail = function(resp)
  if not PT.dlg or not resp.targets then return end
  -- Map target names to expression field IDs
  local target_to_field = {
    denoise_strength = "expr_denoise",
    cfg_scale = "expr_cfg",
    noise_amplitude = "expr_noise",
    controlnet_scale = "expr_controlnet",
    seed_offset = "expr_seed",
    palette_shift = "expr_palette",
    frame_cadence = "expr_cadence",
    motion_x = "expr_motion_x",
    motion_y = "expr_motion_y",
    motion_zoom = "expr_motion_zoom",
    motion_rotation = "expr_motion_rot",
    motion_tilt_x = "expr_motion_tilt_x",
    motion_tilt_y = "expr_motion_tilt_y",
  }
  -- Enable expressions and fill fields
  PT.dlg:modify{ id = "audio_use_expressions", selected = true }
  for target, expr in pairs(resp.targets) do
    local field = target_to_field[target]
    if field then
      PT.dlg:modify{ id = field, text = expr }
    end
  end
  PT.update_status("Expression preset '" .. (resp.name or "?") .. "' loaded")
end

handlers.choreography_preset_detail = function(resp)
  if not PT.dlg then return end
  -- Hydrate slots (reuse modulation_preset_detail logic)
  if resp.slots and #resp.slots > 0 then
    handlers.modulation_preset_detail({
      name = resp.name, slots = resp.slots,
    })
  end
  -- Hydrate expressions
  if resp.expressions then
    handlers.expression_preset_detail({
      name = resp.name, targets = resp.expressions,
    })
  end
  PT.update_status("Choreography '" .. (resp.name or "?") .. "' loaded")
end

handlers.choreography_presets_list = function(resp)
  -- Populate choreography combobox if presets were received
  if not PT.dlg or not resp.presets then return end
  local opts = { "(none)" }
  for _, p in ipairs(resp.presets) do
    opts[#opts + 1] = p.name
  end
  PT.dlg:modify{ id = "audio_choreography", options = opts }
end

-- ─── MP4 Export ──────────────────────────────────────────────

handlers.export_mp4_complete = function(resp)
  if PT.dlg then
    local size_str = string.format("%.1f MB", resp.size_mb or 0)
    PT.update_status("MP4 exported: " .. size_str)
    PT.dlg:modify{ id = "export_mp4_btn", enabled = true }
  end
end

handlers.export_mp4_error = function(resp)
  if PT.dlg then
    PT.update_status("MP4 export failed: " .. (resp.message or "unknown error"))
    PT.dlg:modify{ id = "export_mp4_btn", enabled = true }
  end
end

-- ─── Shutdown Ack ────────────────────────────────────────────

handlers.shutdown_ack = function(resp)
  -- Server confirmed shutdown — disconnect cleanly
  if PT.dlg then PT.update_status("Server shutting down...") end
end

-- ─── Decoupled Refresh Timer ────────────────────────────────
-- Frame imports no longer call app.refresh() directly (that caused
-- re-entrant event pumping → stack overflow or frame batching).
-- Instead, a ~30fps timer repaints the canvas independently.
-- This yields to the event loop between repaints, so each frame
-- appears on canvas in real-time with zero overhead per import.

local _frame_dirty = false
local _refresh_timer = nil

function PT.mark_frame_dirty()
  _frame_dirty = true
end

function PT.start_refresh_timer()
  if _refresh_timer then return end
  local ok, t = pcall(Timer, {
    interval = 0.033,  -- ~30fps visual refresh
    ontick = function()
      if _frame_dirty then
        _frame_dirty = false
        pcall(app.refresh)
      end
    end,
  })
  if ok and t then _refresh_timer = t; t:start() end
end

function PT.stop_refresh_timer()
  if _refresh_timer then
    pcall(function() if _refresh_timer.isRunning then _refresh_timer:stop() end end)
    _refresh_timer = nil
  end
  -- Final refresh to flush last frame
  if _frame_dirty then
    _frame_dirty = false
    pcall(app.refresh)
  end
end

-- ─── Dispatch (anti-re-entrancy safety) ─────────────────────
-- With app.refresh() removed from frame imports, re-entrance should
-- not occur. The guard + timer drain remain as a safety net in case
-- app.transaction() or dlg:modify() pump the event loop internally.

local _response_queue = {}
local _processing = false
local _drain_timer = nil

local function _schedule_drain()  -- forward declaration filled below
end

local function _drain_next()
  if #_response_queue == 0 then return end
  -- If cancel was requested between ticks, flush remaining queue
  if PT.state.cancel_pending then
    PT.clear_response_queue()
    return
  end
  -- Process up to 4 messages per tick to clear backlog faster (B6 fix)
  local batch = math.min(#_response_queue, 4)
  for _ = 1, batch do
    if #_response_queue == 0 then break end
    local queued = table.remove(_response_queue, 1)
    _processing = true
    local ok, err = pcall(function()
      local handler = handlers[queued.type]
      if handler then handler(queued) end
    end)
    _processing = false
    if not ok then
      pcall(PT.update_status, "Handler error: " .. tostring(err))
    end
  end
  if #_response_queue > 0 then
    _schedule_drain()
  end
end

_schedule_drain = function()
  if _drain_timer then return end
  local ok, t = pcall(Timer, {
    interval = 0.001,  -- 1ms — minimal yield to event loop
    ontick = function()
      if _drain_timer then
        pcall(function() _drain_timer:stop() end)
      end
      _drain_timer = nil
      _drain_next()
    end,
  })
  if ok and t then _drain_timer = t; t:start() end
end

function PT.handle_response(resp)
  if _processing then
    _response_queue[#_response_queue + 1] = resp
    return
  end
  _processing = true
  local ok, err = pcall(function()
    local handler = handlers[resp.type]
    if handler then handler(resp) end
  end)
  _processing = false
  if not ok then
    pcall(PT.update_status, "Handler error: " .. tostring(err))
  end
  -- If messages queued during processing, drain via timer
  -- (one per event loop turn — allows repaint between messages)
  if #_response_queue > 0 then
    _schedule_drain()
  end
end

function PT.clear_response_queue()
  for i = #_response_queue, 1, -1 do _response_queue[i] = nil end
  if _drain_timer then
    pcall(function() _drain_timer:stop() end)
    _drain_timer = nil
  end
end

end
