--
-- SDDj — Request Builders
--

return function(PT)

function PT.parse_size()
  local s = PT.dlg.data.output_size
  local w, h = s:match("(%d+)x(%d+)")
  return tonumber(w) or 512, tonumber(h) or 512
end

function PT.parse_seed()
  local v = tonumber(PT.dlg.data.seed) or -1
  if v ~= math.floor(v) then v = -1 end
  return v
end

function PT.attach_lora(req)
  local sel = PT.dlg.data.lora_name
  if sel and sel ~= "(default)" then
    req.lora = { name = sel, weight = PT.dlg.data.lora_weight / 100.0 }
  end
end

function PT.attach_neg_ti(req)
  if PT.dlg.data.use_neg_ti and #PT.res.embeddings > 0 then
    local ti_list = {}
    local w = PT.dlg.data.neg_ti_weight / 100.0
    for _, name in ipairs(PT.res.embeddings) do
      ti_list[#ti_list + 1] = { name = name, weight = w }
    end
    req.negative_ti = ti_list
  end
end

function PT.attach_source_image(req)
  local mode = req.mode or "txt2img"
  if mode == "img2img" or mode:find("controlnet_") then
    local b64 = PT.capture_active_layer()
    if not b64 then
      app.alert("No active layer to use as source.")
      return false
    end
    if mode == "img2img" then req.source_image = b64
    else req.control_image = b64 end
  end
  if mode == "inpaint" then
    local src = PT.capture_flattened()
    if not src then
      app.alert("Inpaint requires an open sprite.")
      return false
    end
    req.source_image = src
    local mask = PT.capture_mask()
    if not mask then
      app.alert("Inpaint requires a mask.\n- Make a selection, or\n- Create a 'Mask' layer, or\n- Draw on active layer")
      return false
    end
    req.mask_image = mask
  end
  return true
end

function PT.build_post_process()
  local d = PT.dlg.data
  local pp = {
    pixelate = {
      enabled = d.pixelate,
      target_size = d.pixel_size,
    },
    quantize_enabled = d.quantize_enabled,
    quantize_method = d.quantize_method,
    quantize_colors = d.colors,
    dither = d.dither,
    palette = { mode = d.palette_mode },
    remove_bg = d.remove_bg,
  }
  if d.palette_mode == "preset" then
    pp.palette.name = d.palette_name
  elseif d.palette_mode == "custom" then
    local hex_str = d.palette_custom_colors or ""
    local colors = {}
    for hex in hex_str:gmatch("#?(%x%x%x%x%x%x)") do
      colors[#colors + 1] = "#" .. hex
    end
    if #colors > 0 then pp.palette.colors = colors end
  end
  return pp
end

-- ─── Lock Subject ─────────────────────────────────────────

function PT.build_locked_fields()
  if not PT.dlg then return {} end
  local d = PT.dlg.data
  if not d.lock_subject then return {} end
  local subj = (d.fixed_subject or ""):match("^%s*(.-)%s*$")  -- trim
  if subj == "" then return {} end
  return { subject = subj }
end

-- ─── Audio Requests ───────────────────────────────────────

function PT.build_analyze_audio_request()
  local d = PT.dlg.data
  return {
    action      = "analyze_audio",
    audio_path  = d.audio_file,
    fps         = tonumber(d.audio_fps) or 24,
    enable_stems = d.audio_stems_enable or false,
  }
end

function PT.build_audio_reactive_request()
  if not PT.dlg then return nil end
  local d = PT.dlg.data
  local gw, gh = PT.parse_size()
  local tag_name = d.audio_tag or ""
  if tag_name == "" then tag_name = nil end

  -- Inject fixed_subject into prompt for audio-linked prompt generation.
  -- The server parses base_prompt to extract and lock the subject; ensure
  -- the user's explicit subject override is present in the prompt text.
  local locked = PT.build_locked_fields()
  local effective_prompt = d.prompt
  if locked.subject then
    if not effective_prompt:find(locked.subject, 1, true) then
      effective_prompt = locked.subject .. ", " .. effective_prompt
    end
  end

  -- Build modulation slots from UI (6 slots)
  local slots = {}
  local slot_count = d.mod_slot_count or 2
  for i = 1, slot_count do
    local prefix = "mod" .. i .. "_"
    if d[prefix .. "enable"] then
      local target = d[prefix .. "target"]
      local mn = d[prefix .. "min"] / 100.0
      local mx = d[prefix .. "max"] / 100.0
      -- Scale to target range
      if target == "cfg_scale" then
        mn = mn * 30.0
        mx = mx * 30.0
      elseif target == "seed_offset" then
        mn = math.floor(mn * 1000)
        mx = math.floor(mx * 1000)
      elseif target == "controlnet_scale" then
        mn = mn * 2.0
        mx = mx * 2.0
      elseif target == "frame_cadence" then
        mn = 1.0 + mn * 7.0   -- 0%→1, 100%→8
        mx = 1.0 + mx * 7.0
      elseif target == "motion_x" or target == "motion_y" then
        mn = mn * 10.0 - 5.0  -- 0%→-5, 100%→+5
        mx = mx * 10.0 - 5.0
      elseif target == "motion_zoom" then
        mn = 0.92 + mn * 0.16 -- 0%→0.92, 100%→1.08
        mx = 0.92 + mx * 0.16
      elseif target == "motion_rotation" then
        mn = mn * 4.0 - 2.0   -- 0%→-2, 100%→+2
        mx = mx * 4.0 - 2.0
      elseif target == "motion_tilt_x" or target == "motion_tilt_y" then
        mn = mn * 6.0 - 3.0   -- 0%→-3, 100%→+3
        mx = mx * 6.0 - 3.0
      end
      slots[#slots + 1] = {
        source  = d[prefix .. "source"],
        target  = target,
        min_val = mn,
        max_val = mx,
        attack  = d[prefix .. "attack"],
        release = d[prefix .. "release"],
        enabled = true,
        invert  = d[prefix .. "invert"] or false,
      }
    end
  end

  -- Build expressions if enabled
  local expressions = nil
  if d.audio_use_expressions then
    expressions = {}
    local expr_fields = {
      "expr_denoise", "expr_cfg", "expr_noise",
      "expr_controlnet", "expr_seed", "expr_palette", "expr_cadence",
      "expr_motion_x", "expr_motion_y", "expr_motion_zoom", "expr_motion_rot",
      "expr_motion_tilt_x", "expr_motion_tilt_y",
    }
    local expr_targets = {
      "denoise_strength", "cfg_scale", "noise_amplitude",
      "controlnet_scale", "seed_offset", "palette_shift", "frame_cadence",
      "motion_x", "motion_y", "motion_zoom", "motion_rotation",
      "motion_tilt_x", "motion_tilt_y",
    }
    for idx, field in ipairs(expr_fields) do
      local val = d[field] or ""
      if val ~= "" then
        expressions[expr_targets[idx]] = val
      end
    end
    if next(expressions) == nil then expressions = nil end
  end

  -- Modulation preset
  local mod_preset = nil
  local preset_sel = d.audio_mod_preset
  if preset_sel and preset_sel ~= "(custom)" then
    mod_preset = preset_sel
  end

  -- Random seed per frame: inject seed_offset expression
  if d.audio_random_seed then
    expressions = expressions or {}
    if not expressions["seed_offset"] then
      expressions["seed_offset"] = "t * 7 + floor(global_rms * 500)"
    end
  end

  -- Prompt schedule segments
  local prompt_segments = {}
  if d.audio_prompt_schedule then
    for i = 1, 3 do
      local time_str = d["ps" .. i .. "_time"] or ""
      local prompt_str = d["ps" .. i .. "_prompt"] or ""
      if time_str ~= "" and prompt_str ~= "" then
        local s, e = time_str:match("(%d+%.?%d*)-(%d+%.?%d*)")
        if s and e then
          prompt_segments[#prompt_segments + 1] = {
            start_second = tonumber(s),
            end_second = tonumber(e),
            prompt = prompt_str,
          }
        end
      end
    end
  end

  -- Animation method: "chain" or "animatediff_audio"
  local method = d.audio_method or "chain"
  if method == "animatediff" then method = "animatediff_audio" end
  local enable_freeinit = (method == "animatediff_audio") and (d.audio_freeinit or false)
  local freeinit_iters = enable_freeinit and (d.audio_freeinit_iters or 2) or 2

  -- Max frames limit (0 = no limit)
  local max_frames = d.audio_max_frames or 0
  if max_frames <= 0 then max_frames = nil end

  local req = {
    action            = "generate_audio_reactive",
    audio_path        = d.audio_file,
    fps               = tonumber(d.audio_fps) or 24,
    enable_stems      = d.audio_stems_enable or false,
    max_frames        = max_frames,
    modulation_slots  = #slots > 0 and slots or nil,
    expressions       = expressions,
    modulation_preset = mod_preset,
    prompt_segments   = #prompt_segments > 0 and prompt_segments or nil,
    randomness        = d.randomness or 0,
    locked_fields     = next(locked) and locked or nil,
    method            = method,
    enable_freeinit   = enable_freeinit,
    freeinit_iterations = freeinit_iters,
    prompt            = effective_prompt,
    negative_prompt   = d.negative_prompt,
    mode              = d.mode,
    width             = gw,
    height            = gh,
    seed              = PT.parse_seed(),
    steps             = d.audio_steps,
    cfg_scale         = d.audio_cfg / 10.0,
    clip_skip         = d.clip_skip,
    denoise_strength  = d.audio_denoise / 100.0,

    tag_name          = tag_name,
    post_process      = PT.build_post_process(),
  }
  PT.attach_lora(req)
  PT.attach_neg_ti(req)
  PT.last_request = PT.deep_copy_request(req)
  return req
end

-- Factored from generate button onclick + loop continuation (eliminates duplication).
function PT.build_generate_request()
  if not PT.dlg then return nil end
  local gw, gh = PT.parse_size()
  local req = {
    action           = "generate",
    prompt           = PT.dlg.data.prompt,
    negative_prompt  = PT.dlg.data.negative_prompt,
    mode             = PT.dlg.data.mode,
    width            = gw,
    height           = gh,
    seed             = PT.parse_seed(),
    steps            = PT.dlg.data.steps,
    cfg_scale        = PT.dlg.data.cfg_scale / 10.0,
    clip_skip        = PT.dlg.data.clip_skip,
    denoise_strength = PT.dlg.data.denoise / 100.0,
    post_process     = PT.build_post_process(),
  }
  PT.attach_lora(req)
  PT.attach_neg_ti(req)
  PT.last_request = PT.deep_copy_request(req)
  return req
end

-- Factored from animate button onclick (consistent with build_generate_request).
function PT.build_animation_request()
  if not PT.dlg then return nil end
  local d = PT.dlg.data
  local gw, gh = PT.parse_size()
  local tag_name = d.anim_tag or ""
  if tag_name == "" then tag_name = nil end

  -- Lock Subject: inject fixed subject into animation prompt
  local locked = PT.build_locked_fields()
  local effective_prompt = d.prompt
  if locked.subject then
    if not effective_prompt:find(locked.subject, 1, true) then
      effective_prompt = locked.subject .. ", " .. effective_prompt
    end
  end

  local req = {
    action = "generate_animation",
    method = d.anim_method,
    prompt = effective_prompt,
    negative_prompt = d.negative_prompt,
    mode = d.mode,
    width = gw, height = gh,
    seed = PT.parse_seed(),
    steps = d.anim_steps,
    cfg_scale = d.anim_cfg / 10.0,
    clip_skip = d.clip_skip,
    denoise_strength = d.anim_denoise / 100.0,
    frame_count = d.anim_frames,
    frame_duration_ms = d.anim_duration,
    seed_strategy = d.anim_seed_strategy,
    tag_name = tag_name,
    enable_freeinit = d.anim_freeinit,
    freeinit_iterations = d.anim_freeinit_iters,
    post_process = PT.build_post_process(),
  }
  PT.attach_lora(req)
  PT.attach_neg_ti(req)
  PT.last_request = PT.deep_copy_request(req)
  return req
end

end
