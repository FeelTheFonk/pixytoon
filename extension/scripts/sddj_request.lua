--
-- SDDj — Request Builders
--

return function(PT)

-- Clamp a numeric value to [lo, hi]
local function clamp(v, lo, hi)
  if type(v) ~= "number" or v ~= v then v = lo end
  return math.max(lo, math.min(hi, v))
end

-- Map UI display name to backend scheduler identifier
local SCHEDULER_MAP = {
  ["DPM++ SDE Karras"] = "dpm++_sde_karras",
  ["DPM++ 2M Karras"] = "dpm++_2m_karras",
  ["DDIM"] = "ddim",
  ["Euler Ancestral"] = "euler_a",
  ["Euler"] = "euler",
  ["UniPC"] = "unipc",
  ["LMS"] = "lms",
}

function PT.attach_scheduler(req)
  local sched = PT.dlg.data.scheduler
  if sched and SCHEDULER_MAP[sched] then
    req.scheduler = SCHEDULER_MAP[sched]
  end
end

function PT.parse_size()
  local s = tostring(PT.dlg.data.output_size or "512x512")
  local w, h = s:match("(%d+)x(%d+)")
  w = tonumber(w) or 512
  h = tonumber(h) or 512
  if w <= 0 or w ~= w then w = 512 end
  if h <= 0 or h ~= h then h = 512 end
  return w, h
end

function PT.parse_seed()
  local v = tonumber(PT.dlg.data.seed)
  if type(v) ~= "number" or v ~= v or v ~= math.floor(v) then v = -1 end
  return v
end

function PT.attach_lora(req)
  local d = PT.dlg.data
  local sel = d.lora_name
  if sel and sel ~= "(default)" then
    req.lora = { name = sel, weight = d.lora_weight / 100.0 }
  end
  -- LoRA 2 (multi-LoRA stacking)
  if d.lora2_enabled then
    local sel2 = d.lora2_name
    if sel2 and sel2 ~= "(default)" then
      local w2 = (d.lora2_weight or 100) / 100.0
      req.lora2 = { name = sel2, weight = w2 }
    end
  end
end

function PT.attach_ip_adapter(req)
  local d = PT.dlg.data
  if not d.ip_adapter_enabled then return end
  local scale = (d.ip_adapter_scale or 60) / 100.0
  if scale <= 0 then return end
  -- Capture active layer as reference
  local img_b64 = PT.capture_active_layer()
  if not img_b64 then return end
  req.ip_adapter_image = img_b64
  req.ip_adapter_scale = scale
  req.ip_adapter_mode = d.ip_adapter_mode or "full"
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
  -- Parse upscale factor from "2x"/"4x" string
  local upscale_factor = 4
  local uf_str = d.upscale_factor or "4x"
  local uf_num = tonumber(uf_str:match("(%d+)"))
  if uf_num then upscale_factor = uf_num end
  local pp = {
    pixelate = {
      enabled = d.pixelate,
      target_size = d.pixel_size,
      method = d.pixelate_method or "nearest",
    },
    quantize_enabled = d.quantize_enabled,
    quantize_method = d.quantize_method,
    quantize_colors = d.colors,
    dither = d.dither,
    palette = { mode = d.palette_mode },
    remove_bg = d.remove_bg,
    upscale_enabled = d.upscale_enabled or false,
    upscale_factor = upscale_factor,
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

-- ─── Locked Fields & Prompt Injection ─────────────────────

function PT.build_locked_fields()
  if not PT.dlg then return {} end
  local d = PT.dlg.data
  local fields = {}
  if d.lock_subject then
    local subj = (d.fixed_subject or ""):match("^%s*(.-)%s*$")
    if subj ~= "" then fields.subject = subj end
  end
  if d.lock_custom then
    local cust = (d.fixed_custom or ""):match("^%s*(.-)%s*$")
    if cust ~= "" then fields.custom = cust end
  end
  return fields
end

-- Inject locked fields into a prompt string based on position settings.
-- Returns the effective prompt with Subject/Custom prepended or appended.
function PT.inject_locked_prompt(prompt)
  if not PT.dlg then return prompt end
  local d = PT.dlg.data
  local result = prompt or ""

  -- Collect insertions by position
  local prefixes = {}
  local suffixes = {}

  -- Subject
  if d.lock_subject then
    local subj = (d.fixed_subject or ""):match("^%s*(.-)%s*$")
    if subj ~= "" and not result:find(subj, 1, true) then
      local pos = d.subject_position or "prefix"
      if pos == "prefix" then
        prefixes[#prefixes + 1] = subj
      elseif pos == "suffix" then
        suffixes[#suffixes + 1] = subj
      end
      -- "off" = don't inject
    end
  end

  -- Custom
  if d.lock_custom then
    local cust = (d.fixed_custom or ""):match("^%s*(.-)%s*$")
    if cust ~= "" and not result:find(cust, 1, true) then
      local pos = d.custom_position or "suffix"
      if pos == "prefix" then
        prefixes[#prefixes + 1] = cust
      elseif pos == "suffix" then
        suffixes[#suffixes + 1] = cust
      end
    end
  end

  -- Build final prompt (avoid trailing/leading commas when result is empty)
  local parts = {}
  if #prefixes > 0 then parts[#parts + 1] = table.concat(prefixes, ", ") end
  if result ~= "" then parts[#parts + 1] = result end
  if #suffixes > 0 then parts[#parts + 1] = table.concat(suffixes, ", ") end
  return table.concat(parts, ", ")
end

-- ─── Prompt Schedule Helper ───────────────────────────────

function PT.extract_prompt_schedule(total_frames, fps)
  if not PT.dlg then return nil end
  local d = PT.dlg.data
  local dsl_text = d.generate_prompt_schedule_dsl or ""
  
  -- Zéro plantage : on omet totalement le payload si vide
  if type(dsl_text) ~= "string" or dsl_text:match("^%s*$") then return nil end
  
  if not PT.dsl_parser then
    app.alert("SDDj DSL parser not loaded.")
    return nil
  end
  
  local success, sched = pcall(PT.dsl_parser.parse, dsl_text, total_frames, fps)
  if success and sched and #sched.keyframes > 0 then
    -- Inject locked fields (custom/subject) into every keyframe, even empty ones
    for _, kf in ipairs(sched.keyframes) do
      kf.prompt = PT.inject_locked_prompt(kf.prompt or "")
    end
    return sched
  elseif not success then
    app.alert("Syntax error in Prompt Schedule DSL. Please check your formatting.\n\n" .. tostring(sched))
  end
  return nil
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

  -- Inject locked fields (Subject/Custom) based on position settings
  local locked = PT.build_locked_fields()
  local effective_prompt = PT.inject_locked_prompt(d.prompt)

  -- Build modulation slots from UI (6 slots)
  local slots = {}
  local slot_count = d.mod_slot_count or 2
  for i = 1, slot_count do
    local prefix = "mod" .. i .. "_"
    if d[prefix .. "enable"] then
      local target = d[prefix .. "target"]
      local mn = PT.scale_mod_value(target, d[prefix .. "min"])
      local mx = PT.scale_mod_value(target, d[prefix .. "max"])
      -- Integer targets: floor the result
      if target == "seed_offset" or target == "frame_cadence" then
        mn = math.floor(mn)
        mx = math.floor(mx)
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

  -- Prompt schedule via global DSL
  local total_audio_frames = PT.audio and PT.audio.total_frames or 100
  local max_f = d.audio_max_frames or 0
  if max_f > 0 then total_audio_frames = max_f end
  local fps = tonumber(d.audio_fps) or 24
  local prompt_schedule = PT.extract_prompt_schedule(total_audio_frames, fps)

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
    prompt_schedule   = prompt_schedule,
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
    steps             = clamp(d.audio_steps, 1, 100),
    cfg_scale         = clamp(d.audio_cfg / 10.0, 0, 30),
    clip_skip         = clamp(d.clip_skip, 1, 12),
    denoise_strength  = clamp(d.audio_denoise / 100.0, 0, 1),

    tag_name          = tag_name,
    post_process      = PT.build_post_process(),
  }
  PT.attach_lora(req)
  PT.attach_neg_ti(req)
  PT.attach_scheduler(req)
  PT.attach_ip_adapter(req)
  PT.last_request = PT.shallow_copy_request(req)
  return req
end

-- Factored from generate button onclick + loop continuation (eliminates duplication).
function PT.build_generate_request()
  if not PT.dlg then return nil end
  local gw, gh = PT.parse_size()
  local req = {
    action           = "generate",
    prompt           = PT.inject_locked_prompt(PT.dlg.data.prompt),
    negative_prompt  = PT.dlg.data.negative_prompt,
    mode             = PT.dlg.data.mode,
    width            = gw,
    height           = gh,
    seed             = PT.parse_seed(),
    steps            = clamp(PT.dlg.data.steps, 1, 100),
    cfg_scale        = clamp(PT.dlg.data.cfg_scale / 10.0, 0, 30),
    clip_skip        = clamp(PT.dlg.data.clip_skip, 1, 12),
    denoise_strength = clamp(PT.dlg.data.denoise / 100.0, 0, 1),
    post_process     = PT.build_post_process(),
    prompt_schedule  = nil,  -- O-06: skip schedule parse for single-image
  }
  PT.attach_lora(req)
  PT.attach_neg_ti(req)
  PT.attach_scheduler(req)
  PT.attach_ip_adapter(req)
  PT.last_request = PT.shallow_copy_request(req)
  return req
end

-- Factored from animate button onclick (consistent with build_generate_request).
function PT.build_animation_request()
  if not PT.dlg then return nil end
  local d = PT.dlg.data
  local gw, gh = PT.parse_size()
  local tag_name = d.anim_tag or ""
  if tag_name == "" then tag_name = nil end

  -- Inject locked fields (Subject/Custom) based on position settings
  local effective_prompt = PT.inject_locked_prompt(d.prompt)

  -- Prompt schedule via global DSL
  local fps = 24
  if d.anim_duration and d.anim_duration > 0 then
    fps = 1000.0 / d.anim_duration
  end
  local prompt_schedule = PT.extract_prompt_schedule(d.anim_frames or 100, fps)

  -- AnimateDiff Lightning frame cap warning (max 32 frames)
  local frame_count = clamp(d.anim_frames, 2, 256)
  if d.anim_method == "animatediff" and frame_count > 32 then
    frame_count = 32
    PT.update_status("AnimateDiff Lightning: max 32 frames — clamped")
  end

  local req = {
    action = "generate_animation",
    method = d.anim_method,
    prompt = effective_prompt,
    negative_prompt = d.negative_prompt,
    mode = d.mode,
    width = gw, height = gh,
    seed = PT.parse_seed(),
    steps = clamp(d.anim_steps, 1, 100),
    cfg_scale = clamp(d.anim_cfg / 10.0, 0, 30),
    clip_skip = clamp(d.clip_skip, 1, 12),
    denoise_strength = clamp(d.anim_denoise / 100.0, 0, 1),
    frame_count = frame_count,
    frame_duration_ms = d.anim_duration,
    seed_strategy = d.anim_seed_strategy,
    tag_name = tag_name,
    enable_freeinit = d.anim_freeinit,
    freeinit_iterations = d.anim_freeinit_iters,
    prompt_schedule = prompt_schedule,
    post_process = PT.build_post_process(),
  }
  -- Frame interpolation
  local interp = d.frame_interpolation
  if interp and interp ~= "None" then
    local match = interp:match("%d+")
    if match then req.interpolation_factor = tonumber(match) end
  end
  PT.attach_lora(req)
  PT.attach_neg_ti(req)
  PT.attach_scheduler(req)
  PT.attach_ip_adapter(req)
  -- Attach guidance start/end for ControlNet animation modes
  if d.mode and d.mode:find("controlnet_") then
    req.control_guidance_start = clamp((d.anim_guidance_start or 0) / 100.0, 0, 1)
    req.control_guidance_end   = clamp((d.anim_guidance_end or 80) / 100.0, 0, 1)
  end
  PT.last_request = PT.shallow_copy_request(req)
  return req
end

-- ─── QR / Illusion Art Request ───────────────────────────────

function PT.build_qr_request()
  if not PT.dlg then return nil end
  local d = PT.dlg.data
  local gw, gh = PT.parse_size()
  local use_source = d.qr_use_source or false
  local req = {
    action                        = "generate",
    mode                          = "controlnet_qrcode",
    prompt                        = PT.inject_locked_prompt(d.prompt),
    negative_prompt               = d.negative_prompt,
    width                         = gw,
    height                        = gh,
    seed                          = PT.parse_seed(),
    steps                         = clamp(d.qr_steps or 20, 1, 100),
    cfg_scale                     = clamp(d.qr_cfg / 10.0, 0, 30),
    clip_skip                     = clamp(d.clip_skip, 1, 12),
    denoise_strength              = use_source and clamp(d.qr_denoise / 100.0, 0, 1) or 1.0,
    controlnet_conditioning_scale = clamp(d.qr_conditioning_scale / 100.0, 0, 3),
    control_guidance_start        = clamp(d.qr_guidance_start / 100.0, 0, 1),
    control_guidance_end          = clamp(d.qr_guidance_end / 100.0, 0, 1),
    post_process                  = PT.build_post_process(),
  }
  PT.attach_lora(req)
  PT.attach_neg_ti(req)
  PT.attach_scheduler(req)
  PT.attach_ip_adapter(req)
  PT.last_request = PT.shallow_copy_request(req)
  return req, use_source
end

end
