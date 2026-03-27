--
-- SDDj — Settings Persistence
--

return function(PT)

function PT.save_settings()
  if not PT.dlg then return end
  local d = PT.dlg.data
  local s = {
    server_url         = d.server_url,
    prompt             = d.prompt,
    negative_prompt    = d.negative_prompt,
    mode               = d.mode,
    output_size        = d.output_size,
    seed               = d.seed,
    steps              = d.steps,
    cfg_scale          = d.cfg_scale,
    clip_skip          = d.clip_skip,
    denoise            = d.denoise,
    lora_name          = d.lora_name,
    lora_weight        = d.lora_weight,
    use_neg_ti         = d.use_neg_ti,
    neg_ti_weight      = d.neg_ti_weight,
    pixelate           = d.pixelate,
    pixel_size         = d.pixel_size,
    colors             = d.colors,
    quantize_method    = d.quantize_method,
    dither             = d.dither,
    palette_mode       = d.palette_mode,
    palette_name       = d.palette_name,
    palette_custom_colors = d.palette_custom_colors,
    remove_bg          = d.remove_bg,
    anim_method        = d.anim_method,
    anim_steps         = d.anim_steps,
    anim_cfg           = d.anim_cfg,
    anim_frames        = d.anim_frames,
    anim_duration      = d.anim_duration,
    anim_denoise       = d.anim_denoise,
    anim_seed_strategy = d.anim_seed_strategy,
    preset_name        = d.preset_name,
    lock_subject       = d.lock_subject,
    fixed_subject      = d.fixed_subject,
    anim_tag           = d.anim_tag,
    anim_freeinit      = d.anim_freeinit,
    anim_freeinit_iters = d.anim_freeinit_iters,
    randomize_before   = d.randomize_before,
    randomness         = d.randomness,
    loop_seed_combo    = d.loop_seed_combo,
    loop_check         = d.loop_check,
    random_loop_check  = d.random_loop_check,
    output_mode        = d.output_mode,
    -- Output
    save_output        = d.save_output,
    -- Audio tab
    audio_file         = d.audio_file,
    audio_fps          = d.audio_fps,
    audio_steps        = d.audio_steps,
    audio_cfg          = d.audio_cfg,
    audio_denoise      = d.audio_denoise,
    audio_stems_enable = d.audio_stems_enable,

    audio_max_frames   = d.audio_max_frames,
    audio_tag          = d.audio_tag,
    audio_mod_preset   = d.audio_mod_preset,
    mod_slot_count     = d.mod_slot_count,
    audio_advanced     = d.audio_advanced,
    audio_use_expressions = d.audio_use_expressions,
    expr_denoise       = d.expr_denoise,
    expr_cfg           = d.expr_cfg,
    expr_noise         = d.expr_noise,
    expr_controlnet    = d.expr_controlnet,
    expr_seed          = d.expr_seed,
    expr_palette       = d.expr_palette,
    expr_cadence       = d.expr_cadence,
    expr_motion_x      = d.expr_motion_x,
    expr_motion_y      = d.expr_motion_y,
    expr_motion_zoom   = d.expr_motion_zoom,
    expr_motion_rot    = d.expr_motion_rot,
    expr_motion_tilt_x = d.expr_motion_tilt_x,
    expr_motion_tilt_y = d.expr_motion_tilt_y,
  }
  -- Modulation slots (loop over 6 slots × 8 fields)
  local _mod_fields = {"enable", "source", "target", "min", "max", "attack", "release", "invert"}
  for i = 1, 6 do
    for _, f in ipairs(_mod_fields) do
      local key = "mod" .. i .. "_" .. f
      s[key] = d[key]
    end
  end
  s.quantize_enabled     = d.quantize_enabled
  s.audio_choreography   = d.audio_choreography
  s.audio_expr_preset    = d.audio_expr_preset
  -- Audio method & FreeInit
  s.audio_method         = d.audio_method
  s.audio_freeinit       = d.audio_freeinit
  s.audio_freeinit_iters = d.audio_freeinit_iters
  -- Audio advanced sub-fields
  s.audio_random_seed    = d.audio_random_seed
  -- Prompt schedule DSL text (shared across Generate/Animation/Audio tabs)
  s.generate_prompt_schedule_dsl = d.generate_prompt_schedule_dsl
  -- MP4 export
  s.mp4_quality          = d.mp4_quality
  s.mp4_scale            = d.mp4_scale
  -- QR Code tab
  s.qr_use_source         = d.qr_use_source
  s.qr_denoise            = d.qr_denoise
  s.qr_conditioning_scale = d.qr_conditioning_scale
  s.qr_guidance_start     = d.qr_guidance_start
  s.qr_guidance_end       = d.qr_guidance_end
  s.qr_steps              = d.qr_steps
  s.qr_cfg                = d.qr_cfg
  local ok, encoded = pcall(PT.json.encode, s)
  if not ok then return end
  local f, ferr = io.open(PT.cfg.SETTINGS_FILE, "w")
  if f then
    local wok, werr = pcall(function() f:write(encoded); f:close() end)
    if not wok then
      PT.update_status("Settings save error: " .. tostring(werr))
    end
  else
    PT.update_status("Cannot save settings: " .. tostring(ferr))
  end
end

function PT.load_settings()
  local f = io.open(PT.cfg.SETTINGS_FILE, "r")
  if not f then return nil end
  local data = f:read("*a")
  f:close()
  local ok, s = pcall(PT.json.decode, data)
  if not ok or type(s) ~= "table" then return nil end
  return s
end

function PT.apply_settings(s)
  if not s or not PT.dlg then return end
  -- Text fields
  local texts = { "server_url", "prompt", "negative_prompt", "seed", "fixed_subject", "palette_custom_colors", "anim_tag", "audio_file", "audio_tag",
                   "expr_denoise", "expr_cfg", "expr_noise", "expr_controlnet", "expr_seed",
                   "expr_palette", "expr_cadence", "expr_motion_x", "expr_motion_y",
                   "expr_motion_zoom", "expr_motion_rot",
                   "expr_motion_tilt_x", "expr_motion_tilt_y",
                   "generate_prompt_schedule_dsl" }
  for _, id in ipairs(texts) do
    if s[id] ~= nil then PT.dlg:modify{ id = id, text = s[id] } end
  end
  -- Option (combobox) fields
  local opts = {
    "mode", "output_size", "output_mode", "quantize_method", "dither", "palette_mode",
    "palette_name", "lora_name", "anim_method", "anim_seed_strategy", "preset_name",
    "loop_seed_combo",
    "audio_fps", "audio_mod_preset", "audio_method",
    "audio_choreography", "audio_expr_preset",
    "mp4_quality", "mp4_scale",
  }
  -- Modulation slot comboboxes
  for i = 1, 6 do
    opts[#opts + 1] = "mod" .. i .. "_source"
    opts[#opts + 1] = "mod" .. i .. "_target"
  end
  for _, id in ipairs(opts) do
    if s[id] ~= nil then PT.dlg:modify{ id = id, option = s[id] } end
  end
  -- Numeric value (slider) fields
  local vals = {
    "steps", "cfg_scale", "clip_skip", "denoise", "lora_weight",
    "neg_ti_weight", "pixel_size", "colors",
    "anim_steps", "anim_cfg", "anim_frames", "anim_duration", "anim_denoise", "anim_freeinit_iters",
    "randomness",
    "audio_steps", "audio_cfg", "audio_denoise",
    "audio_max_frames", "audio_freeinit_iters",
    "mod_slot_count",
    "qr_conditioning_scale", "qr_guidance_start", "qr_guidance_end",
    "qr_steps", "qr_cfg", "qr_denoise",
  }
  -- Modulation slot sliders
  for i = 1, 6 do
    for _, f in ipairs({"min", "max", "attack", "release"}) do
      vals[#vals + 1] = "mod" .. i .. "_" .. f
    end
  end
  for _, id in ipairs(vals) do
    if s[id] ~= nil then PT.dlg:modify{ id = id, value = s[id] } end
  end
  -- Boolean (checkbox) fields
  local bools = { "use_neg_ti", "pixelate", "quantize_enabled", "remove_bg", "lock_subject", "anim_freeinit",
                   "randomize_before", "loop_check", "random_loop_check", "save_output",
                   "audio_stems_enable", "audio_advanced", "audio_use_expressions",
                   "audio_freeinit", "audio_random_seed",
                   "qr_use_source",
  }
  -- Modulation slot booleans
  for i = 1, 6 do
    bools[#bools + 1] = "mod" .. i .. "_enable"
    bools[#bools + 1] = "mod" .. i .. "_invert"
  end
  for _, id in ipairs(bools) do
    if s[id] ~= nil then PT.dlg:modify{ id = id, selected = s[id] } end
  end
  -- Sync all slider labels from registry (single source of truth)
  for id in pairs(PT.SLIDER_LABELS) do
    PT.sync_slider_label(id)
  end
  -- Special cases not in registry (custom display logic)
  local d = PT.dlg.data
  PT.dlg:modify{ id = "audio_max_frames",
    label = d.audio_max_frames == 0 and "Max Frames (0=all)" or ("Max Frames (" .. d.audio_max_frames .. ")") }

  -- Mode label hint
  if s.mode then
    if s.mode == "inpaint" then
      PT.dlg:modify{ id = "mode", label = "Mode (needs mask)" }
    elseif s.mode == "controlnet_qrcode" then
      PT.dlg:modify{ id = "mode", label = "Mode (QR)" }
    elseif s.mode == "img2img" or (s.mode and s.mode:find("controlnet_")) then
      PT.dlg:modify{ id = "mode", label = "Mode (needs layer)" }
    else
      PT.dlg:modify{ id = "mode", label = "Mode" }
    end
  end
  -- Sync randomness label
  if s.randomness then
    local v = s.randomness
    local names = { [0]="Off", [5]="Subtle", [10]="Moderate", [15]="Wild", [20]="Chaos" }
    local name = names[v] or ""
    local suffix = name ~= "" and (" — " .. name) or ""
    PT.dlg:modify{ id = "randomness", label = "Randomness (" .. v .. suffix .. ")" }
  end
  -- Sync output state
  if s.save_output ~= nil then
    PT.output.enabled = s.save_output
  end
  -- Migration v0.7.5→v0.7.7: copy shared sliders to dedicated pipeline sliders
  if s.anim_steps == nil and s.steps then
    PT.dlg:modify{ id = "anim_steps", value = s.steps }
  end
  if s.anim_cfg == nil and s.cfg_scale then
    PT.dlg:modify{ id = "anim_cfg", value = s.cfg_scale }
    PT.dlg:modify{ id = "anim_cfg", label = string.format("CFG (%.1f)", s.cfg_scale / 10.0) }
  end
  if s.audio_steps == nil and s.steps then
    PT.dlg:modify{ id = "audio_steps", value = s.steps }
  end
  if s.audio_cfg == nil and s.cfg_scale then
    PT.dlg:modify{ id = "audio_cfg", value = s.cfg_scale }
    PT.dlg:modify{ id = "audio_cfg", label = string.format("CFG (%.1f)", s.cfg_scale / 10.0) }
  end
  if s.audio_denoise == nil and s.denoise then
    PT.dlg:modify{ id = "audio_denoise", value = s.denoise }
    PT.dlg:modify{ id = "audio_denoise", label = string.format("Strength (%.2f)", s.denoise / 100.0) }
  end

  -- Initialize prompt schedule state from restored DSL text
  if PT.update_schedule_state then
    local restored_dsl = PT.dlg.data.generate_prompt_schedule_dsl or ""
    if restored_dsl ~= "" then
      PT.update_schedule_state(restored_dsl)
    end
  end
end

end
