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
    live_strength      = d.live_strength,
    live_steps         = d.live_steps,
    live_cfg           = d.live_cfg,
    live_opacity       = d.live_opacity,
    live_mode          = d.live_mode,
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
    audio_fps          = d.audio_fps,
    audio_steps        = d.audio_steps,
    audio_cfg          = d.audio_cfg,
    audio_denoise      = d.audio_denoise,
    audio_stems_enable = d.audio_stems_enable,
    audio_frame_duration = d.audio_frame_duration,
    audio_max_frames   = d.audio_max_frames,
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
    mod1_enable = d.mod1_enable, mod1_source = d.mod1_source, mod1_target = d.mod1_target,
    mod1_min = d.mod1_min, mod1_max = d.mod1_max, mod1_attack = d.mod1_attack, mod1_release = d.mod1_release,
    mod2_enable = d.mod2_enable, mod2_source = d.mod2_source, mod2_target = d.mod2_target,
    mod2_min = d.mod2_min, mod2_max = d.mod2_max, mod2_attack = d.mod2_attack, mod2_release = d.mod2_release,
    mod3_enable = d.mod3_enable, mod3_source = d.mod3_source, mod3_target = d.mod3_target,
    mod3_min = d.mod3_min, mod3_max = d.mod3_max, mod3_attack = d.mod3_attack, mod3_release = d.mod3_release,
    mod4_enable = d.mod4_enable, mod4_source = d.mod4_source, mod4_target = d.mod4_target,
    mod4_min = d.mod4_min, mod4_max = d.mod4_max, mod4_attack = d.mod4_attack, mod4_release = d.mod4_release,
    -- Audio method & FreeInit
    audio_method         = d.audio_method,
    audio_freeinit       = d.audio_freeinit,
    audio_freeinit_iters = d.audio_freeinit_iters,
    -- Audio advanced sub-fields
    audio_random_seed    = d.audio_random_seed,
    audio_prompt_schedule = d.audio_prompt_schedule,
    ps1_time = d.ps1_time, ps1_prompt = d.ps1_prompt,
    ps2_time = d.ps2_time, ps2_prompt = d.ps2_prompt,
    ps3_time = d.ps3_time, ps3_prompt = d.ps3_prompt,
    -- MP4 export
    mp4_quality          = d.mp4_quality,
    mp4_scale            = d.mp4_scale,
  }
  local ok, encoded = pcall(PT.json.encode, s)
  if not ok then return end
  local f = io.open(PT.cfg.SETTINGS_FILE, "w")
  if f then
    f:write(encoded)
    f:close()
  end
end

function PT.load_settings()
  local f = io.open(PT.cfg.SETTINGS_FILE, "r")
  if not f then
    -- Migration: try old settings file from pre-0.7.5 (PixyToon era)
    local old_path = app.fs.joinPath(app.fs.userConfigPath, "pixytoon_settings.json")
    f = io.open(old_path, "r")
    if f then
      -- Read old settings, will be saved under new name on next save
      local data = f:read("*a")
      f:close()
      local ok, s = pcall(PT.json.decode, data)
      if ok and type(s) == "table" then return s end
    end
    return nil
  end
  local data = f:read("*a")
  f:close()
  local ok, s = pcall(PT.json.decode, data)
  if not ok or type(s) ~= "table" then return nil end
  return s
end

function PT.apply_settings(s)
  if not s or not PT.dlg then return end
  -- Text fields
  local texts = { "server_url", "prompt", "negative_prompt", "seed", "fixed_subject", "palette_custom_colors", "anim_tag",
                   "expr_denoise", "expr_cfg", "expr_noise", "expr_controlnet", "expr_seed",
                   "expr_palette", "expr_cadence", "expr_motion_x", "expr_motion_y",
                   "expr_motion_zoom", "expr_motion_rot",
                   "ps1_time", "ps1_prompt", "ps2_time", "ps2_prompt",
                   "ps3_time", "ps3_prompt" }
  for _, id in ipairs(texts) do
    if s[id] ~= nil then PT.dlg:modify{ id = id, text = s[id] } end
  end
  -- Option (combobox) fields
  local opts = {
    "mode", "output_size", "output_mode", "quantize_method", "dither", "palette_mode",
    "palette_name", "lora_name", "anim_method", "anim_seed_strategy", "preset_name",
    "loop_seed_combo", "live_mode",
    "audio_fps", "audio_mod_preset", "audio_method",
    "mp4_quality", "mp4_scale",
    "mod1_source", "mod1_target", "mod2_source", "mod2_target",
    "mod3_source", "mod3_target", "mod4_source", "mod4_target",
  }
  for _, id in ipairs(opts) do
    if s[id] ~= nil then PT.dlg:modify{ id = id, option = s[id] } end
  end
  -- Numeric value (slider) fields
  local vals = {
    "steps", "cfg_scale", "clip_skip", "denoise", "lora_weight",
    "neg_ti_weight", "pixel_size", "colors",
    "anim_steps", "anim_cfg", "anim_frames", "anim_duration", "anim_denoise", "anim_freeinit_iters",
    "live_strength", "live_steps", "live_cfg", "live_opacity",
    "randomness",
    "audio_steps", "audio_cfg", "audio_denoise",
    "audio_frame_duration", "audio_max_frames", "audio_freeinit_iters",
    "mod_slot_count",
    "mod1_min", "mod1_max", "mod1_attack", "mod1_release",
    "mod2_min", "mod2_max", "mod2_attack", "mod2_release",
    "mod3_min", "mod3_max", "mod3_attack", "mod3_release",
    "mod4_min", "mod4_max", "mod4_attack", "mod4_release",
  }
  for _, id in ipairs(vals) do
    if s[id] ~= nil then PT.dlg:modify{ id = id, value = s[id] } end
  end
  -- Boolean (checkbox) fields
  local bools = { "use_neg_ti", "pixelate", "remove_bg", "lock_subject", "anim_freeinit",
                   "randomize_before", "loop_check", "random_loop_check", "save_output",
                   "audio_stems_enable", "audio_advanced", "audio_use_expressions",
                   "audio_freeinit", "audio_random_seed", "audio_prompt_schedule",
                   "mod1_enable", "mod2_enable", "mod3_enable", "mod4_enable" }
  for _, id in ipairs(bools) do
    if s[id] ~= nil then PT.dlg:modify{ id = id, selected = s[id] } end
  end
  -- Update slider labels (dlg:modify{value=...} doesn't fire onchange)
  local d = PT.dlg.data
  PT.dlg:modify{ id = "cfg_scale", label = string.format("CFG (%.1f)", d.cfg_scale / 10.0) }
  PT.dlg:modify{ id = "denoise", label = string.format("Strength (%.2f)", d.denoise / 100.0) }
  PT.dlg:modify{ id = "lora_weight", label = string.format("LoRA (%.2f)", d.lora_weight / 100.0) }
  PT.dlg:modify{ id = "neg_ti_weight", label = string.format("Emb. (%.2f)", d.neg_ti_weight / 100.0) }
  PT.dlg:modify{ id = "pixel_size", label = "Target (" .. d.pixel_size .. "px)" }
  PT.dlg:modify{ id = "colors", label = "Colors (" .. d.colors .. ")" }
  PT.dlg:modify{ id = "anim_cfg", label = string.format("CFG (%.1f)", d.anim_cfg / 10.0) }
  PT.dlg:modify{ id = "anim_denoise", label = string.format("Strength (%.2f)", d.anim_denoise / 100.0) }
  PT.dlg:modify{ id = "audio_cfg", label = string.format("CFG (%.1f)", d.audio_cfg / 10.0) }
  PT.dlg:modify{ id = "audio_denoise", label = string.format("Strength (%.2f)", d.audio_denoise / 100.0) }
  PT.dlg:modify{ id = "live_strength", label = string.format("Strength (%.2f)", d.live_strength / 100.0) }
  PT.dlg:modify{ id = "live_cfg", label = string.format("CFG (%.1f)", d.live_cfg / 10.0) }
  PT.dlg:modify{ id = "live_opacity", label = string.format("Preview (%d%%)", d.live_opacity) }
  PT.dlg:modify{ id = "steps", label = "Steps (" .. d.steps .. ")" }
  PT.dlg:modify{ id = "clip_skip", label = "CLIP Skip (" .. d.clip_skip .. ")" }
  PT.dlg:modify{ id = "anim_steps", label = "Steps (" .. d.anim_steps .. ")" }
  PT.dlg:modify{ id = "anim_frames", label = "Frames (" .. d.anim_frames .. ")" }
  PT.dlg:modify{ id = "anim_duration", label = "Duration (" .. d.anim_duration .. "ms)" }
  PT.dlg:modify{ id = "audio_steps", label = "Steps (" .. d.audio_steps .. ")" }
  PT.dlg:modify{ id = "mod_slot_count", label = "Slots (" .. d.mod_slot_count .. ")" }
  -- Sync visibility for conditional fields
  if s.use_neg_ti ~= nil then
    PT.dlg:modify{ id = "neg_ti_weight", visible = s.use_neg_ti }
  end
  if s.pixelate ~= nil then
    PT.dlg:modify{ id = "pixel_size", visible = s.pixelate }
  end
  if s.lock_subject ~= nil then
    PT.dlg:modify{ id = "fixed_subject", visible = s.lock_subject }
  end
  if s.palette_mode ~= nil then
    PT.dlg:modify{ id = "palette_name", visible = (s.palette_mode == "preset") }
    PT.dlg:modify{ id = "palette_custom_colors", visible = (s.palette_mode == "custom") }
  end
  if s.anim_method ~= nil then
    local ad = (s.anim_method == "animatediff")
    PT.dlg:modify{ id = "anim_freeinit", visible = ad }
    PT.dlg:modify{ id = "anim_freeinit_iters", visible = ad }
  end
  -- Sync audio method → audio freeinit visibility
  if s.audio_method ~= nil then
    local ad = (s.audio_method == "animatediff")
    PT.dlg:modify{ id = "audio_freeinit", visible = ad }
    PT.dlg:modify{ id = "audio_freeinit_iters", visible = ad and (s.audio_freeinit == true) }
  end
  -- Sync loop seed combo visibility
  local show_loop_seed = (s.loop_check == true) or (s.random_loop_check == true)
  PT.dlg:modify{ id = "loop_seed_combo", visible = show_loop_seed }
  -- Sync audio slot + advanced visibility (uses shared helper from dialog)
  if s.mod_slot_count or s.audio_advanced ~= nil or s.audio_use_expressions ~= nil then
    local n = PT.dlg.data.mod_slot_count
    local adv = PT.dlg.data.audio_advanced
    for i = 1, 4 do
      local vis = (i <= n)
      PT.dlg:modify{ id = "mod" .. i .. "_enable",  visible = vis }
      PT.dlg:modify{ id = "mod" .. i .. "_source",  visible = vis }
      PT.dlg:modify{ id = "mod" .. i .. "_target",  visible = vis }
      PT.dlg:modify{ id = "mod" .. i .. "_min",     visible = vis }
      PT.dlg:modify{ id = "mod" .. i .. "_max",     visible = vis }
      PT.dlg:modify{ id = "mod" .. i .. "_attack",  visible = vis and adv }
      PT.dlg:modify{ id = "mod" .. i .. "_release", visible = vis and adv }
    end
    PT.dlg:modify{ id = "audio_use_expressions", visible = adv }
    local expr_vis = adv and PT.dlg.data.audio_use_expressions
    local expr_ids = {
      "expr_denoise", "expr_cfg", "expr_noise", "expr_controlnet",
      "expr_seed", "expr_palette", "expr_cadence",
      "expr_motion_x", "expr_motion_y", "expr_motion_zoom", "expr_motion_rot",
    }
    for _, eid in ipairs(expr_ids) do
      PT.dlg:modify{ id = eid, visible = expr_vis }
    end
    -- Sync audio_random_seed visibility (advanced only)
    PT.dlg:modify{ id = "audio_random_seed", visible = adv }
    -- Sync audio_prompt_schedule + sub-fields visibility
    PT.dlg:modify{ id = "audio_prompt_schedule", visible = adv }
    local ps_vis = adv and PT.dlg.data.audio_prompt_schedule
    for i = 1, 3 do
      PT.dlg:modify{ id = "ps" .. i .. "_time", visible = ps_vis }
      PT.dlg:modify{ id = "ps" .. i .. "_prompt", visible = ps_vis }
    end
  end
  -- Sync max frames label
  if s.audio_max_frames then
    local v = s.audio_max_frames
    PT.dlg:modify{ id = "audio_max_frames",
      label = v == 0 and "Max Frames (0=all)" or ("Max Frames (" .. v .. ")") }
  end
  -- Sync frame duration label
  if s.audio_frame_duration then
    PT.dlg:modify{ id = "audio_frame_duration",
      label = "Frame (" .. s.audio_frame_duration .. "ms)" }
  end
  -- Sync denoise visibility based on mode
  if s.mode then
    PT.dlg:modify{ id = "denoise", visible = (s.mode ~= "txt2img") }
    -- Mode label hint
    if s.mode == "inpaint" then
      PT.dlg:modify{ id = "mode", label = "Mode (needs mask)" }
    elseif s.mode == "img2img" or (s.mode and s.mode:find("controlnet_")) then
      PT.dlg:modify{ id = "mode", label = "Mode (needs layer)" }
    else
      PT.dlg:modify{ id = "mode", label = "Mode" }
    end
  end
  -- Sync randomness slider visibility
  local show_rand = (s.randomize_before == true) or (s.random_loop_check == true)
  PT.dlg:modify{ id = "randomness", visible = show_rand }
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
end

end
