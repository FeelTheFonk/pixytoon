--
-- PixyToon — Settings Persistence
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
    loop_seed_combo    = d.loop_seed_combo,
    loop_check         = d.loop_check,
    random_loop_check  = d.random_loop_check,
    output_mode        = d.output_mode,
    -- Audio tab
    audio_fps          = d.audio_fps,
    audio_stems_enable = d.audio_stems_enable,
    audio_frame_duration = d.audio_frame_duration,
    audio_mod_preset   = d.audio_mod_preset,
    mod_slot_count     = d.mod_slot_count,
    audio_advanced     = d.audio_advanced,
    audio_use_expressions = d.audio_use_expressions,
    expr_denoise       = d.expr_denoise,
    expr_cfg           = d.expr_cfg,
    expr_noise         = d.expr_noise,
    mod1_enable = d.mod1_enable, mod1_source = d.mod1_source, mod1_target = d.mod1_target,
    mod1_min = d.mod1_min, mod1_max = d.mod1_max, mod1_attack = d.mod1_attack, mod1_release = d.mod1_release,
    mod2_enable = d.mod2_enable, mod2_source = d.mod2_source, mod2_target = d.mod2_target,
    mod2_min = d.mod2_min, mod2_max = d.mod2_max, mod2_attack = d.mod2_attack, mod2_release = d.mod2_release,
    mod3_enable = d.mod3_enable, mod3_source = d.mod3_source, mod3_target = d.mod3_target,
    mod3_min = d.mod3_min, mod3_max = d.mod3_max, mod3_attack = d.mod3_attack, mod3_release = d.mod3_release,
    mod4_enable = d.mod4_enable, mod4_source = d.mod4_source, mod4_target = d.mod4_target,
    mod4_min = d.mod4_min, mod4_max = d.mod4_max, mod4_attack = d.mod4_attack, mod4_release = d.mod4_release,
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
  local texts = { "server_url", "prompt", "negative_prompt", "seed", "fixed_subject", "palette_custom_colors", "anim_tag",
                   "expr_denoise", "expr_cfg", "expr_noise" }
  for _, id in ipairs(texts) do
    if s[id] ~= nil then PT.dlg:modify{ id = id, text = s[id] } end
  end
  -- Option (combobox) fields
  local opts = {
    "mode", "output_size", "output_mode", "quantize_method", "dither", "palette_mode",
    "palette_name", "lora_name", "anim_method", "anim_seed_strategy", "preset_name",
    "loop_seed_combo", "live_mode",
    "audio_fps", "audio_mod_preset",
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
    "anim_frames", "anim_duration", "anim_denoise", "anim_freeinit_iters",
    "live_strength", "live_steps", "live_cfg", "live_opacity",
    "audio_frame_duration", "mod_slot_count",
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
                   "loop_check", "random_loop_check",
                   "audio_stems_enable", "audio_advanced", "audio_use_expressions",
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
  PT.dlg:modify{ id = "anim_denoise", label = string.format("Strength (%.2f)", d.anim_denoise / 100.0) }
  PT.dlg:modify{ id = "live_strength", label = string.format("Strength (%.2f)", d.live_strength / 100.0) }
  PT.dlg:modify{ id = "live_cfg", label = string.format("CFG (%.1f)", d.live_cfg / 10.0) }
  PT.dlg:modify{ id = "live_opacity", label = string.format("Preview (%d%%)", d.live_opacity) }
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
    PT.dlg:modify{ id = "expr_denoise", visible = adv and PT.dlg.data.audio_use_expressions }
    PT.dlg:modify{ id = "expr_cfg",     visible = adv and PT.dlg.data.audio_use_expressions }
    PT.dlg:modify{ id = "expr_noise",   visible = adv and PT.dlg.data.audio_use_expressions }
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
end

end
