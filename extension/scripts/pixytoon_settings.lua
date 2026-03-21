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
  local texts = { "server_url", "prompt", "negative_prompt", "seed", "fixed_subject", "palette_custom_colors", "anim_tag" }
  for _, id in ipairs(texts) do
    if s[id] ~= nil then PT.dlg:modify{ id = id, text = s[id] } end
  end
  -- Option (combobox) fields
  local opts = {
    "mode", "output_size", "quantize_method", "dither", "palette_mode",
    "palette_name", "lora_name", "anim_method", "anim_seed_strategy", "preset_name",
    "loop_seed_combo", "live_mode",
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
  }
  for _, id in ipairs(vals) do
    if s[id] ~= nil then PT.dlg:modify{ id = id, value = s[id] } end
  end
  -- Boolean (checkbox) fields
  local bools = { "use_neg_ti", "pixelate", "remove_bg", "lock_subject", "anim_freeinit",
                   "loop_check", "random_loop_check" }
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
end

end
