--
-- SDDj — Settings Persistence
--

return function(PT)

local _SETTINGS_VERSION = 2

-- Field types: "text" (entry), "option" (combobox), "value" (slider), "bool" (check)
local _FIELD_SCHEMA = {
  -- Connection
  { "server_url",          "text" },
  -- Generate tab
  { "prompt",              "text" },
  { "negative_prompt",     "text" },
  { "seed",                "text" },
  { "fixed_subject",       "text" },
  { "mode",                "option" },
  { "output_size",         "option" },
  { "output_mode",         "option" },
  { "lora_name",           "option" },
  { "preset_name",         "option" },
  { "loop_seed_combo",     "option" },
  { "steps",               "value" },
  { "cfg_scale",           "value" },
  { "clip_skip",           "value" },
  { "denoise",             "value" },
  { "lora_weight",         "value" },
  { "neg_ti_weight",       "value" },
  { "randomness",          "value" },
  { "use_neg_ti",          "bool" },
  { "lock_subject",        "bool" },
  { "randomize_before",    "bool" },
  { "loop_check",          "bool" },
  { "random_loop_check",   "bool" },
  { "save_output",         "bool" },
  -- Post-Process tab
  { "pixel_size",          "value" },
  { "colors",              "value" },
  { "quantize_method",     "option" },
  { "dither",              "option" },
  { "palette_mode",        "option" },
  { "palette_name",        "option" },
  { "palette_custom_colors", "text" },
  { "pixelate",            "bool" },
  { "quantize_enabled",    "bool" },
  { "remove_bg",           "bool" },
  -- Animation tab
  { "anim_method",         "option" },
  { "anim_seed_strategy",  "option" },
  { "anim_tag",            "text" },
  { "anim_steps",          "value" },
  { "anim_cfg",            "value" },
  { "anim_frames",         "value" },
  { "anim_duration",       "value" },
  { "anim_denoise",        "value" },
  { "anim_freeinit_iters", "value" },
  { "anim_freeinit",       "bool" },
  -- Audio tab
  { "audio_file",          "text" },
  { "audio_tag",           "text" },
  { "audio_fps",           "option" },
  { "audio_mod_preset",    "option" },
  { "audio_method",        "option" },
  { "audio_choreography",  "option" },
  { "audio_expr_preset",   "option" },
  { "mp4_quality",         "option" },
  { "mp4_scale",           "option" },
  { "audio_steps",         "value" },
  { "audio_cfg",           "value" },
  { "audio_denoise",       "value" },
  { "audio_max_frames",    "value" },
  { "audio_freeinit_iters","value" },
  { "mod_slot_count",      "value" },
  { "audio_stems_enable",  "bool" },
  { "audio_advanced",      "bool" },
  { "audio_use_expressions","bool" },
  { "audio_freeinit",      "bool" },
  { "audio_random_seed",   "bool" },
  -- Expression fields
  { "expr_denoise",        "text" },
  { "expr_cfg",            "text" },
  { "expr_noise",          "text" },
  { "expr_controlnet",     "text" },
  { "expr_seed",           "text" },
  { "expr_palette",        "text" },
  { "expr_cadence",        "text" },
  { "expr_motion_x",       "text" },
  { "expr_motion_y",       "text" },
  { "expr_motion_zoom",    "text" },
  { "expr_motion_rot",     "text" },
  { "expr_motion_tilt_x",  "text" },
  { "expr_motion_tilt_y",  "text" },
  -- Schedule
  { "generate_prompt_schedule_dsl", "text" },
  -- QR Code tab
  { "qr_use_source",         "bool" },
  { "qr_denoise",            "value" },
  { "qr_conditioning_scale", "value" },
  { "qr_guidance_start",     "value" },
  { "qr_guidance_end",       "value" },
  { "qr_steps",              "value" },
  { "qr_cfg",                "value" },
}

-- Modulation slot sub-fields and their types
local _MOD_FIELDS = { "enable", "source", "target", "min", "max", "attack", "release", "invert" }
local _MOD_TYPES = {
  enable = "bool", invert = "bool",
  source = "option", target = "option",
  min = "value", max = "value", attack = "value", release = "value",
}

-- Apply a single field value to the dialog with type validation + pcall for comboboxes
local function _apply_field(dlg, id, ft, val)
  if val == nil then return end
  if ft == "text" then
    if type(val) == "string" then dlg:modify{ id = id, text = val } end
  elseif ft == "option" then
    if type(val) == "string" then pcall(dlg.modify, dlg, { id = id, option = val }) end
  elseif ft == "value" then
    if type(val) == "number" then dlg:modify{ id = id, value = val } end
  elseif ft == "bool" then
    if type(val) == "boolean" then dlg:modify{ id = id, selected = val } end
  end
end

function PT.save_settings()
  if not PT.dlg then return end
  local d = PT.dlg.data
  local s = { _version = _SETTINGS_VERSION }
  -- Core fields from schema
  for _, field in ipairs(_FIELD_SCHEMA) do
    s[field[1]] = d[field[1]]
  end
  -- Modulation slots (6 × 8 fields)
  for i = 1, 6 do
    for _, f in ipairs(_MOD_FIELDS) do
      local key = "mod" .. i .. "_" .. f
      s[key] = d[key]
    end
  end
  -- Non-dialog state
  s.schedule_last_profile = PT.schedule_last_profile

  local ok, encoded = pcall(PT.json.encode, s)
  if not ok then
    PT.update_status("Settings encode error — not saved")
    return
  end
  PT._last_encoded_settings = encoded
  -- Atomic write: .tmp then rename
  local tmp_path = PT.cfg.SETTINGS_FILE .. ".tmp"
  local f, ferr = io.open(tmp_path, "w")
  if f then
    local wok, werr = pcall(function() f:write(encoded); f:close() end)
    if not wok then
      PT.update_status("Settings save error: " .. tostring(werr))
      pcall(os.remove, tmp_path)
      return
    end
    pcall(os.remove, PT.cfg.SETTINGS_FILE)
    local rok, rerr = os.rename(tmp_path, PT.cfg.SETTINGS_FILE)
    if not rok then
      pcall(os.remove, tmp_path)
      local ff = io.open(PT.cfg.SETTINGS_FILE, "w")
      if ff then ff:write(encoded); ff:close()
      else PT.update_status("Settings save failed: " .. tostring(rerr)) end
    end
  else
    PT.update_status("Cannot save settings: " .. tostring(ferr))
  end
end

function PT.load_settings()
  local f = io.open(PT.cfg.SETTINGS_FILE, "r")
  if not f then
    -- Recovery: .tmp may exist if process crashed between remove and rename
    local tmp = PT.cfg.SETTINGS_FILE .. ".tmp"
    f = io.open(tmp, "r")
    if not f then return nil end
    local data = f:read("*a")
    f:close()
    pcall(os.rename, tmp, PT.cfg.SETTINGS_FILE)
    local rok, s = pcall(PT.json.decode, data)
    if rok and type(s) == "table" then return s end
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
  local dlg = PT.dlg
  -- Core fields from schema (type-validated + pcall for comboboxes)
  for _, field in ipairs(_FIELD_SCHEMA) do
    _apply_field(dlg, field[1], field[2], s[field[1]])
  end
  -- Modulation slots
  for i = 1, 6 do
    for _, f in ipairs(_MOD_FIELDS) do
      local key = "mod" .. i .. "_" .. f
      _apply_field(dlg, key, _MOD_TYPES[f], s[key])
    end
  end
  -- Sync all slider labels from registry
  for id in pairs(PT.SLIDER_LABELS) do
    PT.sync_slider_label(id)
  end
  -- Special display cases not in slider registry
  local d = dlg.data
  dlg:modify{ id = "audio_max_frames",
    label = d.audio_max_frames == 0 and "Max Frames (0=all)" or ("Max Frames (" .. d.audio_max_frames .. ")") }
  -- Mode label hint
  if s.mode then
    if s.mode == "inpaint" then
      dlg:modify{ id = "mode", label = "Mode (needs mask)" }
    elseif s.mode == "controlnet_qrcode" then
      dlg:modify{ id = "mode", label = "Mode (QR)" }
    elseif s.mode == "img2img" or (s.mode and s.mode:find("controlnet_")) then
      dlg:modify{ id = "mode", label = "Mode (needs layer)" }
    else
      dlg:modify{ id = "mode", label = "Mode" }
    end
  end
  -- Sync randomness label
  if s.randomness then
    local v = s.randomness
    local names = { [0]="Off", [5]="Subtle", [10]="Moderate", [15]="Wild", [20]="Chaos" }
    local name = names[v] or ""
    local suffix = name ~= "" and (" — " .. name) or ""
    dlg:modify{ id = "randomness", label = "Randomness (" .. v .. suffix .. ")" }
  end
  -- Sync output state
  if s.save_output ~= nil then PT.output.enabled = s.save_output end

  -- Sync conditional widget states (onchange not fired by modify)
  dlg:modify{ id = "pixel_size", enabled = d.pixelate == true }
  local qen = d.quantize_enabled == true
  dlg:modify{ id = "colors", enabled = qen }
  dlg:modify{ id = "quantize_method", enabled = qen }
  dlg:modify{ id = "dither", enabled = qen }
  local pm = d.palette_mode or "auto"
  dlg:modify{ id = "palette_name", enabled = (pm == "preset") }
  dlg:modify{ id = "palette_custom_colors", enabled = (pm == "custom") }
  dlg:modify{ id = "anim_freeinit_iters", enabled = d.anim_freeinit == true }
  dlg:modify{ id = "audio_freeinit_iters", enabled = d.audio_freeinit == true }
  local expr_en = d.audio_use_expressions == true
  dlg:modify{ id = "audio_expr_preset", enabled = expr_en }
  for _, field in ipairs(_FIELD_SCHEMA) do
    if field[1]:sub(1, 5) == "expr_" then
      dlg:modify{ id = field[1], enabled = expr_en }
    end
  end
  local count = d.mod_slot_count or 2
  for i = 1, 6 do
    local en = (i <= count)
    local p = "mod" .. i .. "_"
    for _, sf in ipairs(_MOD_FIELDS) do
      dlg:modify{ id = p .. sf, enabled = en }
    end
  end

  -- Migration v0.7.5→v0.7.7: copy shared sliders to dedicated pipeline sliders
  if s.anim_steps == nil and s.steps then
    dlg:modify{ id = "anim_steps", value = s.steps }
  end
  if s.anim_cfg == nil and s.cfg_scale then
    dlg:modify{ id = "anim_cfg", value = s.cfg_scale }
    dlg:modify{ id = "anim_cfg", label = string.format("CFG (%.1f)", s.cfg_scale / 10.0) }
  end
  if s.audio_steps == nil and s.steps then
    dlg:modify{ id = "audio_steps", value = s.steps }
  end
  if s.audio_cfg == nil and s.cfg_scale then
    dlg:modify{ id = "audio_cfg", value = s.cfg_scale }
    dlg:modify{ id = "audio_cfg", label = string.format("CFG (%.1f)", s.cfg_scale / 10.0) }
  end
  if s.audio_denoise == nil and s.denoise then
    dlg:modify{ id = "audio_denoise", value = s.denoise }
    dlg:modify{ id = "audio_denoise", label = string.format("Strength (%.2f)", s.denoise / 100.0) }
  end

  -- Restore non-dialog state
  PT.schedule_last_profile = s.schedule_last_profile or "dynamic"
  -- Initialize prompt schedule state from restored DSL
  if PT.update_schedule_state then
    local restored_dsl = dlg.data.generate_prompt_schedule_dsl or ""
    if restored_dsl ~= "" then PT.update_schedule_state(restored_dsl) end
  end
end

end
