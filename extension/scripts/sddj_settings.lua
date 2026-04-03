--
-- SDDj — Settings Persistence
--

return function(PT)

local _SETTINGS_VERSION = 2
local _MOD_SLOT_COUNT = 6

-- Field types: "text" (entry), "option" (combobox), "value" (slider), "bool" (check)
local _FIELD_SCHEMA = {
  -- Connection
  { "server_url",          "text" },
  -- Generate tab
  { "prompt",              "text" },
  { "negative_prompt",     "text" },
  { "seed",                "text" },
  { "fixed_subject",       "text" },
  { "fixed_custom",        "text" },
  { "subject_position",    "option" },
  { "custom_position",     "option" },
  { "mode",                "option" },
  { "output_size",         "option" },
  { "output_mode",         "option" },
  { "lora_name",           "option" },
  { "lora2_name",          "option" },
  { "preset_name",         "option" },
  { "scheduler",           "option" },
  { "loop_seed_combo",     "option" },
  { "ip_adapter_mode",     "option" },
  { "steps",               "value" },
  { "cfg_scale",           "value" },
  { "clip_skip",           "value" },
  { "denoise",             "value" },
  { "lora_weight",         "value" },
  { "lora2_weight",        "value" },
  { "neg_ti_weight",       "value" },
  { "randomness",          "value" },
  { "ip_adapter_scale",    "value" },
  { "gen_guidance_start",  "value" },
  { "gen_guidance_end",    "value" },
  { "guidance_rescale",    "value" },
  { "use_neg_ti",          "bool" },
  { "lock_subject",        "bool" },
  { "lock_custom",         "bool" },
  { "randomize_before",    "bool" },
  { "loop_check",          "bool" },
  { "random_loop_check",   "bool" },
  { "save_output",         "bool" },
  { "lora2_enabled",       "bool" },
  { "ip_adapter_enabled",  "bool" },
  { "pag_enabled",         "bool" },
  { "pag_scale",           "value" },
  -- Post-Process tab
  { "pixel_size",          "value" },
  { "colors",              "value" },
  { "quantize_method",     "option" },
  { "dither",              "option" },
  { "palette_mode",        "option" },
  { "palette_name",        "option" },
  { "palette_custom_colors", "text" },
  { "pixelate",            "bool" },
  { "pixelate_method",     "option" },
  { "quantize_enabled",    "bool" },
  { "remove_bg",           "bool" },
  { "upscale_enabled",     "bool" },
  { "upscale_factor",      "option" },
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
  { "anim_guidance_start", "value" },
  { "anim_guidance_end",   "value" },
  { "frame_interpolation", "option" },
  -- Actions panel
  { "ab_compare",          "bool" },
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
  -- UI state
  { "main_tabs",            "option" },
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
  -- Modulation slots (_MOD_SLOT_COUNT × 8 fields)
  for i = 1, _MOD_SLOT_COUNT do
    for _, f in ipairs(_MOD_FIELDS) do
      local key = "mod" .. i .. "_" .. f
      s[key] = d[key]
    end
  end
  -- Non-dialog state
  s.schedule_last_profile = PT.schedule_last_profile
  -- Prompt history (table, not a dialog field)
  if PT.prompt_history and #PT.prompt_history > 0 then
    s.prompt_history = PT.prompt_history
  end

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

  PT._ui_transaction_depth = (PT._ui_transaction_depth or 0) + 1

  -- Core fields from schema (type-validated + pcall for comboboxes)
  for _, field in ipairs(_FIELD_SCHEMA) do
    _apply_field(dlg, field[1], field[2], s[field[1]])
  end
  -- Modulation slots
  for i = 1, _MOD_SLOT_COUNT do
    for _, f in ipairs(_MOD_FIELDS) do
      local key = "mod" .. i .. "_" .. f
      _apply_field(dlg, key, _MOD_TYPES[f], s[key])
    end
  end
  -- Sync all slider labels from registry
  for id in pairs(PT.SLIDER_LABELS) do
    PT.sync_slider_label(id)
  end
  -- Sync output state
  if s.save_output ~= nil then PT.output.enabled = s.save_output end

  PT._ui_transaction_depth = PT._ui_transaction_depth - 1

  -- Centralized sync of all conditional widget states, labels, and mode hints
  PT.sync_ui_conditional_states()

  -- Legacy migration v0.7.x removed in v0.9.84 (all users past v0.7.7)

  -- Restore non-dialog state
  PT.schedule_last_profile = s.schedule_last_profile or "dynamic"
  -- Restore prompt history
  if s.prompt_history and type(s.prompt_history) == "table" then
    PT.prompt_history = s.prompt_history
  end
  -- Initialize prompt schedule state from restored DSL
  if PT.update_schedule_state then
    local restored_dsl = dlg.data.generate_prompt_schedule_dsl or ""
    if restored_dsl ~= "" then PT.update_schedule_state(restored_dsl) end
  end

  -- Cache encoded JSON for exit() fallback (covers crash-before-any-save edge case)
  local ok, encoded = pcall(PT.json.encode, s)
  if ok then PT._last_encoded_settings = encoded end
end

end
