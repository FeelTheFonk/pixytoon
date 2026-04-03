--
-- SDDj — Dialog Construction
--

return function(PT)

-- ─── Slider Label Helpers ────────────────────────────────────

-- Returns an onchange callback that syncs a slider's label from the
-- PT.SLIDER_LABELS registry (single source of truth).
local function onchange_sync(id)
  return function() PT.sync_slider_label(id) end
end

-- ─── Loop State Helpers ──────────────────────────────────────

-- Initialize loop state for the given target ("generate"|"animate"|"audio").
-- Returns true if loop mode is active; false if no looping requested.
local function init_loop_state(target)
  local d = PT.dlg.data
  local is_loop = d.loop_check or d.random_loop_check
  if not is_loop then return false end
  if PT.loop.mode then return true end  -- already initialized (re-entry)
  PT.loop.mode = true
  PT.loop.counter = 0
  PT.loop.seed_mode = d.loop_seed_combo or "random"
  PT.loop.random_mode = d.random_loop_check or false
  PT.loop.locked_fields = PT.build_locked_fields()
  PT.loop.target = target
  if PT.loop.seed_mode == "random" then
    PT.dlg:modify{ id = "seed", text = "-1" }
  end
  return true
end

-- ─── Constants ───────────────────────────────────────────────

local GLOBAL_SOURCES = {
  -- Energy & dynamics
  "global_rms", "global_onset", "global_centroid", "global_beat",
  -- 9-band frequency segmentation
  "global_sub_bass", "global_bass", "global_low_mid",
  "global_mid", "global_upper_mid", "global_presence",
  "global_brilliance", "global_air", "global_ultrasonic",
  -- Backward-compat aliases
  "global_low", "global_high",
  -- Spectral timbral features
  "global_spectral_contrast", "global_spectral_flatness",
  "global_spectral_bandwidth", "global_spectral_rolloff",
  "global_spectral_flux",
  -- CQT chromagram (12 pitch classes + aggregate)
  "global_chroma_energy",
  "global_chroma_C", "global_chroma_Cs", "global_chroma_D",
  "global_chroma_Ds", "global_chroma_E", "global_chroma_F",
  "global_chroma_Fs", "global_chroma_G", "global_chroma_Gs",
  "global_chroma_A", "global_chroma_As", "global_chroma_B",
}

local MOD_TARGETS = {
  "denoise_strength", "cfg_scale", "noise_amplitude",
  "controlnet_scale", "seed_offset", "palette_shift",
  "frame_cadence",
  -- Motion / camera (smooth Deforum-like)
  "motion_x", "motion_y", "motion_zoom", "motion_rotation",
  "motion_tilt_x", "motion_tilt_y",
}

-- Format strings for mod slot label display (ranges from PT.PARAM_DEFS)
local MOD_TARGET_FMT = {
  denoise_strength = "%.2f",
  cfg_scale        = "%.1f",
  noise_amplitude  = "%.2f",
  controlnet_scale = "%.2f",
  seed_offset      = "%.0f",
  palette_shift    = "%.2f",
  frame_cadence    = "%.1f",
  motion_x         = "%.1f",
  motion_y         = "%.1f",
  motion_zoom      = "%.3f",
  motion_rotation  = "%.1f",
  motion_tilt_x    = "%.1f",
  motion_tilt_y    = "%.1f",
}

local EXPR_FIELDS = {
  { "expr_denoise",       "denoise" },
  { "expr_cfg",           "cfg_scale" },
  { "expr_noise",         "noise_amp" },
  { "expr_controlnet",    "cn_scale" },
  { "expr_seed",          "seed_off" },
  { "expr_palette",       "pal_shift" },
  { "expr_cadence",       "cadence" },
  { "expr_motion_x",      "motion_x" },
  { "expr_motion_y",      "motion_y" },
  { "expr_motion_zoom",   "zoom" },
  { "expr_motion_rot",    "rotation" },
  { "expr_motion_tilt_x", "tilt_x" },
  { "expr_motion_tilt_y", "tilt_y" },
}

-- Modulation slot defaults: [source, target, min, max, attack, release]
local SLOT_DEFAULTS = {
  { "global_rms",   "denoise_strength",  30, 65,  2,  8 },
  { "global_onset", "cfg_scale",         30, 80,  2,  8 },
  { "global_low",   "noise_amplitude",    0, 30,  2,  8 },
  { "global_high",  "seed_offset",        0, 50,  2,  8 },
  { "global_mid",   "motion_x",          30, 70,  3, 12 },
  { "global_rms",   "motion_zoom",       40, 60,  4, 15 },
}

-- ─── Dispatch Tables ─────────────────────────────────────────

local TAB_PENDING = {
  tab_gen   = "generate",
  tab_pp    = "generate",
  tab_anim  = "animate",
  tab_qr    = "qr_generate",
  tab_audio = "audio",
}

-- ─── UI Sync ────────────────────────────────────────────────

function PT.sync_ui_conditional_states()
  if not PT.dlg then return end
  if PT._ui_transaction_depth and PT._ui_transaction_depth > 0 then return end
  PT._ui_transaction_depth = (PT._ui_transaction_depth or 0) + 1
  local ok, err = pcall(function()

  local dlg = PT.dlg
  local d = dlg.data

  -- Mode label hint with live sprite/layer validation
  local m = d.mode
  if m then
    local spr = app.sprite
    local has_sprite = (spr ~= nil)
    local has_layer = has_sprite and (app.cel ~= nil and app.cel.image ~= nil)
    if m == "inpaint" then
      if not has_sprite then
        pcall(dlg.modify, dlg, { id = "mode", label = "Mode [!no sprite]" })
      else
        pcall(dlg.modify, dlg, { id = "mode", label = "Mode (needs mask)" })
      end
    elseif m == "controlnet_qrcode" then
      pcall(dlg.modify, dlg, { id = "mode", label = "Mode (QR)" })
    elseif m == "img2img" or m:find("controlnet_") then
      if not has_layer then
        pcall(dlg.modify, dlg, { id = "mode", label = "Mode [!no layer]" })
      else
        pcall(dlg.modify, dlg, { id = "mode", label = "Mode (source ready)" })
      end
    else
      pcall(dlg.modify, dlg, { id = "mode", label = "Mode" })
    end
  end

  -- Randomness label
  local v = d.randomness
  if v then
    local names = { [0]="Off", [5]="Subtle", [10]="Moderate", [15]="Wild", [20]="Chaos" }
    local name = names[v] or ""
    local suffix = name ~= "" and (" — " .. name) or ""
    pcall(dlg.modify, dlg, { id = "randomness", label = "Randomness (" .. v .. suffix .. ")" })
  end

  -- Post Process (visibility: works around Aseprite slider disabled rendering bug)
  pcall(dlg.modify, dlg, { id = "pixel_size", visible = (d.pixelate == true) })
  pcall(dlg.modify, dlg, { id = "pixelate_method", visible = (d.pixelate == true) })

  local qen = (d.quantize_enabled == true)
  pcall(dlg.modify, dlg, { id = "colors", visible = qen })
  pcall(dlg.modify, dlg, { id = "quantize_method", visible = qen })
  pcall(dlg.modify, dlg, { id = "dither", visible = qen })

  local pm = d.palette_mode or "auto"
  pcall(dlg.modify, dlg, { id = "palette_name", visible = (pm == "preset") })
  pcall(dlg.modify, dlg, { id = "palette_custom_colors", visible = (pm == "custom") })

  -- Animation / Audio (visibility fix)
  pcall(dlg.modify, dlg, { id = "anim_freeinit_iters", visible = (d.anim_freeinit == true) })
  pcall(dlg.modify, dlg, { id = "audio_freeinit_iters", visible = (d.audio_freeinit == true) })

  -- Audio advanced sections visibility
  if d.audio_advanced ~= nil then
    local show = (d.audio_advanced == true)
    pcall(dlg.modify, dlg, { id = "audio_use_expressions", visible = show })
    pcall(dlg.modify, dlg, { id = "audio_random_seed", visible = show })
    pcall(dlg.modify, dlg, { id = "audio_choreography", visible = show })

    local show_expr = show and (d.audio_use_expressions == true)
    pcall(dlg.modify, dlg, { id = "audio_expr_preset", visible = show_expr })
    -- Show expression variables hint only when expressions are visible
    pcall(dlg.modify, dlg, { id = "expr_vars_hint", visible = show_expr })
    for _, field in ipairs(EXPR_FIELDS) do
      pcall(dlg.modify, dlg, { id = field[1], visible = show_expr })
    end
  end

  -- ControlNet mode detection (shared)
  local is_cn_mode = d.mode and (d.mode:find("controlnet_") ~= nil)
  local is_non_qr_cn = is_cn_mode and d.mode ~= "controlnet_qrcode"

  -- Generate tab: ControlNet guidance + guidance_rescale (visible in CN modes, not QR)
  pcall(dlg.modify, dlg, { id = "gen_guidance_start", visible = is_non_qr_cn })
  pcall(dlg.modify, dlg, { id = "gen_guidance_end", visible = is_non_qr_cn })
  pcall(dlg.modify, dlg, { id = "guidance_rescale", visible = is_non_qr_cn })

  -- IP-Adapter: mode + scale + hint only visible when checkbox is enabled
  local ip_en = (d.ip_adapter_enabled == true)
  pcall(dlg.modify, dlg, { id = "ip_adapter_mode", visible = ip_en })
  pcall(dlg.modify, dlg, { id = "ip_adapter_scale", visible = ip_en })
  pcall(dlg.modify, dlg, { id = "ip_adapter_hint", visible = ip_en })

  -- Animation guidance sliders visible only in ControlNet modes
  pcall(dlg.modify, dlg, { id = "anim_guidance_start", visible = is_cn_mode })
  pcall(dlg.modify, dlg, { id = "anim_guidance_end", visible = is_cn_mode })

  -- Modulation slots (full visibility toggle including slider widgets)
  local count = d.mod_slot_count or 2
  for i = 1, 6 do
    local vis = (i <= count)
    local p = "mod" .. i .. "_"
    pcall(function()
      for _, s in ipairs({ "enable", "invert", "source", "target", "min", "max", "attack", "release" }) do
        dlg:modify{ id = p .. s, visible = vis }
      end
    end)
  end

  -- Audio max frames label
  if d.audio_max_frames ~= nil then
    pcall(dlg.modify, dlg, { id = "audio_max_frames",
      label = d.audio_max_frames == 0 and "Max Frames (0=all)" or ("Max Frames (" .. d.audio_max_frames .. ")") })
  end

  -- Modulation slot min/max real-value labels
  if PT.sync_all_mod_labels then PT.sync_all_mod_labels() end

  end)
  PT._ui_transaction_depth = PT._ui_transaction_depth - 1
  if not ok then error(err) end
end

-- ─── Connection Section ─────────────────────────────────────

local function build_connection_section()
  local dlg = PT.dlg

  dlg:separator{ text = "Connection", hexpand = true }

  dlg:entry{
    id = "server_url",
    label = "Server",
    text = PT.cfg.DEFAULT_SERVER_URL,
    hexpand = true,
  }

  dlg:label{ id = "status", text = "Disconnected", hexpand = true }

  dlg:button{
    id = "connect_btn",
    text = "Connect",
    hexpand = true,
    onclick = function()
      if PT.state.connected then
        PT.disconnect()
      else
        PT.cfg.DEFAULT_SERVER_URL = dlg.data.server_url or PT.cfg.DEFAULT_SERVER_URL
        PT.connect()
      end
    end,
  }
  dlg:newrow()
  dlg:button{
    id = "refresh_btn",
    text = "Refresh Resources",
    onclick = function()
      if PT.state.connected then
        PT.res.requested = false
        PT.request_resources()
        PT.update_status("Refreshing...")
      else
        PT.update_status("Not connected")
      end
    end,
  }
  dlg:button{
    id = "cleanup_btn",
    text = "Cleanup GPU",
    onclick = function()
      if PT.state.connected and not PT.state.generating
          and not PT.state.animating then
        PT.send({ action = "cleanup" })
        PT.update_status("Cleaning up GPU...")
      else
        PT.update_status("Cannot cleanup during generation")
      end
    end,
  }
  dlg:button{
    id = "open_output_btn",
    text = "Open Output",
    onclick = function()
      PT.open_output_dir()
    end,
  }

  dlg:check{
    id = "save_output",
    text = "Save to output",
    selected = true,
    onchange = function()
      PT.output.enabled = dlg.data.save_output
    end,
  }
end

-- Update prompt preview label with locked fields injected (shared helper)
local function _update_prompt_preview()
  if not PT.dlg then return end
  local preview = PT.inject_locked_prompt(PT.dlg.data.prompt or "")
  if #preview > 80 then preview = preview:sub(1, 77) .. "..." end
  pcall(PT.dlg.modify, PT.dlg, { id = "prompt_preview", text = preview ~= "" and preview or "(empty)" })
end

-- ─── Tab: Generate ──────────────────────────────────────────

local function build_tab_generate()
  local dlg = PT.dlg

  -- Preset selector
  dlg:combobox{
    id = "preset_name",
    label = "Preset",
    options = { "(none)" },
    option = "(none)",
    onchange = function()
      local sel = dlg.data.preset_name
      if sel and sel ~= "(none)" then
        PT.send({ action = "get_preset", preset_name = sel })
      end
    end,
  }
  dlg:button{
    id = "preset_save_btn",
    text = "Save",
    onclick = function()
      -- Build summary of what will be saved
      local prompt_len = #(dlg.data.prompt or "")
      local summary = string.format(
        "Steps: %d | CFG: %.1f | Mode: %s | Scheduler: %s | Prompt: %d chars",
        dlg.data.steps or 8,
        (dlg.data.cfg_scale or 50) / 10.0,
        dlg.data.mode or "txt2img",
        dlg.data.scheduler or "DPM++ SDE Karras",
        prompt_len
      )
      local name_dlg = Dialog{ title = "Save Preset" }
      name_dlg:entry{ id = "pname", label = "Name", text = "", hexpand = true }
      name_dlg:label{ text = summary }
      name_dlg:button{ id = "ok", text = "Save" }
      name_dlg:button{ id = "cancel", text = "Cancel" }
      name_dlg:show()
      if not name_dlg.data.ok then return end
      local pname = name_dlg.data.pname or ""
      if pname == "" then return end
      local gw, gh = PT.parse_size()
      local preset_data = {
        prompt_prefix = dlg.data.prompt,
        negative_prompt = dlg.data.negative_prompt,
        mode = dlg.data.mode,
        width = gw, height = gh,
        steps = dlg.data.steps,
        cfg_scale = dlg.data.cfg_scale / 10.0,
        clip_skip = dlg.data.clip_skip,
        scheduler = dlg.data.scheduler,
        denoise_strength = dlg.data.denoise / 100.0,
        post_process = PT.build_post_process(),
        randomness = dlg.data.randomness,
        lock_subject = dlg.data.lock_subject,
        fixed_subject = dlg.data.fixed_subject,
        subject_position = dlg.data.subject_position,
        lock_custom = dlg.data.lock_custom,
        fixed_custom = dlg.data.fixed_custom,
        custom_position = dlg.data.custom_position,
        randomize_before = dlg.data.randomize_before,
      }
      PT.send({ action = "save_preset", preset_name = pname, preset_data = preset_data })
    end,
  }
  dlg:button{
    id = "preset_delete_btn",
    text = "Del",
    onclick = function()
      local sel = dlg.data.preset_name
      if sel and sel ~= "(none)" then
        if app.alert{ title = "Delete Preset", text = "Delete preset '" .. sel .. "'?", buttons = { "Delete", "Cancel" } } == 1 then
          PT.send({ action = "delete_preset", preset_name = sel })
        end
      end
    end,
  }
  dlg:button{
    id = "load_meta_btn",
    text = "Load JSON",
    onclick = function()
      local ok, path = pcall(app.fs.fileDialog, {
        title = "Load Generation Metadata",
        open = true,
        filetypes = { "json" },
      })
      if not ok or not path or path == "" then return end
      local meta, err = PT.load_metadata_file(path)
      if not meta then
        app.alert("Failed to load metadata: " .. tostring(err))
        return
      end
      PT.apply_metadata(meta)
    end,
  }


  dlg:combobox{
    id = "lora_name",
    label = "LoRA",
    options = { "(default)" },
    option = "(default)",
  }

  dlg:slider{
    id = "lora_weight",
    label = "LoRA (1.00)",
    min = -200, max = 200, value = 100,
    onchange = onchange_sync("lora_weight"),
  }

  dlg:check{
    id = "lora2_enabled",
    text = "LoRA 2",
    selected = false,
  }
  dlg:combobox{
    id = "lora2_name",
    label = "",
    options = { "(default)" },
    option = "(default)",
  }
  dlg:slider{
    id = "lora2_weight",
    label = "Weight (1.00)",
    min = -200, max = 200, value = 100,
    onchange = onchange_sync("lora2_weight"),
  }

  -- ── IP-Adapter Reference Image ──────────────────────
  dlg:separator{ text = "Reference Image" }
  dlg:check{
    id = "ip_adapter_enabled",
    text = "Use Reference Image",
    selected = false,
  }
  dlg:combobox{
    id = "ip_adapter_mode",
    label = "Mode",
    options = { "full", "style", "composition" },
    option = "full",
  }
  dlg:slider{
    id = "ip_adapter_scale",
    label = "Scale (0.60)",
    min = 0, max = 200, value = 60,
    onchange = onchange_sync("ip_adapter_scale"),
  }
  dlg:label{
    id = "ip_adapter_hint",
    text = "First use loads IP-Adapter (~2GB VRAM)",
    visible = false,
  }

  dlg:entry{
    id = "prompt",
    label = "Prompt",
    text = "",
    hexpand = true,
    onchange = _update_prompt_preview,
  }
  dlg:button{
    id = "randomize_btn",
    text = "Randomize",
    onclick = function()
      local locked = PT.build_locked_fields()
      PT.send({ action = "generate_prompt", locked_fields = locked, randomness = PT.dlg.data.randomness })
      PT.update_status("Generating prompt...")
    end,
  }
  -- Prompt History button
  dlg:button{
    id = "prompt_history_btn",
    text = "History",
    onclick = function()
      if not PT.prompt_history or #PT.prompt_history == 0 then
        app.alert("No prompt history yet.")
        return
      end
      local hist_dlg = Dialog{ title = "Prompt History" }
      hist_dlg:combobox{
        id = "hist_select",
        label = "Recent",
        options = PT.prompt_history,
        option = PT.prompt_history[1],
      }
      hist_dlg:button{ id = "ok", text = "Use" }
      hist_dlg:button{ id = "cancel", text = "Cancel" }
      hist_dlg:show()
      if hist_dlg.data.ok then
        local sel = hist_dlg.data.hist_select
        if sel and sel ~= "" then
          dlg:modify{ id = "prompt", text = sel }
          _update_prompt_preview()
        end
      end
    end,
  }
  dlg:check{
    id = "lock_subject",
    label = "Lock Subject",
    selected = false,
    onchange = _update_prompt_preview,
  }
  dlg:entry{
    id = "fixed_subject",
    label = "Subject",
    text = "",
    hexpand = true,
    onchange = _update_prompt_preview,
  }
  dlg:combobox{
    id = "subject_position",
    label = "Position",
    options = { "prefix", "suffix", "off" },
    option = "prefix",
    onchange = _update_prompt_preview,
  }

  dlg:check{
    id = "lock_custom",
    label = "Lock Custom",
    selected = false,
    onchange = _update_prompt_preview,
  }
  dlg:entry{
    id = "fixed_custom",
    label = "Custom",
    text = "",
    hexpand = true,
    onchange = _update_prompt_preview,
  }
  dlg:combobox{
    id = "custom_position",
    label = "Position",
    options = { "prefix", "suffix", "off" },
    option = "suffix",
    onchange = _update_prompt_preview,
  }

  -- Prompt preview label (shows final prompt with locks injected)
  dlg:label{
    id = "prompt_preview",
    label = "Preview",
    text = "(empty)",
  }

  dlg:entry{
    id = "negative_prompt",
    label = "Neg. Prompt",
    text = PT.cfg.DEFAULT_NEGATIVE_PROMPT,
    hexpand = true,
  }

  dlg:check{
    id = "use_neg_ti",
    label = "Neg. Embeddings",
    selected = false,
  }

  dlg:slider{
    id = "neg_ti_weight",
    label = "Emb. (1.00)",
    min = 10, max = 200, value = 100,
    onchange = onchange_sync("neg_ti_weight"),
  }
  -- All negative TI embeddings share this weight
  dlg:label{ text = "Shared weight for all embeddings" }

  dlg:combobox{
    id = "output_size",
    label = "Size",
    options = {
      "64x64", "96x96", "128x128", "256x256",
      "384x384", "512x512", "512x768", "768x512",
      "768x768", "1024x1024",
    },
    option = "512x512",
  }

  dlg:combobox{
    id = "output_mode",
    label = "Output",
    options = { "layer", "sequence" },
    option = "layer",
  }

  dlg:entry{
    id = "seed",
    label = "Seed (-1=rand)",
    text = "-1",
    hexpand = true,
  }

  dlg:slider{
    id = "denoise",
    label = "Strength (1.00)",
    min = 0, max = 100, value = 100,
    onchange = onchange_sync("denoise"),
  }

  dlg:slider{
    id = "steps",
    label = "Steps (8)",
    min = 1, max = 100, value = 8,
    onchange = onchange_sync("steps"),
  }

  dlg:slider{
    id = "cfg_scale",
    label = "CFG (5.0)",
    min = 0, max = 300, value = 50,
    onchange = onchange_sync("cfg_scale"),
  }

  dlg:combobox{
    id = "scheduler",
    label = "Scheduler",
    options = { "DPM++ SDE Karras", "DPM++ 2M Karras", "DDIM", "Euler Ancestral", "Euler", "UniPC", "LMS" },
    option = "DPM++ SDE Karras",
  }

  dlg:slider{
    id = "clip_skip",
    label = "CLIP Skip (2)",
    min = 1, max = 12, value = 2,
    onchange = onchange_sync("clip_skip"),
  }

  -- ── Advanced ControlNet Guidance (visible only in controlnet_* modes) ──
  dlg:slider{
    id = "gen_guidance_start",
    label = "CN Start (0%)",
    min = 0, max = 100, value = 0,
    visible = false,
    onchange = onchange_sync("gen_guidance_start"),
  }
  dlg:slider{
    id = "gen_guidance_end",
    label = "CN End (100%)",
    min = 0, max = 100, value = 100,
    visible = false,
    onchange = onchange_sync("gen_guidance_end"),
  }
  dlg:slider{
    id = "guidance_rescale",
    label = "Rescale (0.00)",
    min = 0, max = 100, value = 0,
    visible = false,
    onchange = onchange_sync("guidance_rescale"),
  }
end

-- ─── Tab: Post-Process ──────────────────────────────────────

local function build_tab_postprocess()
  local dlg = PT.dlg

  dlg:separator{ text = "Upscale" }
  dlg:check{
    id = "upscale_enabled",
    text = "Upscale before pixelate",
    selected = false,
  }
  dlg:combobox{
    id = "upscale_factor",
    label = "Factor",
    options = { "2x", "4x" },
    option = "4x",
  }

  dlg:check{
    id = "pixelate",
    label = "Pixelate",
    selected = false,
    onchange = function()
      PT.sync_ui_conditional_states()
    end,
  }

  dlg:slider{
    id = "pixel_size",
    label = "Target (128px)",
    min = 8, max = 512, value = 128,
    onchange = onchange_sync("pixel_size"),
  }

  dlg:combobox{
    id = "pixelate_method",
    label = "Pixel Mode",
    options = { "nearest", "box", "pixeloe" },
    option = "nearest",
  }

  dlg:check{
    id = "quantize_enabled",
    label = "Quantize Colors",
    selected = false,
    onchange = function()
      PT.sync_ui_conditional_states()
    end,
  }

  dlg:slider{
    id = "colors",
    label = "Colors (32)",
    min = 2, max = 256, value = 32,
    onchange = onchange_sync("colors"),
  }

  dlg:combobox{
    id = "quantize_method",
    label = "Quantize",
    options = { "kmeans", "median_cut", "octree", "octree_lab" },
    option = "kmeans",
  }

  dlg:combobox{
    id = "dither",
    label = "Dithering",
    options = { "none", "floyd_steinberg", "bayer_2x2", "bayer_4x4", "bayer_8x8" },
    option = "none",
  }

  dlg:combobox{
    id = "palette_mode",
    label = "Palette",
    options = { "auto", "preset", "custom" },
    option = "auto",
    onchange = function()
      PT.sync_ui_conditional_states()
    end,
  }

  dlg:combobox{
    id = "palette_name",
    label = "Preset",
    options = { "pico8" },
    option = "pico8",
  }

  dlg:entry{
    id = "palette_custom_colors",
    label = "Custom Hex",
    text = "",
    hexpand = true,
  }

  dlg:button{
    id = "palette_save_btn",
    text = "Save Palette",
    onclick = function()
      local hex_str = dlg.data.palette_custom_colors or ""
      local colors = {}
      for hex in hex_str:gmatch("#?(%x%x%x%x%x%x)") do
        colors[#colors + 1] = "#" .. hex
      end
      if #colors == 0 then
        app.alert("Enter hex colors first (e.g. #FF0000 #00FF00 #0000FF)")
        return
      end
      local name_dlg = Dialog{ title = "Save Palette" }
      name_dlg:entry{ id = "pname", label = "Name", text = "", hexpand = true }
      name_dlg:button{ id = "ok", text = "Save" }
      name_dlg:button{ id = "cancel", text = "Cancel" }
      name_dlg:show()
      if not name_dlg.data.ok then return end
      local pname = name_dlg.data.pname or ""
      if pname ~= "" then
        PT.send({ action = "save_palette", palette_save_name = pname, palette_save_colors = colors })
      end
    end,
  }

  dlg:button{
    id = "palette_del_btn",
    text = "Del Palette",
    onclick = function()
      local name = dlg.data.palette_name
      if not name or name == "" then return end
      if app.alert{ title = "Delete Palette", text = "Delete palette '" .. name .. "'?", buttons = { "Delete", "Cancel" } } == 1 then
        PT.send({ action = "delete_palette", palette_save_name = name })
      end
    end,
  }

  dlg:check{
    id = "remove_bg",
    label = "Remove BG",
    selected = false,
  }

  -- Initial conditional states will be set by PT.sync_ui_conditional_states() at dialogue build complete
end

-- ─── Tab: Animation ─────────────────────────────────────────

local function build_tab_animation()
  local dlg = PT.dlg

  dlg:combobox{
    id = "anim_method",
    label = "Method",
    options = { "chain", "animatediff" },
    option = "chain",
  }

  dlg:slider{
    id = "anim_steps",
    label = "Steps (8)",
    min = 1, max = 50, value = 8,
    onchange = onchange_sync("anim_steps"),
  }
  dlg:slider{
    id = "anim_cfg",
    label = "CFG (5.0)",
    min = 0, max = 200, value = 50,
    onchange = onchange_sync("anim_cfg"),
  }

  dlg:slider{
    id = "anim_frames",
    label = "Frames (8)",
    min = 2, max = 256, value = 8,
    onchange = onchange_sync("anim_frames"),
  }

  dlg:slider{
    id = "anim_duration",
    label = "Duration (100ms)",
    min = 50, max = 2000, value = 100,
    onchange = onchange_sync("anim_duration"),
  }

  dlg:slider{
    id = "anim_denoise",
    label = "Strength (0.30)",
    min = 1, max = 100, value = 30,
    onchange = onchange_sync("anim_denoise"),
  }

  dlg:combobox{
    id = "anim_seed_strategy",
    label = "Seed Mode",
    options = { "increment", "fixed", "random" },
    option = "increment",
  }

  dlg:entry{
    id = "anim_tag",
    label = "Tag Name",
    text = "",
    hexpand = true,
  }

  dlg:check{
    id = "anim_freeinit",
    label = "FreeInit",
    selected = false,
    onchange = function()
      PT.sync_ui_conditional_states()
    end,
  }

  dlg:slider{
    id = "anim_freeinit_iters",
    label = "FreeInit Iters",
    min = 1, max = 3, value = 2,
  }

  dlg:combobox{
    id = "frame_interpolation",
    label = "Frame Interpolation",
    options = { "None", "2x", "3x", "4x" },
    option = "None",
  }

  -- Guidance Start/End for Animation ControlNet modes
  dlg:slider{
    id = "anim_guidance_start",
    label = "Guide Start (0.00)",
    min = 0, max = 100, value = 0,
    onchange = onchange_sync("anim_guidance_start"),
  }
  dlg:slider{
    id = "anim_guidance_end",
    label = "Guide End (0.80)",
    min = 0, max = 100, value = 80,
    onchange = onchange_sync("anim_guidance_end"),
  }

  -- ─── Prompt Schedule (frame-based prompt evolution) ────────

  dlg:separator{ text = "Prompt Schedule" }

  -- DSL text entry (data store, edit via popup)
  dlg:entry{
    id = "generate_prompt_schedule_dsl",
    text = "",
    visible = false,
    onchange = function()
      PT.update_schedule_state(dlg.data.generate_prompt_schedule_dsl)
    end,
  }

  -- Visual timeline canvas
  dlg:canvas{
    id = "schedule_timeline",
    width = 300,
    height = 30,
    hexpand = true,
    vexpand = false,
    onpaint = function(ev)
      -- Cache canvas width for click handler (ev.context only in onpaint)
      PT.schedule_timeline_width = ev.context.width
      PT.schedule_timeline_height = ev.context.height
      if PT.paint_schedule_timeline then
        PT.paint_schedule_timeline(ev)
      end
    end,
    onmousedown = function(ev)
      if PT.on_timeline_click then
        PT.on_timeline_click(ev)
      end
    end,
  }

  -- Parse status label
  dlg:label{
    id = "schedule_status",
    label = "Schedule",
    text = "None",
  }

  -- Action buttons row
  dlg:button{
    id = "schedule_edit_btn",
    text = "Edit...",
    onclick = function()
      if PT.open_schedule_editor then
        PT.open_schedule_editor()
      end
    end,
  }
  -- Keyframe Preview — generate a single image from first keyframe's prompt
  dlg:button{
    id = "schedule_preview_btn",
    text = "Preview",
    onclick = function()
      if PT.state.generating or PT.state.animating then return end
      local dsl = dlg.data.generate_prompt_schedule_dsl or ""
      if dsl == "" or dsl:match("^%s*$") then
        app.alert("No prompt schedule to preview.")
        return
      end
      -- Parse first keyframe prompt (simple: find first [N]\n<prompt>)
      local first_prompt = nil
      local in_kf = false
      for line in dsl:gmatch("[^\n]+") do
        local stripped = line:match("^%s*(.-)%s*$")
        if stripped:match("^%[%d+%]") then
          in_kf = true
        elseif in_kf and stripped ~= "" and not stripped:match("^%-%-") and not stripped:match("^transition:") and not stripped:match("^blend:") and not stripped:match("^weight:") and not stripped:match("^denoise:") and not stripped:match("^cfg:") and not stripped:match("^steps:") then
          first_prompt = stripped
          break
        end
      end
      if not first_prompt or first_prompt == "" then
        first_prompt = dlg.data.prompt or ""
      end
      if first_prompt == "" then
        app.alert("No prompt found in first keyframe.")
        return
      end
      -- Trigger a normal generate with the keyframe prompt
      local saved_prompt = dlg.data.prompt
      dlg:modify{ id = "prompt", text = first_prompt }
      PT.trigger_generate()
      -- Restore original prompt after request is built
      dlg:modify{ id = "prompt", text = saved_prompt }
    end,
  }
  dlg:button{
    id = "schedule_presets_btn",
    text = "Presets",
    onclick = function()
      if PT.open_schedule_presets then
        PT.open_schedule_presets()
      end
    end,
  }
  dlg:button{
    id = "schedule_random_btn",
    text = "Random",
    onclick = function()
      if not PT.state.connected then
        app.alert("Connect to the server first.")
        return
      end
      -- Confirm before replacing an existing schedule
      local dsl = dlg.data.generate_prompt_schedule_dsl or ""
      if dsl ~= "" and not dsl:match("^%s*$") then
        if app.alert{
          title = "Replace Schedule?",
          text = "The current prompt schedule will be replaced.",
          buttons = { "Replace", "Cancel" }
        } ~= 1 then
          return
        end
      end
      if PT.open_schedule_randomizer then
        PT.open_schedule_randomizer()
      end
    end,
  }
  dlg:button{
    id = "schedule_clear_btn",
    text = "Clear",
    onclick = function()
      dlg:modify{ id = "generate_prompt_schedule_dsl", text = "" }
      PT.update_schedule_state("")
    end,
  }

  -- Conditional states handled by PT.sync_ui_conditional_states() at dialog build complete
end

-- ─── Tab: Audio ───────────────────────────────────────────

local function build_tab_audio()
  local dlg = PT.dlg

  -- File
  dlg:file{
    id = "audio_file",
    label = "File",
    filetypes = { "wav", "mp3", "flac", "ogg", "m4a", "aac" },
    open = true,
  }
  dlg:button{
    id = "audio_analyze_btn",
    text = "Analyze",
    onclick = function()
      if not PT.state.connected then
        app.alert("Connect to the server first.")
        return
      end
      local path = dlg.data.audio_file
      if not path or path == "" then
        app.alert("Select an audio file first.")
        return
      end
      PT.audio.analyzing = true
      PT.audio.analyzed = false
      dlg:modify{ id = "audio_analyze_btn", enabled = false }
      dlg:modify{ id = "audio_status", text = "Analyzing..." }
      PT.update_status("Analyzing audio...")
      PT.send(PT.build_analyze_audio_request())
    end,
  }
  dlg:label{ id = "audio_status", text = "No file | Frames: --" }

  dlg:check{
    id = "audio_stems_enable",
    text = "Enable Stems (CPU)",
    selected = false,
    onchange = function()
      if dlg.data.audio_stems_enable and PT.state.connected then
        PT.send({ action = "check_stems" })
      end
    end,
  }

  dlg:entry{ id = "audio_tag", label = "Tag Name", text = "", hexpand = true }

  dlg:combobox{
    id = "audio_fps",
    label = "FPS",
    options = { "4", "8", "12", "15", "23.976", "24", "25", "29.97", "30", "50", "59.94", "60" },
    option = "24",
  }
  dlg:slider{
    id = "audio_steps",
    label = "Steps (8)",
    min = 1, max = 50, value = 8,
    onchange = onchange_sync("audio_steps"),
  }
  dlg:slider{
    id = "audio_cfg",
    label = "CFG (5.0)",
    min = 0, max = 200, value = 50,
    onchange = onchange_sync("audio_cfg"),
  }
  dlg:slider{
    id = "audio_denoise",
    label = "Strength (0.50)",
    min = 20, max = 100, value = 50,
    onchange = onchange_sync("audio_denoise"),
  }

  dlg:slider{
    id = "audio_max_frames",
    label = "Max Frames (0=all)",
    min = 0, max = 10800, value = 0,
    onchange = function()
      PT.sync_ui_conditional_states()
    end,
  }
  dlg:combobox{
    id = "audio_method",
    label = "Method",
    options = { "chain", "animatediff" },
    option = "chain",
  }
  dlg:check{
    id = "audio_freeinit",
    text = "FreeInit (1st chunk)",
    selected = false,
    onchange = function()
      PT.sync_ui_conditional_states()
    end,
  }
  dlg:slider{
    id = "audio_freeinit_iters",
    label = "FreeInit Iters",
    min = 1, max = 3, value = 2,
  }

  -- Choreography
  dlg:separator{ text = "Choreography" }
  dlg:combobox{
    id = "audio_choreography",
    label = "Camera Journey",
    options = {
      "(none)",
      "orbit_journey", "dolly_zoom_vertigo", "crane_ascending",
      "wandering_voyage", "hypnotic_spiral",
      "breathing_calm", "staccato_cuts",
    },
    option = "(none)",
    onchange = function()
      local sel = dlg.data.audio_choreography
      if sel == "(none)" then return end
      if PT.state.connected then
        PT.send({ action = "get_choreography_preset", preset_name = sel })
      end
      PT.update_status("Choreography '" .. sel .. "' selected")
    end,
  }

  -- Modulation
  dlg:separator{ text = "Modulation" }
  dlg:combobox{
    id = "audio_mod_preset",
    label = "Preset",
    options = {
      "(custom)",
      "electronic_pulse", "rock_energy", "hiphop_bounce",
      "classical_flow", "ambient_drift",
      "glitch_chaos", "smooth_morph", "rhythmic_pulse",
      "atmospheric", "abstract_noise",
      "one_click_easy", "beginner_balanced",
      "intermediate_full", "advanced_max",
      "controlnet_reactive", "seed_scatter", "noise_sculpt",
      "gentle_drift", "pulse_zoom", "slow_rotate", "cinematic_sweep",
      "cinematic_tilt", "zoom_breathe", "parallax_drift", "full_cinematic",
      "voyage_serene", "voyage_exploratory", "voyage_dramatic", "voyage_psychedelic",
      "intelligent_drift", "reactive_pause",
      "spectral_sculptor", "tonal_drift", "ultra_precision", "micro_reactive",
      "energetic", "ambient", "bass_driven",
    },
    option = "(custom)",
    onchange = function()
      local sel = dlg.data.audio_mod_preset
      if sel == "(custom)" then return end
      if PT.state.connected then
        PT.send({ action = "get_modulation_preset", preset_name = sel })
      end
      PT.update_status("Preset '" .. sel .. "' selected")
    end,
  }

  dlg:slider{
    id = "mod_slot_count",
    label = "Slots (2)",
    min = 1, max = 6, value = 2,
    onchange = function()
      PT.sync_slider_label("mod_slot_count")
      PT.sync_ui_conditional_states()
    end,
  }
  -- Modulation slots discovery hint
  dlg:label{
    id = "mod_slot_hint",
    text = "Increase slot count to add more audio mappings",
  }

  -- Sync mod slot min/max labels to show real-value range based on target
  local function sync_mod_slot_labels(slot)
    local d = dlg.data
    local p = "mod" .. slot .. "_"
    local target = d[p .. "target"]
    local def = target and PT.PARAM_DEFS[target]
    local fmt = target and MOD_TARGET_FMT[target]
    if def and fmt then
      local lo, hi = def[1], def[2]
      local min_pct = (d[p .. "min"] or 0) / 100.0
      local max_pct = (d[p .. "max"] or 100) / 100.0
      local min_val = lo + (hi - lo) * min_pct
      local max_val = lo + (hi - lo) * max_pct
      pcall(dlg.modify, dlg, { id = p .. "min", label = "Min (" .. string.format(fmt, min_val) .. ")" })
      pcall(dlg.modify, dlg, { id = p .. "max", label = "Max (" .. string.format(fmt, max_val) .. ")" })
    else
      pcall(dlg.modify, dlg, { id = p .. "min", label = "Min (%)" })
      pcall(dlg.modify, dlg, { id = p .. "max", label = "Max (%)" })
    end
  end

  -- Sync all mod slot labels (called from sync_ui_conditional_states)
  function PT.sync_all_mod_labels()
    for i = 1, 6 do sync_mod_slot_labels(i) end
  end

  -- Auto-switch to (custom) when any mod slot field is changed by the user
  local function mod_slot_changed()
    if PT.audio and PT.audio._hydrating_preset then return end
    local cur = dlg.data.audio_mod_preset
    if cur and cur ~= "(custom)" then
      dlg:modify{ id = "audio_mod_preset", option = "(custom)" }
    end
    -- Update real-value labels for all visible slots
    for i = 1, 6 do sync_mod_slot_labels(i) end
  end

  for i, def in ipairs(SLOT_DEFAULTS) do
    local prefix = "mod" .. i .. "_"

    dlg:check{
      id = prefix .. "enable",
      text = "Slot " .. i,
      selected = (i <= 2),
      onchange = mod_slot_changed,
    }
    dlg:check{
      id = prefix .. "invert",
      text = "Invert",
      selected = false,
      onchange = mod_slot_changed,
    }
    dlg:combobox{
      id = prefix .. "source",
      label = "Source",
      options = GLOBAL_SOURCES,
      option = def[1],
      onchange = mod_slot_changed,
    }
    dlg:combobox{
      id = prefix .. "target",
      label = "Target",
      options = MOD_TARGETS,
      option = def[2],
      onchange = mod_slot_changed,
    }
    dlg:slider{
      id = prefix .. "min",
      label = "Min (%)",
      min = 0, max = 100, value = def[3],
      onchange = mod_slot_changed,
    }
    dlg:slider{
      id = prefix .. "max",
      label = "Max (%)",
      min = 0, max = 100, value = def[4],
      onchange = mod_slot_changed,
    }
    dlg:slider{
      id = prefix .. "attack",
      label = "Attack",
      min = 1, max = 30, value = def[5],
      onchange = mod_slot_changed,
    }
    dlg:slider{
      id = prefix .. "release",
      label = "Release",
      min = 1, max = 60, value = def[6],
      onchange = mod_slot_changed,
    }
  end

  -- Initial: hide slots beyond default count (2)
  for i = 3, 6 do
    local p = "mod" .. i .. "_"
    for _, s in ipairs({ "enable", "invert", "source", "target", "min", "max", "attack", "release" }) do
      pcall(function() dlg:modify{ id = p .. s, visible = false } end)
    end
  end

  dlg:check{
    id = "audio_advanced",
    text = "Advanced",
    selected = false,
    onchange = function()
      PT.sync_ui_conditional_states()
    end,
  }

  dlg:check{
    id = "audio_use_expressions",
    text = "Custom Expressions",
    selected = false,
    onchange = function()
      PT.sync_ui_conditional_states()
    end,
  }
  -- Expression variables tooltip
  dlg:label{
    id = "expr_vars_hint",
    text = "Variables: t, frame, global_rms, global_onset, bass, mid, treble, sub_bass, low_mid, upper_mid, presence, brilliance, air, centroid, flux, sin, cos, sqrt, abs, min, max, pow",
  }
  dlg:combobox{
    id = "audio_expr_preset",
    label = "Expr Preset",
    options = { "(manual)" },
    option = "(manual)",
    onchange = function()
      local sel = dlg.data.audio_expr_preset
      if sel == "(manual)" then return end
      if PT.state.connected then
        PT.send({ action = "get_expression_preset", preset_name = sel })
      end
    end,
  }

  -- Expression entry fields (data-driven)
  for _, e in ipairs(EXPR_FIELDS) do
    dlg:entry{ id = e[1], label = e[2], text = "", hexpand = true }
  end

  dlg:check{
    id = "audio_random_seed",
    text = "Random seed per frame",
    selected = false,
  }

  dlg:combobox{
    id = "mp4_quality",
    label = "MP4 Quality",
    options = { "high", "web", "archive", "raw" },
    option = "high",
  }
  dlg:combobox{
    id = "mp4_scale",
    label = "MP4 Scale",
    options = { "4x", "2x", "1x", "8x" },
    option = "4x",
  }
  dlg:button{
    id = "export_mp4_btn",
    text = "Export MP4",
    hexpand = true,
    onclick = function()
      local out_dir = PT.audio.last_output_dir or PT.anim.output_dir
      if not out_dir then
        app.alert("No audio animation output to export. Generate first.")
        return
      end
      local d = dlg.data
      local scale_str = d.mp4_scale or "4x"
      local scale = tonumber(scale_str:match("(%d+)")) or 4
      dlg:modify{ id = "export_mp4_btn", enabled = false }
      PT.update_status("Exporting MP4...")
      PT.send({
        action       = "export_mp4",
        output_dir   = out_dir,
        audio_path   = (d.audio_file ~= "" and d.audio_file) or nil,
        fps          = tonumber(d.audio_fps) or 24,
        scale_factor = scale,
        quality      = d.mp4_quality or "high",
        prompt       = d.prompt or nil,
      })
    end,
  }

  -- Conditional states handled by PT.sync_ui_conditional_states() at dialog build complete
  -- Action button: must start disabled (not a conditional state)
  dlg:modify{ id = "export_mp4_btn", enabled = false }

  -- Initial: hide advanced sections (expressions, choreography)
  pcall(function() dlg:modify{ id = "audio_use_expressions", visible = false } end)
  pcall(function() dlg:modify{ id = "audio_expr_preset", visible = false } end)
  pcall(function() dlg:modify{ id = "expr_vars_hint", visible = false } end)
  for _, e in ipairs(EXPR_FIELDS) do
    pcall(function() dlg:modify{ id = e[1], visible = false } end)
  end
  pcall(function() dlg:modify{ id = "audio_random_seed", visible = false } end)
  pcall(function() dlg:modify{ id = "audio_choreography", visible = false } end)
end

-- ─── Tab: QR Code ───────────────────────────────────────────

local function build_tab_qrcode()
  local dlg = PT.dlg

  dlg:check{
    id = "qr_use_source",
    text = "Use Layer (Illusion Art)",
    selected = false,
  }

  dlg:slider{
    id = "qr_denoise",
    label = "Denoise (0.75)",
    min = 5, max = 100, value = 75,
    onchange = onchange_sync("qr_denoise"),
  }

  dlg:slider{
    id = "qr_conditioning_scale",
    label = "CN Scale (1.50)",
    min = 0, max = 300, value = 150,
    onchange = onchange_sync("qr_conditioning_scale"),
  }

  dlg:slider{
    id = "qr_guidance_start",
    label = "Guide Start (0.00)",
    min = 0, max = 100, value = 0,
    onchange = onchange_sync("qr_guidance_start"),
  }

  dlg:slider{
    id = "qr_guidance_end",
    label = "Guide End (0.80)",
    min = 0, max = 100, value = 80,
    onchange = onchange_sync("qr_guidance_end"),
  }

  dlg:slider{
    id = "qr_steps",
    label = "Steps (20)",
    min = 4, max = 50, value = 20,
    onchange = onchange_sync("qr_steps"),
  }

  dlg:slider{
    id = "qr_cfg",
    label = "CFG (7.5)",
    min = 10, max = 200, value = 75,
    onchange = onchange_sync("qr_cfg"),
  }
end

-- ─── Trigger Functions ──────────────────────────────────────

function PT.trigger_generate()
  if PT.state.generating or PT.state.animating then
    PT.update_status("Already generating...")
    return
  end
  local dlg = PT.dlg
  local d = dlg.data
  -- Reset sequence for non-loop single gen (new sequence each click)
  local is_loop = d.loop_check or d.random_loop_check
  if not is_loop then
    PT.finalize_sequence()
  end
  -- Initialize loop state (only on first entry, not on re-entry)
  if is_loop then init_loop_state("generate") end

  -- Random loop: first generate a random prompt, then generate image
  if PT.loop.random_mode then
    dlg:modify{ id = "action_btn", text = "LOOPING...", enabled = false }
    dlg:modify{ id = "cancel_btn", enabled = true }
    PT.loop.counter = PT.loop.counter + 1
    PT.update_status("Random Loop #" .. PT.loop.counter .. " — Generating prompt...")
    PT.send({ action = "generate_prompt", locked_fields = PT.loop.locked_fields, randomness = dlg.data.randomness })
    return
  end

  local req = PT.build_generate_request()
  if not req then PT.reset_loop_state(); return end
  if not PT.attach_source_image(req) then PT.reset_loop_state(); return end

  PT.state.generating = true
  PT.state.gen_step_start = os.clock()
  PT.start_gen_timeout()
  dlg:modify{ id = "action_btn", text = PT.loop.mode and "LOOPING..." or "GENERATE", enabled = false }
  dlg:modify{ id = "cancel_btn", enabled = true }
  if PT.loop.mode then
    PT.loop.counter = PT.loop.counter + 1
    PT.update_status("Loop #" .. PT.loop.counter .. " — Generating...")
  else
    PT.update_status("Generating...")
  end
  PT.send(req)
end

function PT.trigger_animate()
  if PT.state.animating or PT.state.generating then return end
  local dlg = PT.dlg
  local d = dlg.data

  -- Initialize loop state (only on first entry, not on re-entry)
  init_loop_state("animate")

  -- Random loop: generate prompt first
  if PT.loop.random_mode and PT.loop.mode and PT.loop.counter == 0 then
    dlg:modify{ id = "action_btn", text = "LOOPING...", enabled = false }
    dlg:modify{ id = "cancel_btn", enabled = true }
    PT.loop.counter = PT.loop.counter + 1
    PT.update_status("Random Loop #" .. PT.loop.counter .. " — Generating prompt...")
    PT.send({ action = "generate_prompt", locked_fields = PT.loop.locked_fields, randomness = dlg.data.randomness })
    return
  end

  local req = PT.build_animation_request()
  if not req then PT.reset_loop_state(); return end
  if not PT.attach_source_image(req) then PT.reset_loop_state(); return end

  PT.state.animating = true
  PT.state.gen_step_start = os.clock()
  PT.start_gen_timeout()
  dlg:modify{ id = "action_btn", text = PT.loop.mode and "LOOPING..." or "ANIMATE", enabled = false }
  dlg:modify{ id = "cancel_btn", enabled = true }
  if PT.loop.mode then
    if PT.loop.counter == 0 then PT.loop.counter = 1 end
    PT.update_status("Animate Loop #" .. PT.loop.counter .. " — Animating...")
  else
    PT.update_status("Animating...")
  end
  PT.send(req)
end

function PT.trigger_audio_generate()
  if PT.state.generating or PT.state.animating or PT.audio.generating then return end
  local dlg = PT.dlg
  local d = dlg.data
  local path = d.audio_file
  if not path or path == "" then
    app.alert("Select an audio file first.")
    return
  end
  if not PT.audio.analyzed then
    app.alert("Analyze the audio file first.")
    return
  end

  -- Initialize loop state (only on first entry, not on re-entry)
  init_loop_state("audio")

  -- Random loop first entry: generate prompt first
  if PT.loop.random_mode and PT.loop.mode and PT.loop.counter == 0 then
    dlg:modify{ id = "action_btn", text = "LOOPING...", enabled = false }
    dlg:modify{ id = "cancel_btn", enabled = true }
    PT.loop.counter = PT.loop.counter + 1
    PT.update_status("Random Loop #" .. PT.loop.counter .. " — Generating prompt...")
    PT.send({ action = "generate_prompt", locked_fields = PT.loop.locked_fields, randomness = dlg.data.randomness })
    return
  end

  local req = PT.build_audio_reactive_request()
  if not req then PT.reset_loop_state(); return end
  if not PT.attach_source_image(req) then PT.reset_loop_state(); return end

  PT.audio.generating = true
  PT.state.animating = true
  PT.state.gen_step_start = os.clock()
  local audio_timeout = 180 + (PT.audio.total_frames * 15)
  PT.start_gen_timeout(math.max(PT.cfg.GEN_TIMEOUT, audio_timeout))
  dlg:modify{ id = "action_btn", text = PT.loop.mode and "LOOPING..." or "AUDIO GEN", enabled = false }
  dlg:modify{ id = "cancel_btn", enabled = true }
  if PT.loop.mode then
    if PT.loop.counter == 0 then PT.loop.counter = 1 end
    PT.update_status("Audio Loop #" .. PT.loop.counter .. " — Generating...")
  else
    PT.update_status("Generating audio animation...")
  end
  PT.send(req)
end

function PT.trigger_qr_generate()
  if PT.state.generating or PT.state.animating then return end
  local dlg = PT.dlg
  local d = dlg.data

  -- Capture active layer as control image
  local control_b64 = PT.capture_active_layer()
  if not control_b64 then
    app.alert("No active layer to use as control image.")
    return
  end

  local req, use_source = PT.build_qr_request()
  if not req then return end
  req.control_image = control_b64

  -- Illusion art: capture flattened sprite as source image for img2img blend
  if use_source then
    local src_b64 = PT.capture_flattened()
    if not src_b64 then
      app.alert("No sprite to use as source for illusion art.")
      return
    end
    req.source_image = src_b64
  end

  PT.last_request = PT.shallow_copy_request(req)
  PT.state.generating = true
  PT.state.gen_step_start = os.clock()
  PT.start_gen_timeout()
  dlg:modify{ id = "action_btn", text = "QR GENERATING...", enabled = false }
  dlg:modify{ id = "cancel_btn", enabled = true }
  PT.update_status("Generating illusion art...")
  PT.send(req)
end

function PT.update_action_button(tab)
  if not PT.dlg then return end
  local texts = {
    tab_gen   = "GENERATE",
    tab_pp    = "GENERATE (PP)",
    tab_anim  = "ANIMATE",
    tab_qr    = "QR GENERATE",
    tab_audio = "AUDIO GEN",
  }
  PT.dlg:modify{ id = "action_btn", text = texts[tab] or "GENERATE" }
  -- Loop controls: supported in gen/pp/anim always, audio when analyzed, never QR
  -- Also disabled during active generation/animation
  local loop_enabled = (tab ~= "tab_qr") and not PT.state.generating and not PT.state.animating
  if tab == "tab_audio" then loop_enabled = PT.audio.analyzed and not PT.state.generating and not PT.state.animating end
  PT.dlg:modify{ id = "loop_check", enabled = loop_enabled }
  PT.dlg:modify{ id = "random_loop_check", enabled = loop_enabled }
end

-- ─── Actions Panel ─────────────────────────────────────────

local function build_actions_panel()
  local dlg = PT.dlg

  dlg:separator{ text = "Actions", hexpand = true }

  -- Visual progress bar (canvas: fills proportionally to progress_pct)
  dlg:canvas{
    id = "progress_bar",
    width = 300,
    height = 10,
    hexpand = true,
    onpaint = function(ev)
      local gc = ev.context
      local w, h = gc.width, gc.height
      -- Background track
      gc.color = Color{ r = 32, g = 32, b = 38, a = 255 }
      gc:fillRect(Rectangle(0, 0, w, h))
      -- Fill bar
      local pct = PT.state.progress_pct or 0
      if pct > 0 then
        local r = math.floor(50 + (1 - pct / 100) * 20)
        local g = math.floor(110 + (pct / 100) * 70)
        local b = math.floor(200 + (pct / 100) * 55)
        gc.color = Color{ r = r, g = g, b = b, a = 255 }
        gc:fillRect(Rectangle(0, 0, math.max(1, math.floor(w * pct / 100)), h))
      end
      -- Border
      gc.color = Color{ r = 60, g = 60, b = 70, a = 255 }
      gc:strokeRect(Rectangle(0, 0, w, h))
    end,
  }

  -- Contextual action button: text and behavior change based on active tab
  dlg:button{
    id = "action_btn",
    text = "GENERATE",
    focus = true,
    hexpand = true,
    onclick = function()
      if PT.state.generating or PT.state.animating then return end
      local d = dlg.data
      local tab = d.main_tabs

      -- Audio requires analysis
      if tab == "tab_audio" and not PT.audio.analyzed then
        app.alert("Analyze the audio file first.")
        return
      end

      -- Randomize-before path: generate prompt first, then dispatch
      if d.randomize_before and not PT.loop.random_mode then
        PT.state.pending_action = TAB_PENDING[tab]
        PT.send({
          action = "generate_prompt",
          locked_fields = PT.build_locked_fields(),
          randomness = d.randomness,
        })
        dlg:modify{ id = "action_btn", enabled = false }
        dlg:modify{ id = "cancel_btn", enabled = true }
        PT.start_gen_timeout(30)
        PT.update_status("Randomizing prompt...")
        return
      end

      -- Direct dispatch based on active tab
      local triggers = {
        tab_gen   = PT.trigger_generate,
        tab_pp    = PT.trigger_generate,
        tab_anim  = PT.trigger_animate,
        tab_qr    = PT.trigger_qr_generate,
        tab_audio = PT.trigger_audio_generate,
      }
      local trigger = triggers[tab]
      if trigger then trigger() end
    end,
  }

  dlg:button{
    id = "cancel_btn",
    text = "CANCEL",
    hexpand = true,
    onclick = function()
      PT.state.pending_action = nil
      PT.reset_loop_state()
      PT.timers.loop = PT.stop_timer(PT.timers.loop)
      if PT.state.generating or PT.state.animating then
        PT.send({ action = "cancel" })
        PT.state.cancel_pending = true
        dlg:modify{ id = "action_btn", enabled = false }
        PT.update_status("Cancelling...")
        -- Immediate: kill pending frames to prevent zombie imports
        PT.clear_response_queue()
        PT.stop_refresh_timer()
        if PT.audio.generating or PT.audio.analyzing then
          dlg:modify{ id = "audio_analyze_btn", enabled = PT.state.connected }
        end
        PT.timers.cancel_safety = PT.stop_timer(PT.timers.cancel_safety)
        PT.timers.cancel_safety = Timer{
          interval = PT.cfg.CANCEL_TIMEOUT,
          ontick = function()
            PT.timers.cancel_safety = PT.stop_timer(PT.timers.cancel_safety)
            if PT.state.cancel_pending then
              PT.state.cancel_pending = false
              PT.state.generating = false
              PT.state.animating = false
              PT.audio.generating = false
              PT.audio.analyzing = false
              PT.stop_gen_timeout()
              PT.stop_refresh_timer()
              PT.clear_response_queue()
              PT.finalize_sequence()
              PT.update_status("Cancel timeout — UI reset")
              PT.reset_ui_buttons()
            end
          end,
        }
        PT.timers.cancel_safety:start()
      else
        PT.finalize_sequence()
        PT.reset_ui_buttons()
        PT.update_status("Cancelled")
      end
    end,
  }

  -- Randomize + Loop options
  dlg:check{
    id = "randomize_before",
    text = "Randomize",
    selected = false,
  }
  -- A/B Comparison Mode
  dlg:check{
    id = "ab_compare",
    text = "A/B Compare",
    selected = false,
  }
  dlg:check{
    id = "loop_check",
    text = "Loop",
    selected = false,
  }
  dlg:check{
    id = "random_loop_check",
    text = "Random Loop",
    selected = false,
  }

  dlg:slider{
    id = "randomness",
    label = "Randomness (0 — Off)",
    min = 0, max = 20, value = 0,
    onchange = function()
      local v = dlg.data.randomness
      local names = { [0]="Off", [5]="Subtle", [10]="Moderate", [15]="Wild", [20]="Chaos" }
      local name = names[v] or ""
      local suffix = name ~= "" and (" — " .. name) or ""
      dlg:modify{ id = "randomness", label = "Randomness (" .. v .. suffix .. ")" }
    end,
  }

  dlg:combobox{
    id = "mode",
    label = "Mode",
    options = {
      "txt2img", "img2img", "inpaint",
      "controlnet_openpose", "controlnet_canny",
      "controlnet_scribble", "controlnet_lineart",
      "controlnet_qrcode",
    },
    option = "txt2img",
    onchange = function()
      PT.sync_ui_conditional_states()
    end,
  }

  dlg:combobox{
    id = "loop_seed_combo",
    label = "Loop Seed",
    options = { "random", "increment" },
    option = "random",
  }

  -- Initial disabled state (must be set AFTER widget creation — Aseprite bug)
  dlg:modify{ id = "action_btn", enabled = false }
  dlg:modify{ id = "cancel_btn", enabled = false }
end

-- ─── Main Build ─────────────────────────────────────────────

function PT.build_dialog()
  PT.dlg = Dialog{
    title = "SDDj",
    resizeable = true,
    onclose = function()
      pcall(PT.save_settings)
      -- Stop all PT.timers (heartbeat, connect, gen_timeout, loop, etc.)
      if PT.timers then
        for key, timer in pairs(PT.timers) do
          if timer then pcall(function() timer:stop() end) end
          PT.timers[key] = nil
        end
      end
      -- Stop module-private timers (not in PT.timers table)
      if PT.stop_refresh_timer then pcall(PT.stop_refresh_timer) end
      if PT.clear_response_queue then pcall(PT.clear_response_queue) end
      -- Disarm connection + WebSocket immediately
      if PT.state then
        PT.state.connected = false
        PT.state.connecting = false
      end
      if PT.reconnect then
        PT.reconnect.manual_disconnect = true
      end
      -- Abandon WebSocket reference (no close, no sendText)
      PT.ws_handle = nil
      PT.dlg = nil
    end,
  }

  build_connection_section()
  build_actions_panel()

  PT.dlg:tab{ id = "tab_gen", text = "Generate" }
  build_tab_generate()

  PT.dlg:tab{ id = "tab_pp", text = "Post-Process" }
  build_tab_postprocess()

  PT.dlg:tab{ id = "tab_anim", text = "Animation" }
  build_tab_animation()

  PT.dlg:tab{ id = "tab_qr", text = "QR Code" }
  build_tab_qrcode()

  PT.dlg:tab{ id = "tab_audio", text = "Audio" }
  build_tab_audio()

  PT.dlg:endtabs{
    id = "main_tabs",
    selected = "tab_gen",
    onchange = function()
      PT.update_action_button(PT.dlg.data.main_tabs)
    end,
  }

  PT.sync_ui_conditional_states()
  PT.dlg:show{ wait = false, autoscrollbars = true }
end

end
