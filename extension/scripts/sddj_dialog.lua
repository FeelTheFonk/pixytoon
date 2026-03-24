--
-- SDDj — Dialog Construction
--

return function(PT)

-- ─── Slider Label Helper ────────────────────────────────────

-- Returns an onchange callback that updates a slider's label with a formatted value.
-- Usage: onchange = slider_label("cfg_scale", "CFG (%.1f)", 10.0)
local function slider_label(id, fmt, divisor)
  return function()
    PT.dlg:modify{ id = id, label = string.format(fmt, PT.dlg.data[id] / divisor) }
  end
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
    onclick = function()
      if PT.state.connected then
        PT.disconnect()
      else
        PT.cfg.DEFAULT_SERVER_URL = dlg.data.server_url or PT.cfg.DEFAULT_SERVER_URL
        PT.connect()
      end
    end,
  }
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
      local name_dlg = Dialog{ title = "Save Preset" }
      name_dlg:entry{ id = "pname", label = "Name", text = "", hexpand = true }
      name_dlg:button{ id = "ok", text = "Save" }
      name_dlg:button{ id = "cancel", text = "Cancel" }
      name_dlg:show()
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
        denoise_strength = dlg.data.denoise / 100.0,
        post_process = PT.build_post_process(),
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
        PT.send({ action = "delete_preset", preset_name = sel })
      end
    end,
  }
  dlg:button{
    id = "load_meta_btn",
    text = "Load",
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
    id = "mode",
    label = "Mode",
    options = {
      "txt2img", "img2img", "inpaint",
      "controlnet_openpose", "controlnet_canny",
      "controlnet_scribble", "controlnet_lineart",
    },
    option = "txt2img",
    onchange = function()
      local m = dlg.data.mode
      -- Show hint about required inputs
      if m == "inpaint" then
        dlg:modify{ id = "mode", label = "Mode (needs mask)" }
      elseif m == "img2img" or m:find("controlnet_") then
        dlg:modify{ id = "mode", label = "Mode (needs layer)" }
      else
        dlg:modify{ id = "mode", label = "Mode" }
      end
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
    onchange = slider_label("lora_weight", "LoRA (%.2f)", 100.0),
  }

  dlg:entry{
    id = "prompt",
    label = "Prompt",
    text = "",
    hexpand = true,
  }
  dlg:button{
    id = "randomize_btn",
    text = "Randomize",
    onclick = function()
      local locked = {}
      if PT.dlg.data.lock_subject and PT.dlg.data.fixed_subject ~= "" then
        locked.subject = PT.dlg.data.fixed_subject
      end
      PT.send({ action = "generate_prompt", locked_fields = locked, randomness = PT.dlg.data.randomness })
      PT.update_status("Generating prompt...")
    end,
  }
  dlg:check{
    id = "lock_subject",
    label = "Lock Subject",
    selected = false,
  }
  dlg:entry{
    id = "fixed_subject",
    label = "Subject",
    text = "",
    hexpand = true,
  }

  dlg:entry{
    id = "negative_prompt",
    label = "Neg. Prompt",
    text = "blurry, antialiased, smooth gradient, photorealistic, 3d render, soft edges, low quality, worst quality",
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
    onchange = slider_label("neg_ti_weight", "Emb. (%.2f)", 100.0),
  }

  dlg:combobox{
    id = "output_size",
    label = "Size",
    options = {
      "512x512", "512x768", "768x512", "768x768",
      "384x384", "256x256", "128x128", "96x96",
      "64x64",
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
    onchange = slider_label("denoise", "Strength (%.2f)", 100.0),
  }

  dlg:slider{
    id = "steps",
    label = "Steps (8)",
    min = 1, max = 100, value = 8,
    onchange = function()
      dlg:modify{ id = "steps", label = "Steps (" .. dlg.data.steps .. ")" }
    end,
  }

  dlg:slider{
    id = "cfg_scale",
    label = "CFG (5.0)",
    min = 0, max = 300, value = 50,
    onchange = slider_label("cfg_scale", "CFG (%.1f)", 10.0),
  }

  dlg:slider{
    id = "clip_skip",
    label = "CLIP Skip (2)",
    min = 1, max = 12, value = 2,
    onchange = function()
      dlg:modify{ id = "clip_skip", label = "CLIP Skip (" .. dlg.data.clip_skip .. ")" }
    end,
  }
end

-- ─── Tab: Post-Process ──────────────────────────────────────

local function build_tab_postprocess()
  local dlg = PT.dlg

  dlg:check{
    id = "pixelate",
    label = "Pixelate",
    selected = false,
  }

  dlg:slider{
    id = "pixel_size",
    label = "Target (128px)",
    min = 8, max = 512, value = 128,
    onchange = function()
      dlg:modify{ id = "pixel_size", label = "Target (" .. dlg.data.pixel_size .. "px)" }
    end,
  }

  dlg:check{
    id = "quantize_enabled",
    label = "Quantize Colors",
    selected = false,
  }

  dlg:slider{
    id = "colors",
    label = "Colors (32)",
    min = 2, max = 256, value = 32,
    onchange = function()
      dlg:modify{ id = "colors", label = "Colors (" .. dlg.data.colors .. ")" }
    end,
  }

  dlg:combobox{
    id = "quantize_method",
    label = "Quantize",
    options = { "kmeans", "median_cut", "octree" },
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
      -- Collect colors from custom hex field or current preset
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
    onchange = function()
      dlg:modify{ id = "anim_steps", label = "Steps (" .. dlg.data.anim_steps .. ")" }
    end,
  }
  dlg:slider{
    id = "anim_cfg",
    label = "CFG (5.0)",
    min = 0, max = 200, value = 50,
    onchange = slider_label("anim_cfg", "CFG (%.1f)", 10.0),
  }

  dlg:slider{
    id = "anim_frames",
    label = "Frames (8)",
    min = 2, max = 120, value = 8,
    onchange = function()
      dlg:modify{ id = "anim_frames", label = "Frames (" .. dlg.data.anim_frames .. ")" }
    end,
  }

  dlg:slider{
    id = "anim_duration",
    label = "Duration (100ms)",
    min = 50, max = 2000, value = 100,
    onchange = function()
      dlg:modify{ id = "anim_duration", label = "Duration (" .. dlg.data.anim_duration .. "ms)" }
    end,
  }

  dlg:slider{
    id = "anim_denoise",
    label = "Strength (0.30)",
    min = 5, max = 100, value = 30,
    onchange = slider_label("anim_denoise", "Strength (%.2f)", 100.0),
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
  }

  dlg:slider{
    id = "anim_freeinit_iters",
    label = "FreeInit Iters",
    min = 1, max = 3, value = 2,
  }
end

-- ─── Tab: Audio ───────────────────────────────────────────

local GLOBAL_SOURCES = {
  "global_rms", "global_onset", "global_centroid",
  "global_low", "global_mid", "global_high",
  "global_sub_bass", "global_upper_mid", "global_presence",
  "global_beat",
}

local MOD_TARGETS = {
  "denoise_strength", "cfg_scale", "noise_amplitude",
  "controlnet_scale", "seed_offset", "palette_shift",
  "frame_cadence",
  -- Motion / camera (smooth Deforum-like)
  "motion_x", "motion_y", "motion_zoom", "motion_rotation",
}

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

  dlg:combobox{
    id = "audio_fps",
    label = "FPS",
    options = { "8", "12", "15", "24", "30" },
    option = "24",
  }
  dlg:slider{
    id = "audio_steps",
    label = "Steps (8)",
    min = 1, max = 50, value = 8,
    onchange = function()
      dlg:modify{ id = "audio_steps", label = "Steps (" .. dlg.data.audio_steps .. ")" }
    end,
  }
  dlg:slider{
    id = "audio_cfg",
    label = "CFG (5.0)",
    min = 0, max = 200, value = 50,
    onchange = slider_label("audio_cfg", "CFG (%.1f)", 10.0),
  }
  dlg:slider{
    id = "audio_denoise",
    label = "Strength (0.50)",
    min = 0, max = 100, value = 50,
    onchange = slider_label("audio_denoise", "Strength (%.2f)", 100.0),
  }
  dlg:slider{
    id = "audio_frame_duration",
    label = "Frame (42ms)",
    min = 30, max = 2000, value = 42,
    onchange = function()
      dlg:modify{ id = "audio_frame_duration",
        label = "Frame (" .. dlg.data.audio_frame_duration .. "ms)" }
    end,
  }
  dlg:slider{
    id = "audio_max_frames",
    label = "Max Frames (0=all)",
    min = 0, max = 3600, value = 0,
    onchange = function()
      local v = dlg.data.audio_max_frames
      dlg:modify{ id = "audio_max_frames",
        label = v == 0 and "Max Frames (0=all)" or ("Max Frames (" .. v .. ")") }
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
  }
  dlg:slider{
    id = "audio_freeinit_iters",
    label = "FreeInit Iters",
    min = 1, max = 3, value = 2,
  }

  -- Modulation
  dlg:separator{ text = "Modulation" }
  dlg:combobox{
    id = "audio_mod_preset",
    label = "Preset",
    options = {
      "(custom)",
      -- Genre-specific
      "electronic_pulse", "rock_energy", "hiphop_bounce",
      "classical_flow", "ambient_drift",
      -- Style-specific
      "glitch_chaos", "smooth_morph", "rhythmic_pulse",
      "atmospheric", "abstract_noise",
      -- Complexity levels
      "one_click_easy", "beginner_balanced",
      "intermediate_full", "advanced_max",
      -- Target-specific
      "controlnet_reactive", "seed_scatter", "noise_sculpt",
      -- Motion / camera
      "gentle_drift", "pulse_zoom", "slow_rotate", "cinematic_sweep",
      -- Legacy
      "energetic", "ambient", "bass_driven",
    },
    option = "(custom)",
    onchange = function()
      local sel = dlg.data.audio_mod_preset
      if sel == "(custom)" then return end
      -- Preset is applied server-side via modulation_preset field in request.
      -- Status feedback so user knows the selection registered.
      PT.update_status("Preset '" .. sel .. "' selected")
    end,
  }

  dlg:slider{
    id = "mod_slot_count",
    label = "Slots (1)",
    min = 1, max = 4, value = 1,
    onchange = function()
      dlg:modify{ id = "mod_slot_count", label = "Slots (" .. dlg.data.mod_slot_count .. ")" }
    end,
  }

  -- Slot defaults: [source, target, min, max, attack, release]
  local slot_defaults = {
    { "global_rms",    "denoise_strength",  15, 65, 2, 8 },
    { "global_onset",  "cfg_scale",         30, 80, 2, 8 },
    { "global_low",    "noise_amplitude",    0, 30, 2, 8 },
    { "global_high",   "seed_offset",        0, 50, 2, 8 },
  }

  for i, def in ipairs(slot_defaults) do
    local prefix = "mod" .. i .. "_"

    dlg:check{
      id = prefix .. "enable",
      text = "Slot " .. i,
      selected = true,
    }
    dlg:combobox{
      id = prefix .. "source",
      label = "Source",
      options = GLOBAL_SOURCES,
      option = def[1],
    }
    dlg:combobox{
      id = prefix .. "target",
      label = "Target",
      options = MOD_TARGETS,
      option = def[2],
    }
    dlg:slider{
      id = prefix .. "min",
      label = "Min (%)",
      min = 0, max = 100, value = def[3],
    }
    dlg:slider{
      id = prefix .. "max",
      label = "Max (%)",
      min = 0, max = 100, value = def[4],
    }
    dlg:slider{
      id = prefix .. "attack",
      label = "Attack",
      min = 1, max = 30, value = def[5],
    }
    dlg:slider{
      id = prefix .. "release",
      label = "Release",
      min = 1, max = 60, value = def[6],
    }
  end

  dlg:check{
    id = "audio_advanced",
    text = "Advanced",
    selected = false,
  }

  dlg:check{
    id = "audio_use_expressions",
    text = "Custom Expressions",
    selected = false,
  }
  dlg:entry{
    id = "expr_denoise",
    label = "denoise",
    text = "",
    hexpand = true,
  }
  dlg:entry{
    id = "expr_cfg",
    label = "cfg_scale",
    text = "",
    hexpand = true,
  }
  dlg:entry{
    id = "expr_noise",
    label = "noise_amp",
    text = "",
    hexpand = true,
  }
  dlg:entry{
    id = "expr_controlnet",
    label = "cn_scale",
    text = "",
    hexpand = true,
  }
  dlg:entry{
    id = "expr_seed",
    label = "seed_off",
    text = "",
    hexpand = true,
  }
  dlg:entry{
    id = "expr_palette",
    label = "pal_shift",
    text = "",
    hexpand = true,
  }
  dlg:entry{
    id = "expr_cadence",
    label = "cadence",
    text = "",
    hexpand = true,
  }
  dlg:entry{
    id = "expr_motion_x",
    label = "motion_x",
    text = "",
    hexpand = true,
  }
  dlg:entry{
    id = "expr_motion_y",
    label = "motion_y",
    text = "",
    hexpand = true,
  }
  dlg:entry{
    id = "expr_motion_zoom",
    label = "zoom",
    text = "",
    hexpand = true,
  }
  dlg:entry{
    id = "expr_motion_rot",
    label = "rotation",
    text = "",
    hexpand = true,
  }

  dlg:check{
    id = "audio_random_seed",
    text = "Random seed per frame",
    selected = false,
  }

  -- Prompt Schedule (advanced only)
  dlg:check{
    id = "audio_prompt_schedule",
    text = "Prompt Schedule",
    selected = false,
  }
  for i = 1, 3 do
    dlg:entry{
      id = "ps" .. i .. "_time",
      label = "T" .. i .. " (s-s)",
      text = "",
      hexpand = true,
    }
    dlg:entry{
      id = "ps" .. i .. "_prompt",
      label = "P" .. i,
      text = "",
      hexpand = true,
    }
  end

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
    enabled = false,
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
        audio_path   = d.audio_file or nil,
        fps          = tonumber(d.audio_fps) or 24,
        scale_factor = scale,
        quality      = d.mp4_quality or "high",
        prompt       = d.prompt or nil,
      })
    end,
  }
end

-- ─── Actions Panel ──────────────────────────────────────────

-- ─── Trigger Functions (extracted for contextual button dispatch) ──────

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
  -- Initialize loop state
  if is_loop then
    PT.loop.mode = true
    PT.loop.counter = 0
    PT.loop.seed_mode = d.loop_seed_combo or "random"
    PT.loop.random_mode = d.random_loop_check or false
    PT.loop.locked_fields = {}
    if d.lock_subject and d.fixed_subject ~= "" then
      PT.loop.locked_fields.subject = d.fixed_subject
    end
  end

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
  if not req then PT.loop.mode = false; PT.loop.random_mode = false; return end
  if not PT.attach_source_image(req) then PT.loop.mode = false; PT.loop.random_mode = false; return end

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
  local gw, gh = PT.parse_size()
  local tag_name = d.anim_tag or ""
  if tag_name == "" then tag_name = nil end

  local req = {
    action = "generate_animation",
    method = d.anim_method,
    prompt = d.prompt,
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
  if not PT.attach_source_image(req) then return end

  PT.state.animating = true
  PT.state.gen_step_start = os.clock()
  PT.start_gen_timeout()
  dlg:modify{ id = "action_btn", enabled = false }
  dlg:modify{ id = "cancel_btn", enabled = true }
  PT.update_status("Animating...")
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

  local req = PT.build_audio_reactive_request()
  if not req then return end
  if not PT.attach_source_image(req) then return end

  PT.audio.generating = true
  PT.state.animating = true
  PT.state.gen_step_start = os.clock()
  local audio_timeout = 180 + (PT.audio.total_frames * 15)
  PT.start_gen_timeout(math.max(PT.cfg.GEN_TIMEOUT, audio_timeout))
  dlg:modify{ id = "action_btn", enabled = false }
  dlg:modify{ id = "cancel_btn", enabled = true }
  PT.update_status("Generating audio animation...")
  PT.send(req)
end

function PT.update_action_button(tab)
  if not PT.dlg then return end
  local texts = {
    tab_gen   = "GENERATE",
    tab_pp    = "GENERATE (PP)",
    tab_anim  = "ANIMATE",
    tab_audio = "AUDIO GEN",
  }
  PT.dlg:modify{ id = "action_btn", text = texts[tab] or "GENERATE" }
end

-- ─── Actions Panel ─────────────────────────────────────────

local function build_actions_panel()
  local dlg = PT.dlg

  dlg:separator{ text = "Actions", hexpand = true }

  -- Contextual action button: text and behavior change based on active tab
  dlg:button{
    id = "action_btn",
    text = "GENERATE",
    enabled = false,
    hexpand = true,
    onclick = function()
      if PT.state.generating or PT.state.animating then return end
      local d = dlg.data

      -- If Randomize is enabled: first generate a random prompt, then dispatch
      if d.randomize_before and not PT.loop.random_mode then
        local tab = d.main_tabs
        if tab == "tab_gen" or tab == "tab_pp" then
          PT.state.pending_action = "generate"
        elseif tab == "tab_anim" then
          PT.state.pending_action = "animate"
        elseif tab == "tab_audio" then
          PT.state.pending_action = "audio"
        end
        local locked = {}
        if d.lock_subject and d.fixed_subject ~= "" then
          locked.subject = d.fixed_subject
        end
        PT.send({
          action = "generate_prompt",
          locked_fields = locked,
          randomness = d.randomness,
        })
        dlg:modify{ id = "action_btn", enabled = false }
        dlg:modify{ id = "cancel_btn", enabled = true }
        PT.update_status("Randomizing prompt...")
        return
      end

      -- Direct dispatch based on active tab
      local tab = d.main_tabs
      if tab == "tab_gen" or tab == "tab_pp" then
        PT.trigger_generate()
      elseif tab == "tab_anim" then
        PT.trigger_animate()
      elseif tab == "tab_audio" then
        PT.trigger_audio_generate()
      end
    end,
  }

  dlg:button{
    id = "cancel_btn",
    text = "CANCEL",
    enabled = false,
    hexpand = true,
    onclick = function()
      PT.state.pending_action = nil
      PT.loop.mode = false
      PT.loop.random_mode = false
      PT.timers.loop = PT.stop_timer(PT.timers.loop)
      if PT.state.generating or PT.state.animating then
        PT.send({ action = "cancel" })
        PT.state.cancel_pending = true
        dlg:modify{ id = "action_btn", enabled = false }
        PT.update_status("Cancelling...")
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
    id = "loop_seed_combo",
    label = "Loop Seed",
    options = { "random", "increment" },
    option = "random",
  }

end

-- ─── Main Build ─────────────────────────────────────────────

function PT.build_dialog()
  PT.dlg = Dialog{
    title = "SDDj",
    resizeable = true,
    onclose = function()
      pcall(PT.save_settings)
      -- Shutdown + cleanup handled by exit(plugin) which is always called after onclose
      pcall(PT.disconnect)
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

  PT.dlg:tab{ id = "tab_audio", text = "Audio" }
  build_tab_audio()

  PT.dlg:endtabs{
    id = "main_tabs",
    selected = "tab_gen",
    onchange = function()
      PT.update_action_button(PT.dlg.data.main_tabs)
    end,
  }

  PT.dlg:show{ wait = false, autoscrollbars = true }
end

end
