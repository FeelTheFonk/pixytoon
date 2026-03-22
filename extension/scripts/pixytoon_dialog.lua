--
-- PixyToon — Dialog Construction
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
          and not PT.state.animating and not PT.live.mode then
        PT.send({ action = "cleanup" })
        PT.update_status("Cleaning up GPU...")
      else
        PT.update_status("Cannot cleanup during generation/live")
      end
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
      local is_txt = (m == "txt2img")
      dlg:modify{ id = "denoise", visible = not is_txt }
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
    text = "pixel art, PixArFK, game sprite, sharp pixels",
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
      PT.send({ action = "generate_prompt", locked_fields = locked })
      PT.update_status("Generating prompt...")
    end,
  }
  dlg:check{
    id = "lock_subject",
    label = "Lock Subject",
    selected = false,
    onchange = function()
      dlg:modify{ id = "fixed_subject", visible = dlg.data.lock_subject }
    end,
  }
  dlg:entry{
    id = "fixed_subject",
    label = "Subject",
    text = "",
    visible = false,
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
    onchange = function()
      dlg:modify{ id = "neg_ti_weight", visible = dlg.data.use_neg_ti }
    end,
  }

  dlg:slider{
    id = "neg_ti_weight",
    label = "Emb. (1.00)",
    min = 10, max = 200, value = 100,
    visible = false,
    onchange = slider_label("neg_ti_weight", "Emb. (%.2f)", 100.0),
  }

  dlg:combobox{
    id = "output_size",
    label = "Size",
    options = {
      "512x512", "512x768", "768x512", "768x768",
      "384x384", "256x256", "128x128", "96x96",
      "64x64", "48x48", "32x32",
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
    label = "Steps",
    min = 1, max = 100, value = 8,
  }

  dlg:slider{
    id = "cfg_scale",
    label = "CFG (5.0)",
    min = 0, max = 300, value = 50,
    onchange = slider_label("cfg_scale", "CFG (%.1f)", 10.0),
  }

  dlg:slider{
    id = "clip_skip",
    label = "CLIP Skip",
    min = 1, max = 12, value = 2,
  }
end

-- ─── Tab: Post-Process ──────────────────────────────────────

local function build_tab_postprocess()
  local dlg = PT.dlg

  dlg:check{
    id = "pixelate",
    label = "Pixelate",
    selected = false,
    onchange = function()
      dlg:modify{ id = "pixel_size", visible = dlg.data.pixelate }
    end,
  }

  dlg:slider{
    id = "pixel_size",
    label = "Target (128px)",
    min = 8, max = 512, value = 128,
    visible = false,
    onchange = function()
      dlg:modify{ id = "pixel_size", label = "Target (" .. dlg.data.pixel_size .. "px)" }
    end,
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
    onchange = function()
      local m = dlg.data.palette_mode
      dlg:modify{ id = "palette_name", visible = (m == "preset") }
      dlg:modify{ id = "palette_custom_colors", visible = (m == "custom") }
    end,
  }

  dlg:combobox{
    id = "palette_name",
    label = "Preset",
    options = { "pico8" },
    option = "pico8",
    visible = false,
  }

  dlg:entry{
    id = "palette_custom_colors",
    label = "Custom Hex",
    text = "",
    visible = false,
    hexpand = true,
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
    onchange = function()
      local ad = dlg.data.anim_method == "animatediff"
      dlg:modify{ id = "anim_freeinit", visible = ad }
      dlg:modify{ id = "anim_freeinit_iters", visible = ad }
    end,
  }

  dlg:slider{
    id = "anim_frames",
    label = "Frames",
    min = 2, max = 120, value = 8,
  }

  dlg:slider{
    id = "anim_duration",
    label = "Duration (ms)",
    min = 50, max = 2000, value = 100,
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
    visible = false,
  }

  dlg:slider{
    id = "anim_freeinit_iters",
    label = "FreeInit Iters",
    min = 1, max = 3, value = 2,
    visible = false,
  }
end

-- ─── Tab: Live ──────────────────────────────────────────────

local function build_tab_live()
  local dlg = PT.dlg

  -- Trigger mode selector
  dlg:combobox{
    id = "live_mode",
    label = "Trigger",
    options = { "Auto (stroke)", "Manual (F5)" },
    option = "Auto (stroke)",
    onchange = function()
      PT.live.auto_mode = (dlg.data.live_mode == "Auto (stroke)")
      if PT.live.mode and PT.dlg then
        if PT.live.auto_mode then
          PT.update_status("Live — auto mode (sends after each stroke)")
        else
          PT.update_status("Live — manual mode (press F5 to send)")
        end
      end
    end,
  }

  -- Batched slider update: sends all changed params once after debounce
  local function schedule_live_slider_update()
    if not PT.live.mode then return end
    PT.live.slider_debounce = PT.stop_timer(PT.live.slider_debounce)
    PT.live.slider_debounce = Timer{
      interval = PT.cfg.LIVE_SLIDER_DEBOUNCE,
      ontick = function()
        PT.live.slider_debounce = PT.stop_timer(PT.live.slider_debounce)
        if not PT.live.mode or not PT.dlg then return end
        PT.send({
          action = "realtime_update",
          denoise_strength = PT.dlg.data.live_strength / 100.0,
          steps = PT.dlg.data.live_steps,
          cfg_scale = PT.dlg.data.live_cfg / 10.0,
        })
      end,
    }
    PT.live.slider_debounce:start()
  end

  dlg:slider{
    id = "live_strength",
    label = "Strength (0.50)",
    min = 5, max = 95, value = 50,
    onchange = function()
      slider_label("live_strength", "Strength (%.2f)", 100.0)()
      schedule_live_slider_update()
    end,
  }

  dlg:slider{
    id = "live_steps",
    label = "Steps",
    min = 2, max = 8, value = 4,
    onchange = function()
      schedule_live_slider_update()
    end,
  }

  dlg:slider{
    id = "live_cfg",
    label = "CFG (2.5)",
    min = 10, max = 100, value = 25,
    onchange = function()
      slider_label("live_cfg", "CFG (%.1f)", 10.0)()
      schedule_live_slider_update()
    end,
  }

  dlg:slider{
    id = "live_opacity",
    label = "Preview (70%)",
    min = 10, max = 100, value = 70,
    onchange = function()
      dlg:modify{ id = "live_opacity",
        label = string.format("Preview (%d%%)", dlg.data.live_opacity) }
      if PT.live.preview_layer then
        PT.live.preview_layer.opacity = math.floor(dlg.data.live_opacity * 255 / 100)
        app.refresh()
      end
    end,
  }
end

-- ─── Tab: Audio ───────────────────────────────────────────

local GLOBAL_SOURCES = {
  "global_rms", "global_onset", "global_centroid",
  "global_low", "global_mid", "global_high",
  "global_beat",
}

local MOD_TARGETS = {
  "denoise_strength", "cfg_scale", "noise_amplitude",
  "controlnet_scale", "seed_offset",
}

-- Syncs slot widget visibility based on slot count and advanced toggle
local function sync_slot_visibility()
  local dlg = PT.dlg
  if not dlg then return end
  local n = dlg.data.mod_slot_count
  local adv = dlg.data.audio_advanced
  for i = 1, 4 do
    local vis = (i <= n)
    dlg:modify{ id = "mod" .. i .. "_enable",  visible = vis }
    dlg:modify{ id = "mod" .. i .. "_source",  visible = vis }
    dlg:modify{ id = "mod" .. i .. "_target",  visible = vis }
    dlg:modify{ id = "mod" .. i .. "_min",     visible = vis }
    dlg:modify{ id = "mod" .. i .. "_max",     visible = vis }
    dlg:modify{ id = "mod" .. i .. "_attack",  visible = vis and adv }
    dlg:modify{ id = "mod" .. i .. "_release", visible = vis and adv }
  end
  dlg:modify{ id = "audio_use_expressions", visible = adv }
  dlg:modify{ id = "expr_denoise", visible = adv and dlg.data.audio_use_expressions }
  dlg:modify{ id = "expr_cfg",     visible = adv and dlg.data.audio_use_expressions }
  dlg:modify{ id = "expr_noise",   visible = adv and dlg.data.audio_use_expressions }
  dlg:modify{ id = "audio_random_seed", visible = adv }
  -- Prompt schedule visibility
  dlg:modify{ id = "audio_prompt_schedule", visible = adv }
  local ps_vis = adv and dlg.data.audio_prompt_schedule
  for i = 1, 3 do
    dlg:modify{ id = "ps" .. i .. "_time", visible = ps_vis }
    dlg:modify{ id = "ps" .. i .. "_prompt", visible = ps_vis }
  end
end

local function build_tab_audio()
  local dlg = PT.dlg

  -- File
  dlg:file{
    id = "audio_file",
    label = "File",
    filetypes = { "wav", "mp3", "flac", "ogg" },
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
    id = "audio_frame_duration",
    label = "Frame (42ms)",
    min = 30, max = 200, value = 42,
    onchange = function()
      dlg:modify{ id = "audio_frame_duration",
        label = "Frame (" .. dlg.data.audio_frame_duration .. "ms)" }
    end,
  }

  -- Modulation
  dlg:separator{ text = "Modulation" }
  dlg:combobox{
    id = "audio_mod_preset",
    label = "Preset",
    options = { "(custom)", "energetic", "ambient", "bass_driven" },
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
    label = "Slots",
    min = 1, max = 4, value = 1,
    onchange = function() sync_slot_visibility() end,
  }

  -- Slot defaults: [source, target, min, max, attack, release, visible]
  local slot_defaults = {
    { "global_rms",    "denoise_strength",  15, 65, 2, 8,  true },
    { "global_onset",  "cfg_scale",         30, 80, 2, 8,  false },
    { "global_low",    "noise_amplitude",    0, 30, 2, 8,  false },
    { "global_high",   "seed_offset",        0, 50, 2, 8,  false },
  }

  for i, def in ipairs(slot_defaults) do
    local prefix = "mod" .. i .. "_"
    local vis = def[7]

    dlg:check{
      id = prefix .. "enable",
      text = "Slot " .. i,
      selected = true,
      visible = vis,
    }
    dlg:combobox{
      id = prefix .. "source",
      label = "Source",
      options = GLOBAL_SOURCES,
      option = def[1],
      visible = vis,
    }
    dlg:combobox{
      id = prefix .. "target",
      label = "Target",
      options = MOD_TARGETS,
      option = def[2],
      visible = vis,
    }
    dlg:slider{
      id = prefix .. "min",
      label = "Min (%)",
      min = 0, max = 100, value = def[3],
      visible = vis,
    }
    dlg:slider{
      id = prefix .. "max",
      label = "Max (%)",
      min = 0, max = 100, value = def[4],
      visible = vis,
    }
    dlg:slider{
      id = prefix .. "attack",
      label = "Attack",
      min = 1, max = 30, value = def[5],
      visible = false,
    }
    dlg:slider{
      id = prefix .. "release",
      label = "Release",
      min = 1, max = 60, value = def[6],
      visible = false,
    }
  end

  -- Advanced toggle (controls attack/release + expressions)
  dlg:check{
    id = "audio_advanced",
    text = "Advanced",
    selected = false,
    onchange = function() sync_slot_visibility() end,
  }

  dlg:check{
    id = "audio_use_expressions",
    text = "Custom Expressions",
    selected = false,
    visible = false,
    onchange = function() sync_slot_visibility() end,
  }
  dlg:entry{
    id = "expr_denoise",
    label = "denoise",
    text = "",
    visible = false,
    hexpand = true,
  }
  dlg:entry{
    id = "expr_cfg",
    label = "cfg_scale",
    text = "",
    visible = false,
    hexpand = true,
  }
  dlg:entry{
    id = "expr_noise",
    label = "noise_amp",
    text = "",
    visible = false,
    hexpand = true,
  }

  dlg:check{
    id = "audio_random_seed",
    text = "Random seed per frame",
    selected = false,
    visible = false,
  }

  -- Prompt Schedule (advanced only)
  dlg:check{
    id = "audio_prompt_schedule",
    text = "Prompt Schedule",
    selected = false,
    visible = false,
    onchange = function()
      local vis = dlg.data.audio_prompt_schedule and dlg.data.audio_advanced
      for i = 1, 3 do
        dlg:modify{ id = "ps" .. i .. "_time", visible = vis }
        dlg:modify{ id = "ps" .. i .. "_prompt", visible = vis }
      end
    end,
  }
  for i = 1, 3 do
    dlg:entry{
      id = "ps" .. i .. "_time",
      label = "T" .. i .. " (s-s)",
      text = "",
      visible = false,
      hexpand = true,
    }
    dlg:entry{
      id = "ps" .. i .. "_prompt",
      label = "P" .. i,
      text = "",
      visible = false,
      hexpand = true,
    }
  end

  dlg:button{
    id = "audio_generate_btn",
    text = "GENERATE AUDIO",
    enabled = false,
    hexpand = true,
    onclick = function()
      if PT.state.generating or PT.state.animating or PT.audio.generating then return end
      local path = dlg.data.audio_file
      if not path or path == "" then
        app.alert("Select an audio file first.")
        return
      end
      if not PT.audio.analyzed then
        app.alert("Analyze the audio file first.")
        return
      end

      local req = PT.build_audio_reactive_request()
      if not PT.attach_source_image(req) then return end

      PT.audio.generating = true
      PT.state.animating = true
      PT.state.gen_step_start = os.clock()
      -- Dynamic timeout: 180s base + 15s per expected frame
      local audio_timeout = 180 + (PT.audio.total_frames * 15)
      PT.start_gen_timeout(math.max(PT.cfg.GEN_TIMEOUT, audio_timeout))
      dlg:modify{ id = "audio_generate_btn", enabled = false }
      dlg:modify{ id = "generate_btn", enabled = false }
      dlg:modify{ id = "animate_btn", enabled = false }
      dlg:modify{ id = "live_btn", enabled = false }
      dlg:modify{ id = "cancel_btn", enabled = true }
      PT.update_status("Generating audio animation...")
      PT.send(req)
    end,
  }
end

-- ─── Actions Panel ──────────────────────────────────────────

local function build_actions_panel()
  local dlg = PT.dlg

  dlg:separator{ text = "Actions", hexpand = true }

  dlg:check{
    id = "loop_check",
    label = "Loop Mode",
    selected = false,
    onchange = function()
      dlg:modify{ id = "loop_seed_combo", visible = dlg.data.loop_check or dlg.data.random_loop_check }
    end,
  }
  dlg:check{
    id = "random_loop_check",
    label = "Random Loop",
    selected = false,
    onchange = function()
      dlg:modify{ id = "loop_seed_combo", visible = dlg.data.loop_check or dlg.data.random_loop_check }
    end,
  }
  dlg:combobox{
    id = "loop_seed_combo",
    label = "Loop Seed",
    options = { "random", "increment" },
    option = "random",
    visible = false,
  }

  dlg:button{
    id = "generate_btn",
    text = "GENERATE",
    enabled = false,
    hexpand = true,
    onclick = function()
      if PT.state.generating or PT.state.animating then return end
      -- Reset sequence for non-loop single gen (new sequence each click)
      local is_loop = dlg.data.loop_check or dlg.data.random_loop_check
      if not is_loop then
        PT.finalize_sequence()
      end
      -- Initialize loop state
      if is_loop then
        PT.loop.mode = true
        PT.loop.counter = 0
        PT.loop.seed_mode = dlg.data.loop_seed_combo or "random"
        PT.loop.random_mode = dlg.data.random_loop_check or false
        -- Build locked_fields from UI
        PT.loop.locked_fields = {}
        if dlg.data.lock_subject and dlg.data.fixed_subject ~= "" then
          PT.loop.locked_fields.subject = dlg.data.fixed_subject
        end
      end

      -- Random loop: first generate a random prompt, then generate image
      if PT.loop.random_mode then
        dlg:modify{ id = "generate_btn", text = "LOOPING...", enabled = false }
        dlg:modify{ id = "animate_btn", enabled = false }
        dlg:modify{ id = "live_btn", enabled = false }
        dlg:modify{ id = "cancel_btn", enabled = true }
        PT.loop.counter = PT.loop.counter + 1
        PT.update_status("Random Loop #" .. PT.loop.counter .. " — Generating prompt...")
        PT.send({ action = "generate_prompt", locked_fields = PT.loop.locked_fields })
        return
      end

      local req = PT.build_generate_request()
      if not PT.attach_source_image(req) then PT.loop.mode = false; PT.loop.random_mode = false; return end

      PT.state.generating = true
      PT.state.gen_step_start = os.clock()
      PT.start_gen_timeout()
      dlg:modify{ id = "generate_btn", text = PT.loop.mode and "LOOPING..." or "GENERATE", enabled = false }
      dlg:modify{ id = "animate_btn", enabled = false }
      dlg:modify{ id = "live_btn", enabled = false }
      dlg:modify{ id = "cancel_btn", enabled = true }
      if PT.loop.mode then
        PT.loop.counter = PT.loop.counter + 1
        PT.update_status("Loop #" .. PT.loop.counter .. " — Generating...")
      else
        PT.update_status("Generating...")
      end
      PT.send(req)
    end,
  }

  dlg:button{
    id = "cancel_btn",
    text = "CANCEL",
    enabled = false,
    hexpand = true,
    onclick = function()
      PT.loop.mode = false
      PT.loop.random_mode = false
      PT.timers.loop = PT.stop_timer(PT.timers.loop)
      if PT.state.generating or PT.state.animating then
        PT.send({ action = "cancel" })
        PT.state.cancel_pending = true
        dlg:modify{ id = "generate_btn", enabled = false }
        PT.update_status("Cancelling...")
        -- Re-enable audio analyze button if audio was active
        if PT.audio.generating or PT.audio.analyzing then
          dlg:modify{ id = "audio_analyze_btn", enabled = PT.state.connected }
        end
        -- Safety timer: force UI unlock if server never responds
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
              PT.finalize_sequence()
              PT.update_status("Cancel timeout — UI reset")
              PT.reset_ui_buttons()
            end
          end,
        }
        PT.timers.cancel_safety:start()
      else
        -- Cancel random loop even if no generation is in flight yet
        PT.finalize_sequence()
        PT.reset_ui_buttons()
        PT.update_status("Cancelled")
      end
    end,
  }

  dlg:button{
    id = "animate_btn",
    text = "ANIMATE",
    enabled = false,
    hexpand = true,
    onclick = function()
      if PT.state.animating or PT.state.generating then return end
      local gw, gh = PT.parse_size()
      local tag_name = dlg.data.anim_tag or ""
      if tag_name == "" then tag_name = nil end

      local req = {
        action = "generate_animation",
        method = dlg.data.anim_method,
        prompt = dlg.data.prompt,
        negative_prompt = dlg.data.negative_prompt,
        mode = dlg.data.mode,
        width = gw, height = gh,
        seed = PT.parse_seed(),
        steps = dlg.data.steps,
        cfg_scale = dlg.data.cfg_scale / 10.0,
        clip_skip = dlg.data.clip_skip,
        denoise_strength = dlg.data.anim_denoise / 100.0,
        frame_count = dlg.data.anim_frames,
        frame_duration_ms = dlg.data.anim_duration,
        seed_strategy = dlg.data.anim_seed_strategy,
        tag_name = tag_name,
        enable_freeinit = dlg.data.anim_freeinit,
        freeinit_iterations = dlg.data.anim_freeinit_iters,
        post_process = PT.build_post_process(),
      }
      PT.attach_lora(req)
      PT.attach_neg_ti(req)
      if not PT.attach_source_image(req) then return end

      PT.state.animating = true
      PT.state.gen_step_start = os.clock()
      PT.start_gen_timeout()
      dlg:modify{ id = "animate_btn", enabled = false }
      dlg:modify{ id = "generate_btn", enabled = false }
      dlg:modify{ id = "live_btn", enabled = false }
      dlg:modify{ id = "cancel_btn", enabled = true }
      PT.update_status("Animating...")
      PT.send(req)
    end,
  }

  dlg:button{
    id = "live_btn",
    text = "START LIVE",
    enabled = false,
    hexpand = true,
    onclick = function()
      if PT.live.mode then
        PT.send({ action = "realtime_stop" })
        PT.stop_live_mode()
      else
        if PT.state.generating or PT.state.animating then return end
        local spr = app.sprite
        if spr == nil then
          app.alert("Open a sprite first to use Live mode.")
          return
        end
        local gw, gh = PT.parse_size()
        local req = {
          action = "realtime_start",
          prompt = dlg.data.prompt,
          negative_prompt = dlg.data.negative_prompt,
          width = gw, height = gh,
          seed = PT.parse_seed(),
          steps = dlg.data.live_steps,
          cfg_scale = dlg.data.live_cfg / 10.0,
          denoise_strength = dlg.data.live_strength / 100.0,
          clip_skip = dlg.data.clip_skip,
          post_process = PT.build_post_process(),
        }
        PT.attach_lora(req)
        PT.attach_neg_ti(req)
        PT.live.frame_id = 0
        PT.live.auto_mode = (dlg.data.live_mode == "Auto (stroke)")
        PT.update_status("Starting live...")
        PT.send(req)
      end
    end,
  }

  dlg:button{
    id = "live_send_btn",
    text = "SEND (F5)",
    visible = false,
    hexpand = true,
    onclick = function()
      PT.live_send_now()
    end,
  }

  dlg:button{
    id = "live_accept_btn",
    text = "ACCEPT",
    visible = false,
    onclick = function()
      local spr = app.sprite
      if spr == nil or PT.live.preview_layer == nil then return end
      local ok_cel, cel = pcall(function() return PT.live.preview_layer:cel(app.frame) end)
      if not ok_cel or cel == nil or cel.image == nil then return end
      PT.live.importing = true
      app.transaction("PixyToon Accept Live", function()
        local new_layer = spr:newLayer()
        new_layer.name = "PixyToon Live"
        spr:newCel(new_layer, app.frame, cel.image:clone(), cel.position)
        pcall(function() spr:deleteCel(cel) end)
      end)
      PT.live.importing = false
      PT.live.prev_canvas = nil
      app.refresh()
      PT.update_status("Live result accepted")
    end,
  }
end

-- ─── Main Build ─────────────────────────────────────────────

function PT.build_dialog()
  PT.dlg = Dialog{
    title = "PixyToon - SD Pixel Art",
    onclose = function()
      PT.save_settings()
      PT.disconnect()
      PT.dlg = nil
    end,
  }

  build_connection_section()

  PT.dlg:tab{ id = "tab_gen", text = "Generate" }
  build_tab_generate()

  PT.dlg:tab{ id = "tab_pp", text = "Post-Process" }
  build_tab_postprocess()

  PT.dlg:tab{ id = "tab_anim", text = "Animation" }
  build_tab_animation()

  PT.dlg:tab{ id = "tab_live", text = "Live" }
  build_tab_live()

  PT.dlg:tab{ id = "tab_audio", text = "Audio" }
  build_tab_audio()

  PT.dlg:endtabs{ id = "main_tabs", selected = "tab_gen" }

  build_actions_panel()

  PT.dlg:show{ wait = false, autoscrollbars = true }
end

end
