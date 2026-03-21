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
      -- Initialize loop state
      local is_loop = dlg.data.loop_check or dlg.data.random_loop_check
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
      if PT.state.generating or PT.state.animating then
        PT.send({ action = "cancel" })
        PT.state.cancel_pending = true
        dlg:modify{ id = "generate_btn", enabled = false }
        PT.update_status("Cancelling...")
      else
        -- Cancel random loop even if no generation is in flight yet
        dlg:modify{ id = "generate_btn", text = "GENERATE", enabled = true }
        dlg:modify{ id = "animate_btn", enabled = true }
        dlg:modify{ id = "live_btn", enabled = true }
        dlg:modify{ id = "cancel_btn", enabled = false }
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
        PT.stop_live_timer()
        PT.live.mode = false
        PT.update_status("Stopping live...")
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
      app.transaction("PixyToon Accept Live", function()
        local new_layer = spr:newLayer()
        new_layer.name = "PixyToon Live"
        spr:newCel(new_layer, app.frame, cel.image:clone(), cel.position)
        pcall(function() spr:deleteCel(cel) end)
      end)
      PT.live.prev_canvas = nil
      app.refresh()
      PT.update_status("Live result accepted")
    end,
  }
end

-- ─── Main Build ─────────────────────────────────────────────

function PT.build_dialog()
  PT.dlg = Dialog{
    title = "PixyToon - AI Pixel Art",
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

  PT.dlg:endtabs{ id = "main_tabs", selected = "tab_gen" }

  build_actions_panel()

  PT.dlg:show{ wait = false, autoscrollbars = true }
end

end
