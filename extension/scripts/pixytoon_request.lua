--
-- PixyToon — Request Builders
--

return function(PT)

function PT.parse_size()
  local s = PT.dlg.data.output_size
  local w, h = s:match("(%d+)x(%d+)")
  return tonumber(w) or 512, tonumber(h) or 512
end

function PT.parse_seed()
  local v = tonumber(PT.dlg.data.seed) or -1
  if v ~= math.floor(v) then v = -1 end
  return v
end

function PT.attach_lora(req)
  local sel = PT.dlg.data.lora_name
  if sel and sel ~= "(default)" then
    req.lora = { name = sel, weight = PT.dlg.data.lora_weight / 100.0 }
  end
end

function PT.attach_neg_ti(req)
  if PT.dlg.data.use_neg_ti and #PT.res.embeddings > 0 then
    local ti_list = {}
    local w = PT.dlg.data.neg_ti_weight / 100.0
    for _, name in ipairs(PT.res.embeddings) do
      ti_list[#ti_list + 1] = { name = name, weight = w }
    end
    req.negative_ti = ti_list
  end
end

function PT.attach_source_image(req)
  local mode = req.mode or "txt2img"
  if mode == "img2img" or mode:find("controlnet_") then
    local b64 = PT.capture_active_layer()
    if not b64 then
      app.alert("No active layer to use as source.")
      return false
    end
    if mode == "img2img" then req.source_image = b64
    else req.control_image = b64 end
  end
  if mode == "inpaint" then
    local src = PT.capture_flattened()
    if not src then
      app.alert("Inpaint requires an open sprite.")
      return false
    end
    req.source_image = src
    local mask = PT.capture_mask()
    if not mask then
      app.alert("Inpaint requires a mask.\n- Make a selection, or\n- Create a 'Mask' layer, or\n- Draw on active layer")
      return false
    end
    req.mask_image = mask
  end
  return true
end

function PT.build_post_process()
  local d = PT.dlg.data
  local pp = {
    pixelate = {
      enabled = d.pixelate,
      target_size = d.pixel_size,
    },
    quantize_method = d.quantize_method,
    quantize_colors = d.colors,
    dither = d.dither,
    palette = { mode = d.palette_mode },
    remove_bg = d.remove_bg,
  }
  if d.palette_mode == "preset" then
    pp.palette.name = d.palette_name
  elseif d.palette_mode == "custom" then
    local hex_str = d.palette_custom_colors or ""
    local colors = {}
    for hex in hex_str:gmatch("#?(%x%x%x%x%x%x)") do
      colors[#colors + 1] = "#" .. hex
    end
    if #colors > 0 then pp.palette.colors = colors end
  end
  return pp
end

-- ─── Audio Requests ───────────────────────────────────────

function PT.build_analyze_audio_request()
  local d = PT.dlg.data
  return {
    action      = "analyze_audio",
    audio_path  = d.audio_file,
    fps         = tonumber(d.audio_fps) or 24,
    enable_stems = d.audio_stems_enable or false,
  }
end

function PT.build_audio_reactive_request()
  local d = PT.dlg.data
  local gw, gh = PT.parse_size()
  local tag_name = d.anim_tag or ""
  if tag_name == "" then tag_name = nil end

  -- Build modulation slots from UI (4 slots)
  local slots = {}
  local slot_count = d.mod_slot_count or 2
  for i = 1, slot_count do
    local prefix = "mod" .. i .. "_"
    if d[prefix .. "enable"] then
      local target = d[prefix .. "target"]
      local mn = d[prefix .. "min"] / 100.0
      local mx = d[prefix .. "max"] / 100.0
      -- Scale to target range
      if target == "cfg_scale" then
        mn = mn * 30.0
        mx = mx * 30.0
      elseif target == "seed_offset" then
        mn = math.floor(mn * 1000)
        mx = math.floor(mx * 1000)
      elseif target == "controlnet_scale" then
        mn = mn * 2.0
        mx = mx * 2.0
      end
      slots[#slots + 1] = {
        source  = d[prefix .. "source"],
        target  = target,
        min_val = mn,
        max_val = mx,
        attack  = d[prefix .. "attack"],
        release = d[prefix .. "release"],
        enabled = true,
      }
    end
  end

  -- Build expressions if enabled
  local expressions = nil
  if d.audio_use_expressions then
    expressions = {}
    local expr_fields = { "expr_denoise", "expr_cfg", "expr_noise" }
    local expr_targets = { "denoise_strength", "cfg_scale", "noise_amplitude" }
    for idx, field in ipairs(expr_fields) do
      local val = d[field] or ""
      if val ~= "" then
        expressions[expr_targets[idx]] = val
      end
    end
    if next(expressions) == nil then expressions = nil end
  end

  -- Modulation preset
  local mod_preset = nil
  local preset_sel = d.audio_mod_preset
  if preset_sel and preset_sel ~= "(custom)" then
    mod_preset = preset_sel
  end

  local req = {
    action            = "generate_audio_reactive",
    audio_path        = d.audio_file,
    fps               = tonumber(d.audio_fps) or 24,
    enable_stems      = d.audio_stems_enable or false,
    modulation_slots  = slots,
    expressions       = expressions,
    modulation_preset = mod_preset,
    prompt            = d.prompt,
    negative_prompt   = d.negative_prompt,
    mode              = d.mode,
    width             = gw,
    height            = gh,
    seed              = PT.parse_seed(),
    steps             = d.steps,
    cfg_scale         = d.cfg_scale / 10.0,
    clip_skip         = d.clip_skip,
    denoise_strength  = d.denoise / 100.0,
    frame_duration_ms = d.audio_frame_duration,
    tag_name          = tag_name,
    post_process      = PT.build_post_process(),
  }
  PT.attach_lora(req)
  PT.attach_neg_ti(req)
  return req
end

-- Factored from generate button onclick + loop continuation (eliminates duplication).
function PT.build_generate_request()
  local gw, gh = PT.parse_size()
  local req = {
    action           = "generate",
    prompt           = PT.dlg.data.prompt,
    negative_prompt  = PT.dlg.data.negative_prompt,
    mode             = PT.dlg.data.mode,
    width            = gw,
    height           = gh,
    seed             = PT.parse_seed(),
    steps            = PT.dlg.data.steps,
    cfg_scale        = PT.dlg.data.cfg_scale / 10.0,
    clip_skip        = PT.dlg.data.clip_skip,
    denoise_strength = PT.dlg.data.denoise / 100.0,
    post_process     = PT.build_post_process(),
  }
  PT.attach_lora(req)
  PT.attach_neg_ti(req)
  return req
end

end
