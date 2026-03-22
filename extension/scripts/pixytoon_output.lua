--
-- PixyToon — Output Directory & Metadata Persistence
--

return function(PT)

-- ─── Project Root Detection ───────────────────────────────────
-- Derive <pixytoon_root> from this script's absolute path.
-- Script is at <root>/extension/scripts/pixytoon_output.lua → go up 3 levels.

local _raw_source = debug.getinfo(1, "S").source
local _script_path = _raw_source:sub(1, 1) == "@" and _raw_source:sub(2) or _raw_source
local _scripts_dir = app.fs.filePath(_script_path)
local _ext_dir     = app.fs.filePath(_scripts_dir)
local _project_root = app.fs.filePath(_ext_dir)

PT.cfg.PROJECT_ROOT = _project_root
PT.cfg.OUTPUT_DIR   = app.fs.joinPath(_project_root, "output")

-- Monotonic counter to prevent same-second filename collisions (loop mode).
local _output_counter = 0

-- ─── Directory Helpers ────────────────────────────────────────

function PT.get_output_root()
  return PT.cfg.OUTPUT_DIR
end

function PT.ensure_date_dir()
  local root = PT.get_output_root()
  if not app.fs.isDirectory(root) then
    if not app.fs.makeDirectory(root) then return nil end
  end
  local date_str = os.date("%Y-%m-%d")
  local date_dir = app.fs.joinPath(root, date_str)
  if not app.fs.isDirectory(date_dir) then
    if not app.fs.makeDirectory(date_dir) then return nil end
  end
  return date_dir
end

-- ─── Single Generation Save ──────────────────────────────────

function PT.save_to_output(resp, meta)
  if not PT.output.enabled then return end
  if not resp or not resp.image then return end

  local ok, err = pcall(function()
    local date_dir = PT.ensure_date_dir()
    if not date_dir then return end
    local time_str = os.date("%H-%M-%S")
    local seed_str = tostring(resp.seed or 0)
    _output_counter = _output_counter + 1
    local base_name = time_str .. "_" .. string.format("%03d", _output_counter) .. "_s" .. seed_str

    -- Decode and validate
    local img_data = PT.base64_decode(resp.image)
    if not img_data or #img_data == 0 then return end

    -- Save PNG
    local png_path = app.fs.joinPath(date_dir, base_name .. ".png")
    local f = io.open(png_path, "wb")
    if f then
      f:write(img_data)
      f:close()
    end

    -- Save metadata JSON
    if meta then
      meta.output_file = base_name .. ".png"
      local json_path = app.fs.joinPath(date_dir, base_name .. ".json")
      local jf = io.open(json_path, "w")
      if jf then
        jf:write(PT.json.encode(meta))
        jf:close()
      end
    end
  end)
  if not ok then
    PT.update_status("Output save error: " .. tostring(err))
  end
end

-- ─── Animation/Audio Incremental Save ───────────────────────────
-- Frames are written to disk one at a time as they arrive (no memory accumulation).
-- Call save_animation_frame() per frame, then save_animation_meta() on complete.

function PT.save_animation_frame(resp)
  if not PT.output.enabled then return end
  if not PT.dlg or not PT.dlg.data.save_output then return end
  if not resp.image or resp.frame_index == nil then return end

  pcall(function()
    -- Create output dir on first frame
    if not PT.anim.output_dir then
      local date_dir = PT.ensure_date_dir()
      if not date_dir then return end
      local time_str = os.date("%H-%M-%S")
      local req = PT.last_request
      local tag = (req and req.tag_name and req.tag_name ~= "") and req.tag_name or "untitled"
      tag = tag:gsub("[^%w%-_]", "_"):sub(1, 50)
      local action_prefix = (req and req.action == "generate_audio_reactive") and "audio" or "anim"
      _output_counter = _output_counter + 1
      local folder_name = time_str .. "_" .. string.format("%03d", _output_counter) .. "_" .. action_prefix .. "_" .. tag
      PT.anim.output_dir = app.fs.joinPath(date_dir, folder_name)
      app.fs.makeDirectory(PT.anim.output_dir)
      PT.anim.output_count = 0
    end

    -- Write frame to disk immediately
    local img_data = PT.base64_decode(resp.image)
    if img_data and #img_data > 0 then
      local frame_name = string.format("frame_%03d.png", resp.frame_index + 1)
      local frame_path = app.fs.joinPath(PT.anim.output_dir, frame_name)
      local f = io.open(frame_path, "wb")
      if f then
        f:write(img_data)
        f:close()
        PT.anim.output_count = PT.anim.output_count + 1
      end
    end
  end)
end

function PT.save_animation_meta(resp)
  if not PT.anim.output_dir then return end
  if PT.anim.output_count == 0 then return end

  local ok, err = pcall(function()
    local meta = PT.build_animation_meta(resp)
    PT.last_result_meta = meta
    meta.output_folder = app.fs.fileName(PT.anim.output_dir)
    meta.frame_count = PT.anim.output_count
    local json_path = app.fs.joinPath(PT.anim.output_dir, "metadata.json")
    local jf = io.open(json_path, "w")
    if jf then
      jf:write(PT.json.encode(meta))
      jf:close()
    end
  end)
  if not ok then
    PT.update_status("Animation meta save error: " .. tostring(err))
  end
end

-- ─── Open Output Directory ────────────────────────────────────

function PT.open_output_dir()
  local dir = PT.get_output_root()
  if not app.fs.isDirectory(dir) then
    app.fs.makeDirectory(dir)
  end
  -- Sanitize path: strip any quotes to prevent command injection
  local safe_dir = dir:gsub("/", "\\"):gsub('"', "")
  os.execute('explorer "' .. safe_dir .. '"')
end

-- ─── Load Metadata from JSON ──────────────────────────────────

function PT.load_metadata_file(path)
  local f = io.open(path, "r")
  if not f then return nil, "Cannot open file" end
  local data = f:read("*a")
  f:close()
  local ok, meta = pcall(PT.json.decode, data)
  if not ok or type(meta) ~= "table" then
    return nil, "Invalid JSON"
  end
  return meta
end

function PT.apply_metadata(meta)
  if not meta or not PT.dlg then return end
  -- Text fields
  if meta.prompt then PT.dlg:modify{ id = "prompt", text = meta.prompt } end
  if meta.negative_prompt then PT.dlg:modify{ id = "negative_prompt", text = meta.negative_prompt } end
  if meta.seed then PT.dlg:modify{ id = "seed", text = tostring(meta.seed) } end

  -- Combobox fields
  if meta.mode then PT.dlg:modify{ id = "mode", option = meta.mode } end
  if meta.output_size then PT.dlg:modify{ id = "output_size", option = meta.output_size } end

  -- Slider fields (requires inverse scaling)
  if meta.steps then PT.dlg:modify{ id = "steps", value = meta.steps } end
  if meta.cfg_scale then
    local v = math.floor(meta.cfg_scale * 10)
    PT.dlg:modify{ id = "cfg_scale", value = v }
    PT.dlg:modify{ id = "cfg_scale", label = string.format("CFG (%.1f)", v / 10.0) }
  end
  if meta.clip_skip then PT.dlg:modify{ id = "clip_skip", value = meta.clip_skip } end
  if meta.denoise_strength then
    local v = math.floor(meta.denoise_strength * 100)
    PT.dlg:modify{ id = "denoise", value = v }
    PT.dlg:modify{ id = "denoise", label = string.format("Strength (%.2f)", v / 100.0) }
  end

  -- Mode-dependent visibility + label (mirror dialog onchange logic)
  if meta.mode then
    local m = meta.mode
    local is_txt = (m == "txt2img")
    PT.dlg:modify{ id = "denoise", visible = not is_txt }
    if m == "inpaint" then
      PT.dlg:modify{ id = "mode", label = "Mode (needs mask)" }
    elseif m == "img2img" or (m:find("controlnet_") ~= nil) then
      PT.dlg:modify{ id = "mode", label = "Mode (needs layer)" }
    else
      PT.dlg:modify{ id = "mode", label = "Mode" }
    end
  end

  -- LoRA
  if meta.lora and meta.lora.name then
    PT.dlg:modify{ id = "lora_name", option = meta.lora.name }
    if meta.lora.weight then
      local v = math.floor(meta.lora.weight * 100)
      PT.dlg:modify{ id = "lora_weight", value = v }
      PT.dlg:modify{ id = "lora_weight", label = string.format("LoRA (%.2f)", v / 100.0) }
    end
  end

  -- Post-process
  if meta.post_process then
    local pp = meta.post_process
    if pp.pixelate ~= nil then
      if type(pp.pixelate) == "table" then
        if pp.pixelate.enabled ~= nil then PT.dlg:modify{ id = "pixelate", selected = pp.pixelate.enabled } end
        if pp.pixelate.target_size then
          PT.dlg:modify{ id = "pixel_size", value = pp.pixelate.target_size }
          PT.dlg:modify{ id = "pixel_size", label = "Target (" .. pp.pixelate.target_size .. "px)" }
          PT.dlg:modify{ id = "pixel_size", visible = pp.pixelate.enabled }
        end
      end
    end
    if pp.quantize_colors then PT.dlg:modify{ id = "colors", value = pp.quantize_colors } end
    if pp.quantize_method then PT.dlg:modify{ id = "quantize_method", option = pp.quantize_method } end
    if pp.dither then PT.dlg:modify{ id = "dither", option = pp.dither } end
    if pp.remove_bg ~= nil then PT.dlg:modify{ id = "remove_bg", selected = pp.remove_bg } end
    if pp.palette then
      if pp.palette.mode then PT.dlg:modify{ id = "palette_mode", option = pp.palette.mode } end
      if pp.palette.name then PT.dlg:modify{ id = "palette_name", option = pp.palette.name } end
    end
  end

  -- Animation-specific fields
  if meta.method then PT.dlg:modify{ id = "anim_method", option = meta.method } end
  if meta.frame_count then PT.dlg:modify{ id = "anim_frames", value = meta.frame_count } end
  if meta.seed_strategy then PT.dlg:modify{ id = "anim_seed_strategy", option = meta.seed_strategy } end
  if meta.tag_name then PT.dlg:modify{ id = "anim_tag", text = meta.tag_name } end

  PT.update_status("Metadata loaded (seed=" .. tostring(meta.seed or "?") .. ")")
end

-- ─── Build Metadata from Request + Response ────────────────────

function PT.build_generation_meta(resp)
  local req = PT.last_request
  local meta = {
    pixytoon_version = "0.7.3",
    type = "generation",
    timestamp = os.date("!%Y-%m-%dT%H:%M:%S"),
    timestamp_local = os.date("%Y-%m-%d %H:%M:%S"),
    seed = resp.seed,
    time_ms = resp.time_ms,
    width = resp.width,
    height = resp.height,
  }
  if req then
    meta.prompt = req.prompt
    meta.negative_prompt = req.negative_prompt
    meta.mode = req.mode
    meta.steps = req.steps
    meta.cfg_scale = req.cfg_scale
    meta.clip_skip = req.clip_skip
    meta.denoise_strength = req.denoise_strength
    meta.lora = req.lora
    meta.negative_ti = req.negative_ti
    meta.post_process = req.post_process
    meta.output_size = tostring(req.width or 512) .. "x" .. tostring(req.height or 512)
  end
  return meta
end

function PT.build_animation_meta(resp)
  local req = PT.last_request
  local meta = {
    pixytoon_version = "0.7.3",
    type = "animation",
    timestamp = os.date("!%Y-%m-%dT%H:%M:%S"),
    timestamp_local = os.date("%Y-%m-%d %H:%M:%S"),
    total_frames = resp.total_frames,
    total_time_ms = resp.total_time_ms,
    tag_name = resp.tag_name,
  }
  if req then
    meta.action = req.action
    meta.prompt = req.prompt
    meta.negative_prompt = req.negative_prompt
    meta.mode = req.mode
    meta.steps = req.steps
    meta.cfg_scale = req.cfg_scale
    meta.clip_skip = req.clip_skip
    meta.denoise_strength = req.denoise_strength
    meta.lora = req.lora
    meta.negative_ti = req.negative_ti
    meta.post_process = req.post_process
    meta.output_size = tostring(req.width or 512) .. "x" .. tostring(req.height or 512)
    meta.seed = req.seed
    -- Animation-specific
    meta.method = req.method
    meta.frame_count_requested = req.frame_count
    meta.frame_duration_ms = req.frame_duration_ms
    meta.seed_strategy = req.seed_strategy
    meta.tag_name = req.tag_name or resp.tag_name
    -- Audio-specific
    if req.audio_path then
      meta.audio_path = req.audio_path
      meta.fps = req.fps
      meta.modulation_slots = req.modulation_slots
      meta.expressions = req.expressions
      meta.modulation_preset = req.modulation_preset
    end
  end
  return meta
end

end
