--
-- SDDj — Output Directory & Metadata Persistence
--

return function(PT)

-- ─── Project Root Detection ───────────────────────────────────
-- The extension runs from %appdata%/Aseprite/extensions/sddj/ (installed copy).
-- Derive <sddj_root> from Aseprite's executable path instead:
-- Aseprite is at <root>/bin/aseprite/aseprite.exe → go up 3 levels.
-- Fallback: script path for dev scenarios where Aseprite is external.

local _project_root
local _ase_exe = app.fs.appPath                                    -- full path to aseprite.exe
local _ase_dir = app.fs.filePath(_ase_exe)                         -- .../bin/aseprite/
local _bin_dir = app.fs.filePath(_ase_dir)                         -- .../bin/
local _candidate = app.fs.filePath(_bin_dir)                       -- .../  (project root)
if app.fs.isFile(app.fs.joinPath(_candidate, "start.ps1")) then
  _project_root = _candidate
else
  -- Dev fallback: script at <root>/extension/scripts/sddj_output.lua → up 3
  local _raw_source = debug.getinfo(1, "S").source
  local _script_path = _raw_source:sub(1, 1) == "@" and _raw_source:sub(2) or _raw_source
  local _scripts_dir = app.fs.filePath(_script_path)
  local _ext_dir     = app.fs.filePath(_scripts_dir)
  _project_root = app.fs.filePath(_ext_dir)
end

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
    if not app.fs.makeDirectory(root) then
      PT.update_status("Cannot create output root: " .. tostring(root))
      return nil
    end
  end
  local date_str = os.date("%Y-%m-%d")
  local date_dir = app.fs.joinPath(root, date_str)
  if not app.fs.isDirectory(date_dir) then
    if not app.fs.makeDirectory(date_dir) then
      PT.update_status("Cannot create output folder: " .. tostring(date_dir))
      return nil
    end
  end
  return date_dir
end

-- ─── Single Generation Save ──────────────────────────────────

function PT.save_to_output(resp, meta)
  if not PT.output.enabled then return end
  if not resp or (not resp._raw_image and not resp.image) then return end

  local ok, err = pcall(function()
    local date_dir = PT.ensure_date_dir()
    if not date_dir then return end
    local time_str = os.date("%H-%M-%S")
    local seed_str = tostring(resp.seed or 0)
    _output_counter = _output_counter + 1
    local base_name = time_str .. "_" .. string.format("%04d", _output_counter) .. "_s" .. seed_str

    -- Reuse decoded bytes from import (no double decode); binary frame is fastest
    local img_data = resp._raw_image or resp._decoded_bytes or PT.base64_decode(resp.image)
    if not img_data or #img_data == 0 then return end

    -- Save PNG (encoding-aware: handles both PNG and raw_rgba)
    local png_path = app.fs.joinPath(date_dir, base_name .. ".png")
    if (resp.encoding == "raw_rgba" or resp._raw_image) and resp.width and resp.height then
      local img = Image(resp.width, resp.height, ColorMode.RGB)
      img.bytes = img_data
      img:saveAs(png_path)
    else
      local f = io.open(png_path, "wb")
      if f then
        f:write(img_data)
        f:close()
      else
        PT.update_status("Save I/O Error: Cannot write to " .. tostring(png_path))
      end
    end

    -- Save metadata JSON
    if meta then
      meta.output_file = base_name .. ".png"
      local json_path = app.fs.joinPath(date_dir, base_name .. ".json")
      local jf = io.open(json_path, "w")
      if jf then
        jf:write(PT.json.encode(meta))
        jf:close()
      else
        PT.update_status("Save I/O Error: Cannot write to " .. tostring(json_path))
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
  if PT.state.cancel_pending then return end
  if not PT.dlg or not PT.dlg.data.save_output then return end
  if (not resp._raw_image and not resp.image) or resp.frame_index == nil then return end

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
      local folder_name = time_str .. "_" .. string.format("%04d", _output_counter) .. "_" .. action_prefix .. "_" .. tag
      PT.anim.output_dir = app.fs.joinPath(date_dir, folder_name)
      if not app.fs.makeDirectory(PT.anim.output_dir) then
        PT.update_status("Cannot create output dir: " .. folder_name)
        PT.anim.output_dir = nil
        return
      end
      PT.anim.output_count = 0
    end

    -- Write frame to disk immediately
    local frame_name = string.format("frame_%05d.png", resp.frame_index + 1)
    local frame_path = app.fs.joinPath(PT.anim.output_dir, frame_name)

    -- Reuse decoded bytes from import_animation_frame (B3: no double decode); binary frame is fastest
    local img_data = resp._raw_image or resp._decoded_bytes or PT.base64_decode(resp.image)

    if img_data and #img_data > 0 then
      if (resp.encoding == "raw_rgba" or resp._raw_image) and resp.width and resp.height then
        -- Raw RGBA: create Image, save as PNG natively (raw bytes ≠ valid PNG)
        local img = Image(resp.width, resp.height, ColorMode.RGB)
        img.bytes = img_data
        img:saveAs(frame_path)
      else
        -- Legacy PNG: write directly
        local f = io.open(frame_path, "wb")
        if f then
          f:write(img_data)
          f:close()
        else
          PT.update_status("Anim I/O Error: Cannot write frame to " .. tostring(frame_path))
        end
      end
      PT.anim.last_saved_frame = frame_path
      PT.anim.output_count = PT.anim.output_count + 1
    elseif PT.anim.last_saved_frame then
      -- Decode failed — copy previous frame to avoid numbering gaps (breaks ffmpeg)
      local src = io.open(PT.anim.last_saved_frame, "rb")
      if src then
        local data = src:read("*a")
        src:close()
        local dst = io.open(frame_path, "wb")
        if dst then
          dst:write(data)
          dst:close()
          PT.anim.output_count = PT.anim.output_count + 1
        else
          PT.update_status("Anim I/O Error: Cannot copy fallback frame")
        end
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
    else
      PT.update_status("Anim Meta I/O Error: Cannot write to " .. tostring(json_path))
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
  -- Sanitize: strip shell metacharacters only (preserve parens, tilde, etc.)
  local safe_dir = dir:gsub('[`$|<>&;%%!]', "")
  if package.config:sub(1, 1) == "\\" then
    safe_dir = safe_dir:gsub("/", "\\")
    os.execute('explorer "' .. safe_dir .. '"')
  else
    -- Cache OS detection (avoid process spawn per click)
    if not PT._cached_os then
      if app.os and app.os.macos then
        PT._cached_os = "Darwin"
      elseif app.os and app.os.windows then
        PT._cached_os = "Windows"
      else
        local ok, handle = pcall(io.popen, "uname -s")
        if ok and handle then
          PT._cached_os = handle:read("*l") or "Linux"
          handle:close()
        else
          PT._cached_os = "Linux"
        end
      end
    end
    if PT._cached_os == "Darwin" then
      os.execute('open "' .. safe_dir .. '"')
    else
      os.execute('xdg-open "' .. safe_dir .. '"')
    end
  end
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

  PT._ui_transaction_depth = (PT._ui_transaction_depth or 0) + 1
  local ok, err = pcall(function()

  local dlg = PT.dlg

  -- Text fields
  if meta.prompt then dlg:modify{ id = "prompt", text = meta.prompt } end
  if meta.negative_prompt then dlg:modify{ id = "negative_prompt", text = meta.negative_prompt } end
  if meta.seed then dlg:modify{ id = "seed", text = tostring(meta.seed) } end

  -- Combobox fields (pcall: option may not exist in current list)
  if meta.mode then pcall(dlg.modify, dlg, { id = "mode", option = meta.mode }) end
  if meta.output_size then pcall(dlg.modify, dlg, { id = "output_size", option = meta.output_size }) end

  -- Slider fields (requires inverse scaling)
  if meta.steps then dlg:modify{ id = "steps", value = meta.steps } end
  if meta.cfg_scale then
    local v = math.floor(meta.cfg_scale * 10)
    dlg:modify{ id = "cfg_scale", value = v }
    PT.sync_slider_label("cfg_scale")
  end
  if meta.clip_skip then dlg:modify{ id = "clip_skip", value = meta.clip_skip } end
  if meta.denoise_strength then
    local v = math.floor(meta.denoise_strength * 100)
    dlg:modify{ id = "denoise", value = v }
    PT.sync_slider_label("denoise")
  end

  -- LoRA
  if meta.lora and meta.lora.name then
    pcall(dlg.modify, dlg, { id = "lora_name", option = meta.lora.name })
    if meta.lora.weight then
      local v = math.floor(meta.lora.weight * 100)
      dlg:modify{ id = "lora_weight", value = v }
      PT.sync_slider_label("lora_weight")
    end
  end

  -- Post-process
  if meta.post_process then
    local pp = meta.post_process
    if pp.pixelate ~= nil then
      if type(pp.pixelate) == "table" then
        if pp.pixelate.enabled ~= nil then dlg:modify{ id = "pixelate", selected = pp.pixelate.enabled } end
        if pp.pixelate.target_size then
          dlg:modify{ id = "pixel_size", value = pp.pixelate.target_size }
          PT.sync_slider_label("pixel_size")
        end
        if pp.pixelate.method then pcall(dlg.modify, dlg, { id = "pixelate_method", option = pp.pixelate.method }) end
      end
    end
    if pp.quantize_enabled ~= nil then dlg:modify{ id = "quantize_enabled", selected = pp.quantize_enabled } end
    if pp.quantize_colors then dlg:modify{ id = "colors", value = pp.quantize_colors } end
    if pp.quantize_method then pcall(dlg.modify, dlg, { id = "quantize_method", option = pp.quantize_method }) end
    if pp.dither then pcall(dlg.modify, dlg, { id = "dither", option = pp.dither }) end
    if pp.remove_bg ~= nil then dlg:modify{ id = "remove_bg", selected = pp.remove_bg } end
    if pp.palette then
      if pp.palette.mode then pcall(dlg.modify, dlg, { id = "palette_mode", option = pp.palette.mode }) end
      if pp.palette.name then pcall(dlg.modify, dlg, { id = "palette_name", option = pp.palette.name }) end
    end
  end

  -- Animation-specific fields
  if meta.method then pcall(dlg.modify, dlg, { id = "anim_method", option = meta.method }) end
  if meta.frame_count then dlg:modify{ id = "anim_frames", value = meta.frame_count } end
  if meta.seed_strategy then pcall(dlg.modify, dlg, { id = "anim_seed_strategy", option = meta.seed_strategy }) end
  if meta.tag_name then
    if meta.action == "generate_audio_reactive" then
      pcall(function() dlg:modify{ id = "audio_tag", text = meta.tag_name } end)
    else
      dlg:modify{ id = "anim_tag", text = meta.tag_name }
    end
  end

  -- Lock Subject state
  if meta.lock_subject ~= nil then
    dlg:modify{ id = "lock_subject", selected = meta.lock_subject }
  end
  if meta.fixed_subject then
    dlg:modify{ id = "fixed_subject", text = meta.fixed_subject }
  end
  if meta.subject_position then
    pcall(dlg.modify, dlg, { id = "subject_position", option = meta.subject_position })
  end
  -- Lock Custom state
  if meta.lock_custom ~= nil then
    dlg:modify{ id = "lock_custom", selected = meta.lock_custom }
  end
  if meta.fixed_custom then
    dlg:modify{ id = "fixed_custom", text = meta.fixed_custom }
  end
  if meta.custom_position then
    pcall(dlg.modify, dlg, { id = "custom_position", option = meta.custom_position })
  end

  end)
  PT._ui_transaction_depth = PT._ui_transaction_depth - 1
  if not ok then
    -- Log but don't crash — sync must still run
    print("[SDDj] apply_metadata error: " .. tostring(err))
  end

  -- Centralized sync of all conditional widget states after data injection
  PT.sync_ui_conditional_states()

  PT.update_status("Metadata loaded (seed=" .. tostring(meta.seed or "?") .. ")")
end

-- ─── Build Metadata from Request + Response ────────────────────

function PT.build_generation_meta(resp)
  local req = PT.last_request
  local meta = {
    sddj_version = PT.VERSION,
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
  -- Lock Subject/Custom state for reproducibility
  if PT.dlg then
    meta.lock_subject = PT.dlg.data.lock_subject or false
    local subj = PT.dlg.data.fixed_subject or ""
    if subj ~= "" then meta.fixed_subject = subj end
    meta.subject_position = PT.dlg.data.subject_position or "prefix"
    meta.lock_custom = PT.dlg.data.lock_custom or false
    local cust = PT.dlg.data.fixed_custom or ""
    if cust ~= "" then meta.fixed_custom = cust end
    meta.custom_position = PT.dlg.data.custom_position or "suffix"
  end
  return meta
end

function PT.build_animation_meta(resp)
  local req = PT.last_request
  local meta = {
    sddj_version = PT.VERSION,
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
    if req.frame_duration_ms then meta.frame_duration_ms = req.frame_duration_ms end
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
  -- Lock Subject/Custom state for reproducibility
  if PT.dlg then
    meta.lock_subject = PT.dlg.data.lock_subject or false
    local subj = PT.dlg.data.fixed_subject or ""
    if subj ~= "" then meta.fixed_subject = subj end
    meta.subject_position = PT.dlg.data.subject_position or "prefix"
    meta.lock_custom = PT.dlg.data.lock_custom or false
    local cust = PT.dlg.data.fixed_custom or ""
    if cust ~= "" then meta.fixed_custom = cust end
    meta.custom_position = PT.dlg.data.custom_position or "suffix"
  end
  return meta
end

end
