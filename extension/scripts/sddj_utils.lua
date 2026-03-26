--
-- SDDj — Utility Functions
--

return function(PT)

-- ─── Temp File Management ───────────────────────────────────

function PT.get_tmp_dir()
  return app.fs.tempPath or os.getenv("TEMP") or os.getenv("TMP") or "."
end

function PT.make_tmp_path(prefix)
  PT.state.file_counter = PT.state.file_counter + 1
  return app.fs.joinPath(PT.get_tmp_dir(),
    "sddj_" .. prefix .. "_" .. PT.state.session_id .. "_" .. PT.state.file_counter .. ".png")
end

-- ─── Image I/O ──────────────────────────────────────────────

function PT.image_to_base64(img)
  local tmp = PT.make_tmp_path("b64")
  if not img:saveAs(tmp) then return nil end
  local f = io.open(tmp, "rb")
  if not f then return nil end
  local data = f:read("*a")
  f:close()
  os.remove(tmp)
  return PT.base64_encode(data)
end

-- ─── Temp File Cleanup ────────────────────────────────────────

-- Clean up temp files for the current session (or all sddj temp files).
function PT.cleanup_session_temp_files(all_sessions)
  local tmp_dir = PT.get_tmp_dir()
  local ok, files = pcall(app.fs.listFiles, tmp_dir)
  if not ok or not files then return end
  local count = 0
  for _, name in ipairs(files) do
    if name:find("^sddj_") and name:find("%.png$") then
      if all_sessions or name:find(PT.state.session_id) then
        pcall(os.remove, app.fs.joinPath(tmp_dir, name))
        count = count + 1
      end
    end
  end
  if count > 0 then
    -- Silent cleanup, no status update needed
  end
end

-- ─── Deep Copy (metadata tracking) ────────────────────────────

-- Deep-copy a request table, excluding heavy base64 image fields.
-- Depth-limited (max 32) with cycle detection to prevent stack overflow.
local _IMAGE_KEYS = { source_image = true, mask_image = true, control_image = true, image = true }
local _MAX_COPY_DEPTH = 32

local function _deep_copy(src, depth, seen)
  if type(src) ~= "table" then return src end
  if depth > _MAX_COPY_DEPTH then return nil end
  if seen[src] then return nil end
  seen[src] = true
  local dst = {}
  for k, v in pairs(src) do
    if _IMAGE_KEYS[k] then
      -- skip (don't store multi-MB base64 blobs)
    elseif type(v) == "table" then
      dst[k] = _deep_copy(v, depth + 1, seen)
    else
      dst[k] = v
    end
  end
  seen[src] = nil  -- DAG-safe: allow same table in sibling branches
  return dst
end

function PT.deep_copy_request(src)
  return _deep_copy(src, 0, {})
end

-- ─── Timer Lifecycle ────────────────────────────────────────

-- Stop a timer safely. Returns nil for idiomatic reassignment:
--   timer = PT.stop_timer(timer)
function PT.stop_timer(t)
  if t then
    pcall(function() if t.isRunning then t:stop() end end)
  end
  return nil
end

-- ─── Loop State Reset (shared across dialog/handler/ws) ───────

function PT.reset_loop_state()
  PT.loop.mode = false
  PT.loop.random_mode = false
  PT.loop.target = nil
end

-- ─── Slider Label Registry ────────────────────────────────────
-- Single source of truth for all slider label formatting.
-- Used by sddj_dialog.lua (onchange), sddj_settings.lua (apply),
-- and sddj_output.lua (apply_metadata).
-- Entry format: { fmt_string, divisor } for float, { fmt_string } for integer.

PT.SLIDER_LABELS = {
  -- Float-divided sliders
  cfg_scale             = { "CFG (%.1f)",        10.0 },
  denoise               = { "Strength (%.2f)",   100.0 },
  lora_weight           = { "LoRA (%.2f)",       100.0 },
  neg_ti_weight         = { "Emb. (%.2f)",       100.0 },
  anim_cfg              = { "CFG (%.1f)",        10.0 },
  anim_denoise          = { "Strength (%.2f)",   100.0 },
  audio_cfg             = { "CFG (%.1f)",        10.0 },
  audio_denoise         = { "Strength (%.2f)",   100.0 },
  qr_denoise            = { "Denoise (%.2f)",    100.0 },
  qr_conditioning_scale = { "CN Scale (%.2f)",   100.0 },
  qr_guidance_start     = { "Guide Start (%.2f)", 100.0 },
  qr_guidance_end       = { "Guide End (%.2f)",  100.0 },
  qr_cfg                = { "CFG (%.1f)",        10.0 },
  -- Integer sliders
  steps                 = { "Steps (%d)" },
  clip_skip             = { "CLIP Skip (%d)" },
  pixel_size            = { "Target (%dpx)" },
  colors                = { "Colors (%d)" },
  anim_steps            = { "Steps (%d)" },
  anim_frames           = { "Frames (%d)" },
  anim_duration         = { "Duration (%dms)" },
  audio_steps           = { "Steps (%d)" },
  mod_slot_count        = { "Slots (%d)" },
  qr_steps              = { "Steps (%d)" },
}

function PT.sync_slider_label(id)
  if not PT.dlg then return end
  local spec = PT.SLIDER_LABELS[id]
  if not spec then return end
  local val = PT.dlg.data[id]
  if val == nil then return end
  local label
  if spec[2] then
    label = string.format(spec[1], val / spec[2])
  else
    label = string.format(spec[1], val)
  end
  PT.dlg:modify{ id = id, label = label }
end

end
