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

-- Shallow copy: only metadata-relevant fields (skips heavy binary blobs).
-- Single-level deep for nested tables — avoids full recursive traversal.
function PT.shallow_copy_request(req)
  if type(req) ~= "table" then return req end
  local copy = {}
  for k, v in pairs(req) do
    if k ~= "image" and k ~= "source_image" and k ~= "mask_image" and k ~= "control_image" and k ~= "_decoded_bytes" and k ~= "_raw_image" then
      if type(v) == "table" then
        local t = {}
        for tk, tv in pairs(v) do t[tk] = tv end
        copy[k] = t
      else
        copy[k] = v
      end
    end
  end
  return copy
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
  lora2_weight          = { "Weight (%.2f)",     100.0 },
  ip_adapter_scale      = { "Scale (%.2f)",      100.0 },
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
  anim_guidance_start   = { "Guide Start (%.2f)", 100.0 },
  anim_guidance_end     = { "Guide End (%.2f)",  100.0 },
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

-- ─── PARAM_DEFS: Authoritative Modulation Target Registry ────
-- Defines the real-value range and the percentage→real mapping
-- for every modulation target. Used by:
--   • sddj_request.lua  (modulation slot scaling)
--   • sddj_dsl_editor.lua  (per-keyframe override validation)
--   • sddj_dialog.lua  (UI hint labels)
--
-- Each entry: { real_min, real_max }
-- The scaling formula is:  real = real_min + pct * (real_max − real_min)
-- where pct = slider_value / 100.0  (0→1 range)

PT.PARAM_DEFS = {
  denoise_strength  = { 0.0,   1.0   },   -- 0..1
  cfg_scale         = { 0.0,  30.0   },   -- 0..30
  noise_amplitude   = { 0.0,   1.0   },   -- 0..1
  controlnet_scale  = { 0.0,   2.0   },   -- 0..2
  seed_offset       = { 0,  1000     },   -- 0..1000 (integer)
  palette_shift     = { 0.0,   1.0   },   -- 0..1
  frame_cadence     = { 1.0,   8.0   },   -- 1..8
  motion_x          = { -5.0,  5.0   },   -- −5..+5 px/frame
  motion_y          = { -5.0,  5.0   },   -- −5..+5 px/frame
  motion_zoom       = { 0.92,  1.08  },   -- zoom factor
  motion_rotation   = { -2.0,  2.0   },   -- degrees/frame
  motion_tilt_x     = { -3.0,  3.0   },   -- perspective tilt
  motion_tilt_y     = { -3.0,  3.0   },   -- perspective tilt
}

--- Scale a 0–100% value to the real range for a given target.
-- @param target string  PARAM_DEFS key
-- @param pct number  0..100 percentage
-- @return number  real value, or pct/100 if target is unknown
function PT.scale_mod_value(target, pct)
  local def = PT.PARAM_DEFS[target]
  if not def then return pct / 100.0 end
  local t = pct / 100.0  -- normalize to 0..1
  return def[1] + t * (def[2] - def[1])
end

--- Inverse: convert a real parameter value back to 0-100% slider range.
-- @param target string  PARAM_DEFS key
-- @param val number  real value
-- @return number  percentage 0..100, clamped and rounded
function PT.inverse_scale_mod_value(target, val)
  local def = PT.PARAM_DEFS[target]
  if not def then return math.max(0, math.min(100, math.floor(val * 100 + 0.5))) end
  if def[2] == def[1] then return 50 end
  return math.max(0, math.min(100, math.floor(((val - def[1]) / (def[2] - def[1])) * 100 + 0.5)))
end

end
