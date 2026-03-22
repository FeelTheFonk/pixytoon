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

-- Raw PNG bytes for binary WebSocket send (skips base64 overhead).
function PT.image_to_png_bytes(img)
  local tmp = PT.make_tmp_path("bin")
  if not img:saveAs(tmp) then return nil end
  local f = io.open(tmp, "rb")
  if not f then return nil end
  local data = f:read("*a")
  f:close()
  os.remove(tmp)
  return data
end

-- ─── Temp File Cleanup ────────────────────────────────────────

-- Clean up temp files for the current session (or all sddj temp files).
function PT.cleanup_session_temp_files(all_sessions)
  local tmp_dir = PT.get_tmp_dir()
  local pattern = all_sessions and "sddj_" or ("sddj_.*_" .. PT.state.session_id .. "_")
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
local _IMAGE_KEYS = { source_image = true, mask_image = true, control_image = true, image = true }

function PT.deep_copy_request(src)
  if type(src) ~= "table" then return src end
  local dst = {}
  for k, v in pairs(src) do
    if _IMAGE_KEYS[k] then
      -- skip (don't store multi-MB base64 blobs)
    elseif type(v) == "table" then
      dst[k] = PT.deep_copy_request(v)
    else
      dst[k] = v
    end
  end
  return dst
end

-- ─── Timer Lifecycle ────────────────────────────────────────

-- Stop a timer safely. Returns nil for idiomatic reassignment:
--   timer = PT.stop_timer(timer)
function PT.stop_timer(t)
  if t then
    if t.isRunning then t:stop() end
  end
  return nil
end

end
