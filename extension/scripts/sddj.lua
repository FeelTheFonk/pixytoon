--
-- SDDj — Aseprite Extension for SD Generation & Animation
--
-- Connects to the local SDDj Python server via WebSocket
-- and provides a full GUI for generating images and animations.
--
-- Architecture: shared context table (PT) loaded by sub-modules via dofile().
-- Module load order matters: later modules reference functions defined by earlier ones.
-- All cross-module calls resolve at runtime (not load time), so circular references work.
--
-- Follows the standard Aseprite multi-file extension pattern:
--   1. Top-level dofile("./module.lua") for loading (relative paths work here)
--   2. init(plugin) for plugin-specific setup (dialog, settings)
--   3. exit(plugin) for cleanup (timers, WebSocket, settings)
--

-- ─── JSON Loader ──────────────────────────────────────────

local json_ok, json = pcall(dofile, "./json.lua")
if not json_ok or not json then
  app.alert("SDDj: Failed to load json.lua\n" .. tostring(json))
  return
end

local parser_ok, dsl_parser = pcall(dofile, "./sddj_dsl_parser.lua")
if not parser_ok or type(dsl_parser) ~= "table" then
  app.alert("SDDj: Failed to load sddj_dsl_parser.lua\n" .. tostring(dsl_parser))
  return
end

-- ─── Shared Context ───────────────────────────────────────

local _PT = { json = json, dsl_parser = dsl_parser }

-- ─── Module Loader ────────────────────────────────────────
-- dofile("./name.lua") resolves relative to the calling script's
-- directory (Aseprite's custom dofile uses a current_script_dirs stack).
-- This only works at the top level, while the file is being executed.

local modules = {
  "sddj_base64",    -- pure codec, no deps
  "sddj_state",     -- constants + state tables
  "sddj_utils",     -- temp files, image I/O, timer helper, deep copy
  "sddj_settings",  -- save/load/apply
  "sddj_ws",        -- WebSocket transport + connection
  "sddj_capture",   -- image capture (active layer, flattened, mask)
  "sddj_request",   -- request builders (parse, attach, build)
  "sddj_dsl_editor",-- schedule editor popup, timeline, presets
  "sddj_import",    -- import result, animation frame
  "sddj_output",    -- output directory, metadata persistence, load/apply
  "sddj_handler",   -- response dispatch table
  "sddj_dialog",    -- dialog construction (tabs + actions)
}

for _, name in ipairs(modules) do
  local ok, init_fn = pcall(dofile, "./" .. name .. ".lua")
  if not ok then
    app.alert("SDDj: Failed to load " .. name .. "\n" .. tostring(init_fn))
    return
  end
  if type(init_fn) ~= "function" then
    app.alert("SDDj: Module " .. name .. " did not return an init function"
      .. "\nGot: " .. type(init_fn) .. " = " .. tostring(init_fn))
    return
  end
  local init_ok, init_err = pcall(init_fn, _PT)
  if not init_ok then
    app.alert("SDDj: Module " .. name .. " init failed\n" .. tostring(init_err))
    return
  end
end

-- ─── Plugin Lifecycle ─────────────────────────────────────
-- Aseprite calls init(plugin) after executing this file.
-- By this point, all modules are loaded and all functions in _PT are ready.

function init(plugin)
  -- Clean up any leftover temp files from previous sessions
  if _PT.cleanup_session_temp_files then
    pcall(_PT.cleanup_session_temp_files, true)  -- all_sessions = true
  end

  _PT.build_dialog()
  _PT.apply_settings(_PT.load_settings())
end

function exit(plugin)
  if not _PT then return end

  -- Cancel any in-progress generation
  if _PT.state and (_PT.state.generating or _PT.state.animating) then
    pcall(function() _PT.send({ action = "cancel" }) end)
  end

  -- Save settings before exit
  if _PT.save_settings then pcall(_PT.save_settings) end

  -- Send shutdown to server (auto-stop)
  if _PT.state and _PT.state.connected and _PT.ws_handle then
    pcall(function() _PT.ws_handle:sendText('{"action":"shutdown"}') end)
  end

  -- Disconnect WebSocket
  if _PT.ws_handle then
    pcall(function() _PT.ws_handle:close() end)
    _PT.ws_handle = nil
  end

  -- Stop all named timers
  if _PT.timers then
    for key, timer in pairs(_PT.timers) do
      if timer then pcall(function() timer:stop() end) end
      _PT.timers[key] = nil
    end
  end

  -- Clean up session temp files
  if _PT.cleanup_session_temp_files then
    pcall(_PT.cleanup_session_temp_files)
  end
end
