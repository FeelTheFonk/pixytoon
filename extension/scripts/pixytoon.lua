--
-- PixyToon — Aseprite Extension for AI Pixel Art Generation
--
-- Connects to the local PixyToon Python server via WebSocket
-- and provides a full GUI for generating pixel art sprites.
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
  app.alert("PixyToon: Failed to load json.lua\n" .. tostring(json))
  return
end

-- ─── Shared Context ───────────────────────────────────────

local _PT = { json = json }

-- ─── Module Loader ────────────────────────────────────────
-- dofile("./name.lua") resolves relative to the calling script's
-- directory (Aseprite's custom dofile uses a current_script_dirs stack).
-- This only works at the top level, while the file is being executed.

local modules = {
  "pixytoon_base64",    -- pure codec, no deps
  "pixytoon_state",     -- constants + state tables
  "pixytoon_utils",     -- temp files, image I/O, timer helper
  "pixytoon_settings",  -- save/load/apply
  "pixytoon_ws",        -- WebSocket transport + connection
  "pixytoon_capture",   -- image capture (active layer, flattened, mask)
  "pixytoon_request",   -- request builders (parse, attach, build)
  "pixytoon_import",    -- import result, animation frame, live preview
  "pixytoon_live",      -- live paint system (event-driven, dirty region, F5 hotkey)
  "pixytoon_handler",   -- response dispatch table
  "pixytoon_dialog",    -- dialog construction (tabs + actions)
}

for _, name in ipairs(modules) do
  local ok, init_fn = pcall(dofile, "./" .. name .. ".lua")
  if not ok then
    app.alert("PixyToon: Failed to load " .. name .. "\n" .. tostring(init_fn))
    return
  end
  if type(init_fn) ~= "function" then
    app.alert("PixyToon: Module " .. name .. " did not return an init function"
      .. "\nGot: " .. type(init_fn) .. " = " .. tostring(init_fn))
    return
  end
  local init_ok, init_err = pcall(init_fn, _PT)
  if not init_ok then
    app.alert("PixyToon: Module " .. name .. " init failed\n" .. tostring(init_err))
    return
  end
end

-- ─── Plugin Lifecycle ─────────────────────────────────────
-- Aseprite calls init(plugin) after executing this file.
-- By this point, all modules are loaded and all functions in _PT are ready.

function init(plugin)
  _PT.plugin = plugin

  -- Register hotkey command for live send (F5 default via .aseprite-keys)
  plugin:newCommand{
    id = "PixyToonLiveSend",
    title = "PixyToon: Send Live Frame",
    group = "sprite_crop",
    onenabled = function()
      return _PT.live.mode and _PT.state.connected and not _PT.live.request_inflight
    end,
    onclick = function()
      if _PT.live_send_now then _PT.live_send_now() end
    end,
  }

  _PT.build_dialog()
  _PT.apply_settings(_PT.load_settings())
end

function exit(plugin)
  if not _PT then return end

  -- Stop all named timers
  if _PT.timers then
    for key, timer in pairs(_PT.timers) do
      if timer then pcall(function() timer:stop() end) end
      _PT.timers[key] = nil
    end
  end

  -- Stop live mode timers + event listeners
  if _PT.stop_live_timer then pcall(_PT.stop_live_timer) end

  -- Disconnect WebSocket
  if _PT.ws_handle then
    pcall(function() _PT.ws_handle:close() end)
    _PT.ws_handle = nil
  end

  -- Save settings before exit
  if _PT.save_settings then pcall(_PT.save_settings) end
end
