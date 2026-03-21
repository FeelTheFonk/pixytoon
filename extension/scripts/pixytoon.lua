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
-- Uses the official Aseprite plugin pattern: package.json contributes.scripts
-- defines this file, Aseprite executes it, then calls init(plugin) with
-- a plugin object whose .path field points to the extension root directory.
--

-- ─── Shared State (accessible by both init and exit) ──────

local _PT = nil

-- ─── Path Detection ───────────────────────────────────────
-- Tries multiple strategies to locate the scripts directory.
-- Strategy 1 (plugin.path) is the most reliable for extensions.

local function find_scripts_dir(plugin)
  local tried = {}

  -- Strategy 1: plugin.path (official Aseprite plugin API)
  -- plugin.path = extension root (where package.json lives)
  if plugin and plugin.path then
    local dir = app.fs.joinPath(plugin.path, "scripts")
    tried[#tried + 1] = "[plugin.path/scripts] " .. dir
    if app.fs.isFile(app.fs.joinPath(dir, "json.lua")) then
      return dir, tried
    end
    -- Maybe all files are directly in plugin root (flat layout)
    tried[#tried + 1] = "[plugin.path] " .. plugin.path
    if app.fs.isFile(app.fs.joinPath(plugin.path, "json.lua")) then
      return plugin.path, tried
    end
  end

  -- Strategy 2: debug.getinfo (fallback for edge cases)
  for level = 1, 6 do
    local ok, info = pcall(debug.getinfo, level, "S")
    if ok and info and info.source then
      local src = info.source
      if src:sub(1, 1) == "@" then
        local raw = src:sub(2)
        local dir = app.fs.filePath(raw)
        if dir and dir ~= "" then
          tried[#tried + 1] = "[debug L" .. level .. "] " .. dir
          if app.fs.isFile(app.fs.joinPath(dir, "json.lua")) then
            return dir, tried
          end
        end
      end
    end
  end

  -- Strategy 3: Well-known Aseprite extension paths
  local config = app.fs.userConfigPath
  local candidates = {
    app.fs.joinPath(config, "extensions", "pixytoon", "scripts"),
    app.fs.joinPath(config, "extensions", "pixytoon"),
  }
  for _, dir in ipairs(candidates) do
    tried[#tried + 1] = "[fallback] " .. dir
    if app.fs.isFile(app.fs.joinPath(dir, "json.lua")) then
      return dir, tried
    end
  end

  return nil, tried
end

-- ─── JSON Loader ──────────────────────────────────────────

local function load_json(scripts_dir)
  local path = app.fs.joinPath(scripts_dir, "json.lua")
  local ok, result = pcall(dofile, path)
  if not ok or not result then
    app.alert("PixyToon: Failed to load json.lua\n" .. tostring(result))
    return nil
  end
  return result
end

-- ─── Module Loader ────────────────────────────────────────

local function load_module(PT, scripts_dir, name)
  -- dofile with absolute path (recommended by Aseprite creator dacap)
  local path = app.fs.joinPath(scripts_dir, name .. ".lua")
  if not app.fs.isFile(path) then
    -- Comprehensive diagnostic for troubleshooting
    local diag = "PixyToon: Module file not found!\n\n"
    diag = diag .. "Module: " .. name .. ".lua\n"
    diag = diag .. "Expected at: " .. path .. "\n"
    diag = diag .. "scripts_dir: " .. scripts_dir .. "\n\n"
    diag = diag .. "Files found in directory:\n"
    local ok_list, files = pcall(app.fs.listFiles, scripts_dir)
    if ok_list and files then
      for _, f in ipairs(files) do
        diag = diag .. "  " .. f .. "\n"
      end
      if #files == 0 then diag = diag .. "  (empty directory)\n" end
    else
      diag = diag .. "  (cannot list: " .. tostring(files) .. ")\n"
    end
    diag = diag .. "\nFix: rebuild and reinstall the extension:\n"
    diag = diag .. "  python scripts/build_extension.py\n"
    diag = diag .. "  Then double-click dist/pixytoon.aseprite-extension"
    app.alert(diag)
    return false
  end

  local ok, init_fn = pcall(dofile, path)
  if not ok then
    app.alert("PixyToon: Error loading " .. name .. "\n" .. tostring(init_fn))
    return false
  end
  if type(init_fn) ~= "function" then
    app.alert("PixyToon: Module " .. name .. " did not return an init function"
      .. "\nGot: " .. type(init_fn) .. " = " .. tostring(init_fn))
    return false
  end
  local init_ok, init_err = pcall(init_fn, PT)
  if not init_ok then
    app.alert("PixyToon: Module " .. name .. " init failed\n" .. tostring(init_err))
    return false
  end
  return true
end

-- ─── Plugin Lifecycle ─────────────────────────────────────
-- Aseprite executes this file top-to-bottom (defining functions),
-- then calls init(plugin) with the plugin object.
-- plugin.path = extension root directory (where package.json lives).

function init(plugin)
  local scripts_dir, tried = find_scripts_dir(plugin)
  if not scripts_dir then
    local msg = "PixyToon: Cannot find scripts directory!\n\n"
    msg = msg .. "Tried " .. #tried .. " locations:\n"
    for _, t in ipairs(tried) do
      msg = msg .. "  " .. t .. "\n"
    end
    msg = msg .. "\nuserConfigPath: " .. tostring(app.fs.userConfigPath) .. "\n"
    if plugin and plugin.path then
      msg = msg .. "plugin.path: " .. tostring(plugin.path) .. "\n"
    end
    msg = msg .. "\nPlease rebuild and reinstall the extension."
    app.alert(msg)
    return
  end

  -- Load JSON library
  local json = load_json(scripts_dir)
  if not json then return end

  -- Create shared context table
  _PT = { json = json, plugin = plugin }

  -- Load modules in dependency order
  local modules = {
    "pixytoon_base64",    -- pure codec, no deps
    "pixytoon_state",     -- constants + state tables
    "pixytoon_utils",     -- temp files, image I/O, timer helper
    "pixytoon_settings",  -- save/load/apply
    "pixytoon_ws",        -- WebSocket transport + connection
    "pixytoon_capture",   -- image capture (active layer, flattened, mask)
    "pixytoon_request",   -- request builders (parse, attach, build)
    "pixytoon_import",    -- import result, animation frame, live preview
    "pixytoon_live",      -- live paint system (hash, dirty region, timers)
    "pixytoon_handler",   -- response dispatch table
    "pixytoon_dialog",    -- dialog construction (tabs + actions)
  }

  for _, name in ipairs(modules) do
    if not load_module(_PT, scripts_dir, name) then return end
  end

  -- Launch the dialog
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

  -- Stop live mode timers
  if _PT.stop_live_timer then pcall(_PT.stop_live_timer) end

  -- Disconnect WebSocket
  if _PT.ws_handle then
    pcall(function() _PT.ws_handle:close() end)
    _PT.ws_handle = nil
  end

  -- Save settings before exit
  if _PT.save_settings then pcall(_PT.save_settings) end
end
