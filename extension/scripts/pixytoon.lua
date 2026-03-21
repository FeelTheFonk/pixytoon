--
-- PixyToon — Aseprite Extension for AI Pixel Art Generation
--
-- Connects to the local PixyToon Python server via WebSocket
-- and provides a full GUI for generating pixel art sprites.
--

-- ─── JSON LOADER (robust) ───────────────────────────────────

local json
do
  local scripts_path = app.fs.joinPath(app.fs.userConfigPath, "scripts", "json.lua")
  local ext_path = app.fs.joinPath(app.fs.userConfigPath, "extensions", "pixytoon", "scripts", "json.lua")
  local load_ok, load_result
  if app.fs.isFile(scripts_path) then
    load_ok, load_result = pcall(dofile, scripts_path)
  elseif app.fs.isFile(ext_path) then
    load_ok, load_result = pcall(dofile, ext_path)
  else
    load_ok = false
    load_result = "json.lua not found at:\n" .. scripts_path .. "\n" .. ext_path
  end
  if not load_ok or not load_result then
    app.alert("PixyToon: Failed to load json.lua\n" .. tostring(load_result))
    return
  end
  json = load_result
end

-- ─── BASE64 ──────────────────────────────────────────────────

local b64chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/'

local function base64_encode(data)
  return ((data:gsub('.', function(x)
    local r, b = '', x:byte()
    for i = 8, 1, -1 do r = r .. (b % 2^i - b % 2^(i-1) > 0 and '1' or '0') end
    return r
  end) .. '0000'):gsub('%d%d%d?%d?%d?%d?', function(x)
    if #x < 6 then return '' end
    local c = 0
    for i = 1, 6 do c = c + (x:sub(i,i) == '1' and 2^(6-i) or 0) end
    return b64chars:sub(c+1, c+1)
  end) .. ({ '', '==', '=' })[#data % 3 + 1])
end

local function base64_decode(data)
  data = data:gsub('[^' .. b64chars .. '=]', '')
  return (data:gsub('.', function(x)
    if x == '=' then return '' end
    local r, f = '', (b64chars:find(x) - 1)
    for i = 6, 1, -1 do r = r .. (f % 2^i - f % 2^(i-1) > 0 and '1' or '0') end
    return r
  end):gsub('%d%d%d?%d?%d?%d?%d?%d?', function(x)
    if #x ~= 8 then return '' end
    local c = 0
    for i = 1, 8 do c = c + (x:sub(i,i) == '1' and 2^(8-i) or 0) end
    return string.char(c)
  end))
end

-- ─── STATE ───────────────────────────────────────────────────

local SERVER_URL = "ws://127.0.0.1:9876/ws"
local ws = nil
local dlg = nil
local connected = false
local generating = false
local available_palettes = {}
local resources_requested = false
local connect_timer = nil
local heartbeat_timer = nil
local gen_step_start = nil
local _file_counter = 0
local _session_id = tostring(os.time()) .. "_" .. tostring(math.random(1000, 9999))

-- Animation state
local animating = false
local anim_layer = nil
local anim_start_frame = 0
local anim_frame_count = 0
local anim_base_seed = 0
local available_loras = {}
local available_embeddings = {}

-- Live paint state
local live_mode = false
local live_timer = nil
local live_canvas_hash = nil
local live_frame_id = 0
local live_request_inflight = false
local live_preview_layer = nil
local live_last_prompt = nil
local live_preview_sprite = nil
local live_prev_canvas = nil
local live_stroke_cooldown = nil
local live_cooldown_timer = nil  -- C6: reusable cooldown timer (avoid leak)

-- Loop mode state
local loop_mode = false
local loop_counter = 0
local loop_seed_mode = "random"

-- Cancel debounce state (H10)
local cancel_pending = false

-- Presets state
local available_presets = {}

-- Generation timeout
local gen_timeout_timer = nil
local GEN_TIMEOUT_SECONDS = 300  -- 5 minutes

-- Settings persistence
local SETTINGS_FILE = app.fs.joinPath(app.fs.userConfigPath, "pixytoon_settings.json")

-- Forward declarations
local handle_response
local import_result
local import_animation_frame
local request_resources
local start_heartbeat
local stop_heartbeat
local stop_live_timer
local start_live_timer
local live_update_preview
local stop_gen_timeout
local start_gen_timeout
local save_settings
local load_settings

-- ─── HELPERS ─────────────────────────────────────────────────

local function get_tmp_dir()
  return app.fs.tempPath or os.getenv("TEMP") or os.getenv("TMP") or "."
end

local function make_tmp_path(prefix)
  _file_counter = _file_counter + 1
  return app.fs.joinPath(get_tmp_dir(),
    "pixytoon_" .. prefix .. "_" .. _session_id .. "_" .. _file_counter .. ".png")
end

local function image_to_base64(img)
  local tmp = make_tmp_path("b64")
  img:saveAs(tmp)
  local f = io.open(tmp, "rb")
  if not f then return nil end
  local data = f:read("*a")
  f:close()
  os.remove(tmp)
  return base64_encode(data)
end

local function build_post_process()
  local pp = {
    pixelate = {
      enabled = dlg.data.pixelate,
      target_size = dlg.data.pixel_size
    },
    quantize_method = dlg.data.quantize_method,
    quantize_colors = dlg.data.colors,
    dither = dlg.data.dither,
    palette = { mode = dlg.data.palette_mode },
    remove_bg = dlg.data.remove_bg
  }
  if dlg.data.palette_mode == "preset" then
    pp.palette.name = dlg.data.palette_name
  elseif dlg.data.palette_mode == "custom" then
    local hex_str = dlg.data.palette_custom_colors or ""
    local colors = {}
    for hex in hex_str:gmatch("#?(%x%x%x%x%x%x)") do
      colors[#colors + 1] = "#" .. hex
    end
    if #colors > 0 then pp.palette.colors = colors end
  end
  return pp
end

-- ─── SETTINGS PERSISTENCE ────────────────────────────────────

save_settings = function()
  if not dlg then return end
  local s = {
    server_url = dlg.data.server_url,
    prompt = dlg.data.prompt,
    negative_prompt = dlg.data.negative_prompt,
    mode = dlg.data.mode,
    output_size = dlg.data.output_size,
    seed = dlg.data.seed,
    steps = dlg.data.steps,
    cfg_scale = dlg.data.cfg_scale,
    clip_skip = dlg.data.clip_skip,
    denoise = dlg.data.denoise,
    lora_name = dlg.data.lora_name,
    lora_weight = dlg.data.lora_weight,
    use_neg_ti = dlg.data.use_neg_ti,
    neg_ti_weight = dlg.data.neg_ti_weight,
    pixelate = dlg.data.pixelate,
    pixel_size = dlg.data.pixel_size,
    colors = dlg.data.colors,
    quantize_method = dlg.data.quantize_method,
    dither = dlg.data.dither,
    palette_mode = dlg.data.palette_mode,
    remove_bg = dlg.data.remove_bg,
    anim_method = dlg.data.anim_method,
    anim_frames = dlg.data.anim_frames,
    anim_duration = dlg.data.anim_duration,
    anim_denoise = dlg.data.anim_denoise,
    anim_seed_strategy = dlg.data.anim_seed_strategy,
    live_strength = dlg.data.live_strength,
    live_steps = dlg.data.live_steps,
    live_cfg = dlg.data.live_cfg,
    live_opacity = dlg.data.live_opacity,
    preset_name = dlg.data.preset_name,
  }
  local ok, encoded = pcall(json.encode, s)
  if not ok then return end
  local f = io.open(SETTINGS_FILE, "w")
  if f then
    f:write(encoded)
    f:close()
  end
end

load_settings = function()
  local f = io.open(SETTINGS_FILE, "r")
  if not f then return nil end
  local data = f:read("*a")
  f:close()
  local ok, s = pcall(json.decode, data)
  if not ok or type(s) ~= "table" then return nil end
  return s
end

local function apply_settings(s)
  if not s or not dlg then return end
  if s.server_url then dlg:modify{ id = "server_url", text = s.server_url } end
  if s.prompt then dlg:modify{ id = "prompt", text = s.prompt } end
  if s.negative_prompt then dlg:modify{ id = "negative_prompt", text = s.negative_prompt } end
  if s.mode then dlg:modify{ id = "mode", option = s.mode } end
  if s.output_size then dlg:modify{ id = "output_size", option = s.output_size } end
  if s.seed then dlg:modify{ id = "seed", text = s.seed } end
  if s.steps then dlg:modify{ id = "steps", value = s.steps } end
  if s.cfg_scale then dlg:modify{ id = "cfg_scale", value = s.cfg_scale } end
  if s.clip_skip then dlg:modify{ id = "clip_skip", value = s.clip_skip } end
  if s.denoise then dlg:modify{ id = "denoise", value = s.denoise } end
  if s.lora_weight then dlg:modify{ id = "lora_weight", value = s.lora_weight } end
  if s.use_neg_ti ~= nil then dlg:modify{ id = "use_neg_ti", selected = s.use_neg_ti } end
  if s.neg_ti_weight then dlg:modify{ id = "neg_ti_weight", value = s.neg_ti_weight } end
  if s.pixelate ~= nil then dlg:modify{ id = "pixelate", selected = s.pixelate } end
  if s.pixel_size then dlg:modify{ id = "pixel_size", value = s.pixel_size } end
  if s.colors then dlg:modify{ id = "colors", value = s.colors } end
  if s.quantize_method then dlg:modify{ id = "quantize_method", option = s.quantize_method } end
  if s.dither then dlg:modify{ id = "dither", option = s.dither } end
  if s.palette_mode then dlg:modify{ id = "palette_mode", option = s.palette_mode } end
  if s.remove_bg ~= nil then dlg:modify{ id = "remove_bg", selected = s.remove_bg } end
  if s.anim_method then dlg:modify{ id = "anim_method", option = s.anim_method } end
  if s.anim_frames then dlg:modify{ id = "anim_frames", value = s.anim_frames } end
  if s.anim_duration then dlg:modify{ id = "anim_duration", value = s.anim_duration } end
  if s.anim_denoise then dlg:modify{ id = "anim_denoise", value = s.anim_denoise } end
  if s.anim_seed_strategy then dlg:modify{ id = "anim_seed_strategy", option = s.anim_seed_strategy } end
  if s.live_strength then dlg:modify{ id = "live_strength", value = s.live_strength } end
  if s.live_steps then dlg:modify{ id = "live_steps", value = s.live_steps } end
  if s.live_cfg then dlg:modify{ id = "live_cfg", value = s.live_cfg } end
  if s.live_opacity then dlg:modify{ id = "live_opacity", value = s.live_opacity } end
  if s.preset_name then dlg:modify{ id = "preset_name", option = s.preset_name } end
end

-- ─── LIVE PAINT ROI HELPERS ─────────────────────────────────

local function detect_dirty_region(prev, curr)
  local w, h = curr.width, curr.height
  local min_x, min_y = w, h
  local max_x, max_y = 0, 0
  local step = math.max(1, math.floor(math.min(w, h) / 64))
  local found = false
  for y = 0, h - 1, step do
    for x = 0, w - 1, step do
      if prev:getPixel(x, y) ~= curr:getPixel(x, y) then
        found = true
        if x < min_x then min_x = x end
        if y < min_y then min_y = y end
        if x > max_x then max_x = x end
        if y > max_y then max_y = y end
      end
    end
  end
  if not found then return nil end
  min_x = math.max(0, min_x - step)
  min_y = math.max(0, min_y - step)
  max_x = math.min(w - 1, max_x + step)
  max_y = math.min(h - 1, max_y + step)
  return { x = min_x, y = min_y, w = max_x - min_x + 1, h = max_y - min_y + 1 }
end

local function generate_dirty_mask(prev, curr, roi)
  local mask = Image(curr.width, curr.height, ColorMode.GRAY)
  mask:clear(Color{ gray = 0 })
  for y = roi.y, roi.y + roi.h - 1 do
    for x = roi.x, roi.x + roi.w - 1 do
      if x < curr.width and y < curr.height then
        if prev:getPixel(x, y) ~= curr:getPixel(x, y) then
          mask:drawPixel(x, y, Color{ gray = 255 })
        end
      end
    end
  end
  return mask
end

-- ─── WEBSOCKET ───────────────────────────────────────────────

local function update_status(text)
  if dlg then dlg:modify{ id = "status", text = text } end
end

local function set_connected(state)
  connected = state
  if state then start_heartbeat() else stop_heartbeat() end
  if not dlg then return end
  if state then
    dlg:modify{ id = "connect_btn", text = "Disconnect" }
    dlg:modify{ id = "generate_btn", enabled = true }
    dlg:modify{ id = "animate_btn", enabled = true }
    dlg:modify{ id = "live_btn", enabled = true }
  else
    dlg:modify{ id = "connect_btn", text = "Connect" }
    dlg:modify{ id = "generate_btn", enabled = false }
    dlg:modify{ id = "cancel_btn", enabled = false }
    dlg:modify{ id = "animate_btn", enabled = false }
    dlg:modify{ id = "live_btn", enabled = false }
    dlg:modify{ id = "live_btn", text = "START LIVE" }
    dlg:modify{ id = "live_accept_btn", visible = false }
    if generating then generating = false end
    if animating then animating = false end
    stop_live_timer()
    stop_gen_timeout()
    live_mode = false
    live_request_inflight = false
    loop_mode = false  -- H: reset loop mode on disconnect
    cancel_pending = false  -- H10: reset cancel debounce on disconnect
  end
end

local function stop_connect_timer()
  if connect_timer then
    if connect_timer.isRunning then connect_timer:stop() end
    connect_timer = nil
  end
end

stop_heartbeat = function()
  if heartbeat_timer then
    if heartbeat_timer.isRunning then heartbeat_timer:stop() end
    heartbeat_timer = nil
  end
end

start_heartbeat = function()
  stop_heartbeat()
  heartbeat_timer = Timer{
    interval = 30.0,
    ontick = function()
      if connected and ws and not generating and not animating and not live_mode then
        pcall(function() ws:sendText('{"action":"ping"}') end)
      end
    end,
  }
  heartbeat_timer:start()
end

stop_gen_timeout = function()
  if gen_timeout_timer then
    if gen_timeout_timer.isRunning then gen_timeout_timer:stop() end
    gen_timeout_timer = nil
  end
end

start_gen_timeout = function()
  stop_gen_timeout()
  gen_timeout_timer = Timer{
    interval = GEN_TIMEOUT_SECONDS,
    ontick = function()
      stop_gen_timeout()
      if generating or animating then
        generating = false
        animating = false
        gen_step_start = nil
        if dlg then
          update_status("Timed out — no response from server")
          dlg:modify{ id = "generate_btn", enabled = not live_mode }
          dlg:modify{ id = "animate_btn", enabled = not live_mode }
          dlg:modify{ id = "cancel_btn", enabled = false }
        end
      end
    end,
  }
  gen_timeout_timer:start()
end

local function connect()
  if ws then pcall(function() ws:close() end); ws = nil end
  update_status("Connecting...")
  ws = WebSocket{
    url = SERVER_URL,
    onreceive = function(msg_type, data)
      if msg_type == WebSocketMessageType.OPEN then
        stop_connect_timer()
        set_connected(true)
        update_status("Connected")
        pcall(function() ws:sendText(json.encode({ action = "ping" })) end)
        -- H13: auto re-request resources on reconnect
        if resources_requested then
          send({ action = "list_loras" })
          send({ action = "list_palettes" })
          send({ action = "list_embeddings" })
          send({ action = "list_presets" })
        end
        return
      end
      if msg_type == WebSocketMessageType.CLOSE then
        set_connected(false)
        resources_requested = false
        update_status("Disconnected (server closed)")
        ws = nil
        return
      end
      if msg_type == WebSocketMessageType.TEXT then
        -- #17: removed redundant set_connected(true) — already called in OPEN handler
        local ok, response = pcall(json.decode, data)
        if not ok then
          update_status("JSON error: " .. tostring(response))
          return
        end
        local hok, herr = pcall(handle_response, response)
        if not hok then update_status("Error: " .. tostring(herr)) end
      end
    end,
    deflate = false,
  }
  ws:connect()

  -- Connection timeout
  stop_connect_timer()
  connect_timer = Timer{
    interval = 5.0,
    ontick = function()
      stop_connect_timer()
      if not connected then
        if ws then pcall(function() ws:close() end); ws = nil end
        update_status("Connection failed - is the server running?")
      end
    end,
  }
  connect_timer:start()
end

local function disconnect()
  stop_connect_timer()
  if ws then pcall(function() ws:close() end); ws = nil end
  set_connected(false)
  resources_requested = false
  anim_layer = nil
  anim_start_frame = 0
  anim_frame_count = 0
  anim_base_seed = 0
  generating = false
  animating = false
  stop_live_timer()
  stop_gen_timeout()
  live_mode = false
  live_request_inflight = false
  live_canvas_hash = nil
  live_preview_layer = nil
  live_preview_sprite = nil
  update_status("Disconnected")
end

local function send(payload)
  if not connected or ws == nil then
    update_status("Not connected")
    return false
  end
  local ok, err = pcall(function() ws:sendText(json.encode(payload)) end)
  if not ok then
    update_status("Send failed: " .. tostring(err))
    return false
  end
  return true
end

request_resources = function()
  resources_requested = true
  send({ action = "list_palettes" })
  send({ action = "list_loras" })
  send({ action = "list_embeddings" })
  send({ action = "list_presets" })
end

-- ─── RESPONSE HANDLER ────────────────────────────────────────

handle_response = function(resp)
  if resp.type == "progress" then
    if not resp.total or resp.total <= 0 then return end
    if not dlg then return end
    local pct = math.floor((resp.step / resp.total) * 100)
    local eta_str = ""
    local now = os.clock()
    if gen_step_start and resp.step > 1 then
      local elapsed = now - gen_step_start
      local steps_done = resp.step - 1
      if steps_done > 0 then
        local remaining = (elapsed / steps_done) * (resp.total - resp.step)
        if remaining < 60 then
          eta_str = string.format(" ~%.0fs", remaining)
        else
          eta_str = string.format(" ~%.1fmin", remaining / 60)
        end
      end
    end
    local frame_ctx = ""
    if resp.frame_index ~= nil and resp.total_frames ~= nil then
      frame_ctx = " [F" .. (resp.frame_index + 1) .. "/" .. resp.total_frames .. "]"
    end
    update_status(resp.step .. "/" .. resp.total .. " (" .. pct .. "%)" .. frame_ctx .. eta_str)

  elseif resp.type == "result" then
    generating = false
    gen_step_start = nil
    stop_gen_timeout()
    cancel_pending = false  -- H10: reset cancel debounce on result
    -- C8: validate critical fields
    if not resp.image then
      update_status("Error: missing image in result response")
      if dlg then
        dlg:modify{ id = "generate_btn", enabled = true }
        dlg:modify{ id = "cancel_btn", enabled = false }
      end
      return
    end
    if not resp.seed then resp.seed = 0 end
    import_result(resp)
    -- Loop mode: schedule next generation
    if loop_mode and dlg then
      local seed_info = tostring(resp.seed or "?")
      update_status("Loop #" .. loop_counter .. " done (seed=" .. seed_info .. ") — next...")
      -- Adjust seed for next iteration
      if loop_seed_mode == "increment" and resp.seed then
        dlg:modify{ id = "seed", text = tostring(resp.seed + 1) }
      else
        dlg:modify{ id = "seed", text = "-1" }
      end
      -- Small delay then trigger next generation
      local loop_timer = Timer{
        interval = 0.1,
        ontick = function(t)
          t:stop()
          if loop_mode and dlg and connected and not generating then
            -- Re-trigger the generate button click
            local gw, gh = parse_size()
            local req = {
              action = "generate",
              prompt = dlg.data.prompt,
              negative_prompt = dlg.data.negative_prompt,
              mode = dlg.data.mode,
              width = gw, height = gh,
              seed = parse_seed(),
              steps = dlg.data.steps,
              cfg_scale = dlg.data.cfg_scale / 10.0,
              clip_skip = dlg.data.clip_skip,
              denoise_strength = dlg.data.denoise / 100.0,
              post_process = build_post_process(),
            }
            attach_lora(req)
            attach_neg_ti(req)
            if not attach_source_image(req) then loop_mode = false; return end
            generating = true
            gen_step_start = os.clock()
            start_gen_timeout()
            loop_counter = loop_counter + 1
            dlg:modify{ id = "generate_btn", enabled = false }
            dlg:modify{ id = "cancel_btn", enabled = true }
            update_status("Loop #" .. loop_counter .. " — Generating...")
            send(req)
          end
        end,
      }
      loop_timer:start()
    elseif dlg then
      update_status("Done (" .. tostring(resp.time_ms or "?") .. "ms, seed=" .. tostring(resp.seed or "?") .. ")")
      dlg:modify{ id = "generate_btn", enabled = true }
      dlg:modify{ id = "cancel_btn", enabled = false }
    end

  elseif resp.type == "animation_frame" then
    -- C8: validate critical fields
    if not resp.image then
      update_status("Error: missing image in animation_frame response")
      return
    end
    if resp.frame_index ~= nil then
      import_animation_frame(resp)
    end

  elseif resp.type == "animation_complete" then
    animating = false
    gen_step_start = nil
    stop_gen_timeout()
    -- #13: validate frame count matches expected total
    if resp.total_frames and anim_frame_count ~= resp.total_frames then
      update_status("Warning: received " .. anim_frame_count .. "/" .. resp.total_frames .. " frames")
    end
    if dlg then
      local tag_str = ""
      if resp.tag_name and resp.tag_name ~= "" then tag_str = ", tag=" .. resp.tag_name end
      update_status("Animation done (" .. tostring(resp.total_frames or "?") .. " frames, "
        .. tostring(resp.total_time_ms or "?") .. "ms" .. tag_str .. ")")
      dlg:modify{ id = "animate_btn", enabled = true }
      dlg:modify{ id = "cancel_btn", enabled = false }
    end

    local spr = app.sprite
    if spr and anim_frame_count > 0 then
      local dur = (dlg and dlg.data.anim_duration or 100) / 1000.0
      for i = 0, anim_frame_count - 1 do
        local fn = anim_start_frame + i
        if spr.frames[fn] then spr.frames[fn].duration = dur end
      end
      local tag_start = anim_start_frame
      local tag_end = anim_start_frame + anim_frame_count - 1
      if resp.tag_name and resp.tag_name ~= "" and spr.frames[tag_start] and spr.frames[tag_end] then
        local tag = spr:newTag(tag_start, tag_end)
        tag.name = resp.tag_name
      end
      app.refresh()
    end
    anim_layer = nil
    anim_start_frame = 0
    anim_frame_count = 0
    anim_base_seed = 0

  elseif resp.type == "error" then
    local was_animating = animating
    generating = false
    animating = false
    loop_mode = false
    gen_step_start = nil
    stop_gen_timeout()

    if was_animating and anim_frame_count > 0 then
      local spr = app.sprite
      if spr then
        local dur = (dlg and dlg.data.anim_duration or 100) / 1000.0
        for i = 0, anim_frame_count - 1 do
          local fn = anim_start_frame + i
          if spr.frames[fn] then spr.frames[fn].duration = dur end
        end
      end
      anim_layer = nil
      anim_start_frame = 0
      anim_frame_count = 0
      anim_base_seed = 0
    end

    if dlg then
      update_status("Error: " .. tostring(resp.message or "Unknown"))
      -- H10: reset cancel debounce and re-enable generate
      cancel_pending = false
      dlg:modify{ id = "generate_btn", enabled = not live_mode }
      dlg:modify{ id = "animate_btn", enabled = not live_mode }
      dlg:modify{ id = "cancel_btn", enabled = false }
      live_request_inflight = false
    end
    if resp.code ~= "CANCELLED" then
      app.alert("PixyToon: " .. tostring(resp.message or "Unknown error"))
    end

  elseif resp.type == "list" then
    local lt = resp.list_type or ""
    local items = resp.items or {}
    if lt == "palettes" then
      available_palettes = items
      if dlg and #items > 0 then
        local opts = {}
        for _, n in ipairs(items) do opts[#opts + 1] = n end
        dlg:modify{ id = "palette_name", options = opts }
      end
    elseif lt == "loras" then
      available_loras = items
      if dlg then
        local opts = { "(default)" }
        for _, n in ipairs(items) do opts[#opts + 1] = n end
        dlg:modify{ id = "lora_name", options = opts }
      end
    elseif lt == "embeddings" then
      available_embeddings = items
    elseif lt == "presets" then
      available_presets = items
      if dlg then
        local opts = { "(none)" }
        for _, n in ipairs(items) do opts[#opts + 1] = n end
        dlg:modify{ id = "preset_name", options = opts }
      end
    end
    local total = #available_palettes + #available_loras + #available_embeddings
    if total > 0 then
      update_status("Resources loaded (" .. #available_loras .. " LoRAs, "
        .. #available_palettes .. " palettes, " .. #available_embeddings .. " embeddings)")
    else
      update_status("Connected (no resources found)")
    end

  elseif resp.type == "realtime_ready" then
    live_mode = true
    live_request_inflight = false
    live_last_prompt = dlg and dlg.data.prompt or nil
    if dlg then
      update_status("Live mode active")
      dlg:modify{ id = "live_btn", text = "STOP LIVE" }
      dlg:modify{ id = "live_accept_btn", visible = true }
      dlg:modify{ id = "generate_btn", enabled = false }
      dlg:modify{ id = "animate_btn", enabled = false }
    end
    start_live_timer()

  elseif resp.type == "realtime_result" then
    live_request_inflight = false
    -- C8: validate critical fields
    if not resp.image then
      update_status("Error: missing image in realtime_result")
      return
    end
    if live_mode then
      live_update_preview(resp)
      if dlg then
        update_status("Live (" .. tostring(resp.latency_ms or "?") .. "ms)")
      end
    end

  elseif resp.type == "realtime_stopped" then
    stop_live_timer()
    live_mode = false
    live_request_inflight = false
    live_canvas_hash = nil
    -- Clean up preview layer
    if live_preview_layer then
      local spr = app.sprite
      if spr then
        pcall(function()
          local cel = live_preview_layer:cel(app.frame)
          if cel then spr:deleteCel(cel) end
          spr:deleteLayer(live_preview_layer)
        end)
      end
      live_preview_layer = nil
      live_preview_sprite = nil
      pcall(app.refresh)
    end
    if dlg then
      update_status("Live mode stopped")
      dlg:modify{ id = "live_btn", text = "START LIVE" }
      dlg:modify{ id = "live_accept_btn", visible = false }
      dlg:modify{ id = "generate_btn", enabled = true }
      dlg:modify{ id = "animate_btn", enabled = true }
    end

  elseif resp.type == "pong" then
    if not connected then set_connected(true) end
    if not resources_requested then request_resources() end
    update_status("Connected")

  elseif resp.type == "prompt_result" then
    if dlg and resp.prompt then
      dlg:modify{ id = "prompt", text = resp.prompt }
      update_status("Prompt generated")
    end

  elseif resp.type == "preset" then
    if dlg and resp.data then
      local d = resp.data
      if d.prompt_prefix then dlg:modify{ id = "prompt", text = d.prompt_prefix } end
      if d.negative_prompt then dlg:modify{ id = "negative_prompt", text = d.negative_prompt } end
      if d.mode then dlg:modify{ id = "mode", option = d.mode } end
      if d.width and d.height then
        dlg:modify{ id = "output_size", option = d.width .. "x" .. d.height }
      end
      if d.steps then dlg:modify{ id = "steps", value = d.steps } end
      if d.cfg_scale then dlg:modify{ id = "cfg_scale", value = math.floor(d.cfg_scale * 10) } end
      if d.clip_skip then dlg:modify{ id = "clip_skip", value = d.clip_skip } end
      if d.denoise_strength then dlg:modify{ id = "denoise", value = math.floor(d.denoise_strength * 100) } end
      if d.post_process then
        local pp = d.post_process
        if pp.pixelate ~= nil then
          local px = pp.pixelate
          if type(px) == "table" then
            if px.enabled ~= nil then dlg:modify{ id = "pixelate", selected = px.enabled } end
            if px.target_size then dlg:modify{ id = "pixel_size", value = px.target_size } end
          else
            dlg:modify{ id = "pixelate", selected = px }
          end
        end
        if pp.quantize_colors then dlg:modify{ id = "colors", value = pp.quantize_colors } end
        if pp.quantize_method then dlg:modify{ id = "quantize_method", option = pp.quantize_method } end
        if pp.dither then dlg:modify{ id = "dither", option = pp.dither } end
      end
      update_status("Preset '" .. tostring(resp.name or "?") .. "' loaded")
    end

  elseif resp.type == "preset_saved" then
    if dlg then
      update_status("Preset '" .. tostring(resp.name or "?") .. "' saved")
      send({ action = "list_presets" })
    end

  elseif resp.type == "preset_deleted" then
    if dlg then
      update_status("Preset '" .. tostring(resp.name or "?") .. "' deleted")
      dlg:modify{ id = "preset_name", option = "(none)" }
      send({ action = "list_presets" })
    end

  elseif resp.type == "cleanup_done" then
    if dlg then
      update_status(tostring(resp.message or "Cleanup done") .. " (freed " .. string.format("%.1f", resp.freed_mb or 0) .. " MB)")
    end
  end
end

-- ─── IMPORT RESULT ───────────────────────────────────────────

import_result = function(resp)
  local img_data = base64_decode(resp.image)
  local tmp = make_tmp_path("res")

  local ok, err = pcall(function()
    local f = io.open(tmp, "wb")
    if not f then error("Failed to create temp file") end
    f:write(img_data)
    f:close()

    local spr = app.sprite
    if spr == nil then
      spr = Sprite(resp.width or 512, resp.height or 512, ColorMode.RGB)
    end

    local layer = spr:newLayer()
    layer.name = "PixyToon #" .. tostring(resp.seed or "?")

    local img = Image{ fromFile = tmp }
    if img then spr:newCel(layer, app.frame, img, Point(0, 0)) end

    os.remove(tmp)
    app.refresh()
  end)
  if not ok then
    pcall(os.remove, tmp)
    update_status("Import error: " .. tostring(err))
  end
end

-- ─── IMPORT ANIMATION FRAME ─────────────────────────────────

import_animation_frame = function(resp)
  if not animating then return end
  if resp.frame_index ~= 0 and anim_layer == nil then return end

  local img_data = base64_decode(resp.image)
  local tmp = make_tmp_path("anim")

  local ok, err = pcall(function()
    local f = io.open(tmp, "wb")
    if not f then error("Failed to open temp file for writing") end
    f:write(img_data)
    f:close()

    local spr = app.sprite
    local created_sprite = false
    if spr == nil then
      spr = Sprite(resp.width or 512, resp.height or 512, ColorMode.RGB)
      created_sprite = true
    end

    -- First frame: create layer and anchor position
    if resp.frame_index == 0 then
      anim_layer = spr:newLayer()
      anim_layer.name = "PixyToon Anim #" .. tostring(resp.seed or "?")
      anim_base_seed = resp.seed or 0
      anim_frame_count = 0
      if created_sprite then
        anim_start_frame = 1  -- reuse the initial empty frame
      else
        anim_start_frame = #spr.frames + 1  -- append after existing content
      end
    end

    -- H9: Determine frame position
    -- If frame_index==0 and we just created the sprite, reuse the first frame (no need to insert).
    -- Otherwise, insert a new empty frame at the correct position (clamped to valid range).
    local frame_num
    if resp.frame_index == 0 and created_sprite then
      frame_num = 1  -- Reuse first frame of new sprite
    else
      -- Insert new frame at correct position
      local target_pos = anim_start_frame + resp.frame_index
      target_pos = math.min(target_pos, #spr.frames + 1)
      local new_frame = spr:newEmptyFrame(target_pos)
      frame_num = new_frame.frameNumber
    end

    local img = Image{ fromFile = tmp }
    if img and anim_layer and spr.frames[frame_num] then
      spr:newCel(anim_layer, spr.frames[frame_num], img, Point(0, 0))
    end

    anim_frame_count = anim_frame_count + 1
    os.remove(tmp)
    app.refresh()

    if dlg then
      update_status("Frame " .. (resp.frame_index + 1) .. "/" .. tostring(resp.total_frames or "?")
        .. " (" .. tostring(resp.time_ms or "?") .. "ms)")
    end
  end)
  if not ok then
    pcall(os.remove, tmp)
    update_status("Import error: " .. tostring(err))
  end
end

-- ─── CAPTURE FUNCTIONS ──────────────────────────────────────

local function capture_active_layer()
  local spr = app.sprite
  if spr == nil then return nil end
  local cel = app.cel
  if cel == nil or cel.image == nil then return nil end
  local full = Image(spr.spec)
  full:clear()
  full:drawImage(cel.image, cel.position)
  return image_to_base64(full)
end

local function capture_flattened()
  local spr = app.sprite
  if spr == nil then return nil end
  local flat_img = Image(spr.spec)
  flat_img:drawSprite(spr, app.frame)
  return image_to_base64(flat_img)
end

-- Capture an inpainting mask from the current sprite.
-- Priority order:
--   1) Active selection (marquee/lasso) -> white where selected
--   2) Layer named "Mask"/"mask"        -> grayscale content of that layer
--   3) Active layer alpha channel       -> white where alpha > 0
local function capture_mask()
  local spr = app.sprite
  if spr == nil then return nil end

  -- Strategy A: active selection
  local sel = spr.selection
  if sel and not sel.isEmpty then
    local mask_img = Image(spr.width, spr.height, ColorMode.GRAY)
    mask_img:clear(Color{ gray = 0 })
    for y = sel.bounds.y, sel.bounds.y + sel.bounds.height - 1 do
      for x = sel.bounds.x, sel.bounds.x + sel.bounds.width - 1 do
        if sel:contains(x, y) then
          mask_img:drawPixel(x, y, Color{ gray = 255 })
        end
      end
    end
    return image_to_base64(mask_img)
  end

  -- Strategy B: "Mask" layer (with correct positioning)
  local function find_mask_layer(layers)
    for _, layer in ipairs(layers) do
      if layer.name == "Mask" or layer.name == "mask" then
        local cel = layer:cel(app.frame)
        if cel and cel.image then
          local full = Image(spr.width, spr.height, ColorMode.GRAY)
          full:clear(Color{ gray = 0 })
          full:drawImage(cel.image, cel.position)
          return image_to_base64(full)
        end
      end
      if layer.isGroup and layer.layers then
        local result = find_mask_layer(layer.layers)
        if result then return result end
      end
    end
    return nil
  end
  local mask_b64 = find_mask_layer(spr.layers)
  if mask_b64 then return mask_b64 end

  -- Strategy C: auto from active layer alpha
  local cel = app.cel
  if cel and cel.image then
    local img = cel.image
    local mask_img = Image(spr.width, spr.height, ColorMode.GRAY)
    mask_img:clear(Color{ gray = 0 })
    local ox, oy = cel.position.x, cel.position.y
    if spr.colorMode == ColorMode.RGB then
      for y = 0, img.height - 1 do
        for x = 0, img.width - 1 do
          local px = img:getPixel(x, y)
          local a = app.pixelColor.rgbaA(px)
          if a > 0 then
            local sx, sy = ox + x, oy + y
            if sx >= 0 and sx < spr.width and sy >= 0 and sy < spr.height then
              mask_img:drawPixel(sx, sy, Color{ gray = 255 })
            end
          end
        end
      end
    else
      -- For non-RGB modes (indexed, grayscale), draw through sprite rendering
      local tmp_img = Image(spr.spec)
      tmp_img:clear()
      tmp_img:drawImage(cel.image, cel.position)
      for y = 0, spr.height - 1 do
        for x = 0, spr.width - 1 do
          local px = tmp_img:getPixel(x, y)
          if px ~= 0 then
            mask_img:drawPixel(x, y, Color{ gray = 255 })
          end
        end
      end
    end
    return image_to_base64(mask_img)
  end

  return nil
end

-- ─── LIVE PAINT HELPERS ─────────────────────────────────────

local function canvas_hash(img)
  local w, h = img.width, img.height
  local hash = 0
  local step = math.max(1, math.floor(math.min(w, h) / 32))
  for y = 0, h - 1, step do
    for x = 0, w - 1, step do
      hash = (hash * 31 + img:getPixel(x, y)) % 2147483647
    end
  end
  return hash
end

stop_live_timer = function()
  if live_timer then
    if live_timer.isRunning then live_timer:stop() end
    live_timer = nil
  end
  -- H11: reset inflight flag when timer is stopped
  live_request_inflight = false
  -- C6: stop cooldown timer on live stop
  if live_cooldown_timer then
    live_cooldown_timer:stop()
    live_cooldown_timer = nil
  end
end

start_live_timer = function()
  stop_live_timer()
  live_prev_canvas = nil  -- C7: release previous reference for GC
  live_stroke_cooldown = nil
  -- C6: stop any lingering cooldown timer
  if live_cooldown_timer then
    live_cooldown_timer:stop()
    live_cooldown_timer = nil
  end
  live_timer = Timer{
    interval = 0.15,
    ontick = function()
      if not live_mode or not connected or live_request_inflight then return end
      local spr = app.sprite
      if spr == nil then
        send({ action = "realtime_stop" })
        stop_live_timer()
        live_mode = false
        live_request_inflight = false
        live_preview_layer = nil
        live_preview_sprite = nil
        if dlg then
          update_status("Live stopped (sprite closed)")
          dlg:modify{ id = "live_btn", text = "START LIVE" }
          dlg:modify{ id = "live_accept_btn", visible = false }
          dlg:modify{ id = "generate_btn", enabled = true }
          dlg:modify{ id = "animate_btn", enabled = true }
        end
        return
      end

      -- Hide preview layer for capture
      local was_visible = true
      if live_preview_layer then
        was_visible = live_preview_layer.isVisible
        live_preview_layer.isVisible = false
      end
      local flat_img = Image(spr.spec)
      flat_img:drawSprite(spr, app.frame)
      if live_preview_layer then
        live_preview_layer.isVisible = was_visible
      end

      -- Check if canvas changed
      local hash = canvas_hash(flat_img)
      if hash == live_canvas_hash then return end
      live_canvas_hash = hash

      -- Debounce: wait 200ms after last change before sending
      live_stroke_cooldown = os.clock()
      -- C6: reuse a single cooldown timer to avoid memory leak
      if live_cooldown_timer then
        live_cooldown_timer:stop()
      end
      live_cooldown_timer = Timer{
        interval = 0.2,
        ontick = function(t)
          t:stop()
          if not live_mode or live_request_inflight then return end
          if live_stroke_cooldown and (os.clock() - live_stroke_cooldown) < 0.18 then return end

          -- Auto-detect prompt changes
          if dlg then
            local current_prompt = dlg.data.prompt
            if current_prompt ~= live_last_prompt then
              live_last_prompt = current_prompt
              send({ action = "realtime_update", prompt = current_prompt })
            end
          end

          -- Recapture after debounce
          if not spr or not app.sprite then return end
          local vis2 = true
          if live_preview_layer then
            vis2 = live_preview_layer.isVisible
            live_preview_layer.isVisible = false
          end
          local curr_img = Image(spr.spec)
          curr_img:drawSprite(spr, app.frame)
          if live_preview_layer then
            live_preview_layer.isVisible = vis2
          end

          -- ROI detection
          local roi = nil
          local mask_b64 = nil
          if live_prev_canvas and live_prev_canvas.width == curr_img.width and live_prev_canvas.height == curr_img.height then
            roi = detect_dirty_region(live_prev_canvas, curr_img)
            if roi then
              local mask_img = generate_dirty_mask(live_prev_canvas, curr_img, roi)
              mask_b64 = image_to_base64(mask_img)
            end
          end
          -- C7: release previous reference for GC before cloning
          live_prev_canvas = nil
          live_prev_canvas = curr_img:clone()

          -- Send frame with ROI data
          local b64 = image_to_base64(curr_img)
          if not b64 then return end
          live_frame_id = live_frame_id + 1
          local payload = {
            action = "realtime_frame",
            image = b64,
            frame_id = live_frame_id,
          }
          if roi then
            payload.roi_x = roi.x
            payload.roi_y = roi.y
            payload.roi_w = roi.w
            payload.roi_h = roi.h
            if mask_b64 then payload.mask = mask_b64 end
          end
          local sent = send(payload)
          if sent then
            live_request_inflight = true
          end
        end,
      }
      live_cooldown_timer:start()
    end,
  }
  live_timer:start()
end

live_update_preview = function(resp)
  local spr = app.sprite
  if spr == nil then return end

  -- Find or create preview layer
  if live_preview_layer == nil or live_preview_sprite ~= spr or not pcall(function() return live_preview_layer.name end) then
    live_preview_layer = nil
    for _, layer in ipairs(spr.layers) do
      if layer.name == "_pixytoon_live" then
        live_preview_layer = layer
        break
      end
    end
    if live_preview_layer == nil then
      live_preview_layer = spr:newLayer()
      live_preview_layer.name = "_pixytoon_live"
    end
    live_preview_sprite = spr
  end

  local img_data = base64_decode(resp.image)
  local tmp = make_tmp_path("live")
  local f = io.open(tmp, "wb")
  if not f then return end
  f:write(img_data)
  f:close()

  -- H12: pcall to safely load the image from temp file
  local ok_img, img = pcall(function() return Image{ fromFile = tmp } end)
  os.remove(tmp)
  if not ok_img or not img then
    update_status("Preview load failed")
    return
  end

  -- Update existing cel in-place (avoids layer churn)
  local cel = live_preview_layer:cel(app.frame)
  if cel then
    cel.image = img
    cel.position = Point(0, 0)
  else
    spr:newCel(live_preview_layer, app.frame, img, Point(0, 0))
  end

  -- Apply opacity
  if dlg then
    live_preview_layer.opacity = math.floor(dlg.data.live_opacity * 255 / 100)
  end

  app.refresh()
end

-- ─── SHARED REQUEST BUILDERS ─────────────────────────────────

local function parse_size()
  local s = dlg.data.output_size
  local w, h = s:match("(%d+)x(%d+)")
  return tonumber(w), tonumber(h)
end

local function parse_seed()
  local v = tonumber(dlg.data.seed) or -1
  if v ~= math.floor(v) then v = -1 end
  return v
end

local function attach_lora(req)
  local sel = dlg.data.lora_name
  if sel and sel ~= "(default)" then
    req.lora = { name = sel, weight = dlg.data.lora_weight / 100.0 }
  end
end

local function attach_neg_ti(req)
  if dlg.data.use_neg_ti and #available_embeddings > 0 then
    local ti_list = {}
    local w = dlg.data.neg_ti_weight / 100.0
    for _, name in ipairs(available_embeddings) do
      ti_list[#ti_list + 1] = { name = name, weight = w }
    end
    req.negative_ti = ti_list
  end
end

local function attach_source_image(req)
  if req.mode == "img2img" or req.mode:find("controlnet_") then
    local b64 = capture_active_layer()
    if not b64 then
      app.alert("No active layer to use as source.")
      return false
    end
    if req.mode == "img2img" then req.source_image = b64
    else req.control_image = b64 end
  end
  if req.mode == "inpaint" then
    local src = capture_flattened()
    if not src then
      app.alert("Inpaint requires an open sprite.")
      return false
    end
    req.source_image = src
    local mask = capture_mask()
    if not mask then
      app.alert("Inpaint requires a mask.\n- Make a selection, or\n- Create a 'Mask' layer, or\n- Draw on active layer")
      return false
    end
    req.mask_image = mask
  end
  return true
end

-- ─── BUILD DIALOG ────────────────────────────────────────────

local function build_dialog()
  dlg = Dialog{
    title = "PixyToon - AI Pixel Art",
    resizeable = true,
    onclose = function() save_settings(); disconnect() end
  }

  -- ══════════════════════════════════════════════════════════
  -- CONNECTION (always visible)
  -- ══════════════════════════════════════════════════════════
  dlg:separator{ text = "Connection", hexpand = true }

  dlg:entry{
    id = "server_url",
    label = "Server",
    text = SERVER_URL,
    hexpand = true,
  }

  dlg:label{ id = "status", text = "Disconnected", hexpand = true }

  dlg:button{
    id = "connect_btn",
    text = "Connect",
    onclick = function()
      if connected then disconnect()
      else
        SERVER_URL = dlg.data.server_url or SERVER_URL
        connect()
      end
    end
  }
  dlg:button{
    id = "refresh_btn",
    text = "Refresh Resources",
    onclick = function()
      if connected then
        resources_requested = false
        request_resources()
        update_status("Refreshing...")
      else
        update_status("Not connected")
      end
    end
  }
  dlg:button{
    id = "cleanup_btn",
    text = "Cleanup GPU",
    onclick = function()
      if connected and not generating and not animating and not live_mode then
        send({ action = "cleanup" })
        update_status("Cleaning up GPU...")
      else
        update_status("Cannot cleanup during generation/live")
      end
    end
  }

  -- ══════════════════════════════════════════════════════════
  -- TAB: Generate
  -- ══════════════════════════════════════════════════════════
  dlg:tab{ id = "tab_gen", text = "Generate" }

  -- Preset selector
  dlg:combobox{
    id = "preset_name",
    label = "Preset",
    options = { "(none)" },
    option = "(none)",
    onchange = function()
      local sel = dlg.data.preset_name
      if sel and sel ~= "(none)" then
        send({ action = "get_preset", preset_name = sel })
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
      local gw, gh = parse_size()
      local preset_data = {
        prompt_prefix = dlg.data.prompt,
        negative_prompt = dlg.data.negative_prompt,
        mode = dlg.data.mode,
        width = gw, height = gh,
        steps = dlg.data.steps,
        cfg_scale = dlg.data.cfg_scale / 10.0,
        clip_skip = dlg.data.clip_skip,
        denoise_strength = dlg.data.denoise / 100.0,
        post_process = build_post_process(),
      }
      send({ action = "save_preset", preset_name = pname, preset_data = preset_data })
    end,
  }
  dlg:button{
    id = "preset_delete_btn",
    text = "Del",
    onclick = function()
      local sel = dlg.data.preset_name
      if sel and sel ~= "(none)" then
        send({ action = "delete_preset", preset_name = sel })
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
    min = -200,
    max = 200,
    value = 100,
    onchange = function()
      dlg:modify{ id = "lora_weight",
        label = string.format("LoRA (%.2f)", dlg.data.lora_weight / 100.0) }
    end,
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
      send({ action = "generate_prompt" })
      update_status("Generating prompt...")
    end,
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
    min = 10,
    max = 200,
    value = 100,
    visible = false,
    onchange = function()
      dlg:modify{ id = "neg_ti_weight",
        label = string.format("Emb. (%.2f)", dlg.data.neg_ti_weight / 100.0) }
    end,
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
    min = 0,
    max = 100,
    value = 100,
    onchange = function()
      dlg:modify{ id = "denoise",
        label = string.format("Strength (%.2f)", dlg.data.denoise / 100.0) }
    end,
  }

  dlg:slider{
    id = "steps",
    label = "Steps",
    min = 1,
    max = 100,
    value = 8,
  }

  dlg:slider{
    id = "cfg_scale",
    label = "CFG (5.0)",
    min = 0,
    max = 300,
    value = 50,
    onchange = function()
      dlg:modify{ id = "cfg_scale",
        label = string.format("CFG (%.1f)", dlg.data.cfg_scale / 10.0) }
    end,
  }

  dlg:slider{
    id = "clip_skip",
    label = "CLIP Skip",
    min = 1,
    max = 12,
    value = 2,
  }

  -- ══════════════════════════════════════════════════════════
  -- TAB: Post-Process
  -- ══════════════════════════════════════════════════════════
  dlg:tab{ id = "tab_pp", text = "Post-Process" }

  dlg:check{
    id = "pixelate",
    label = "Pixelate",
    selected = false,
  }

  dlg:slider{
    id = "pixel_size",
    label = "Target (128px)",
    min = 8,
    max = 512,
    value = 128,
    onchange = function()
      dlg:modify{ id = "pixel_size", label = "Target (" .. dlg.data.pixel_size .. "px)" }
    end,
  }

  dlg:slider{
    id = "colors",
    label = "Colors (32)",
    min = 2,
    max = 256,
    value = 32,
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

  -- ══════════════════════════════════════════════════════════
  -- TAB: Animation
  -- ══════════════════════════════════════════════════════════
  dlg:tab{ id = "tab_anim", text = "Animation" }

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
    min = 2,
    max = 120,
    value = 8,
  }

  dlg:slider{
    id = "anim_duration",
    label = "Duration (ms)",
    min = 50,
    max = 2000,
    value = 100,
  }

  dlg:slider{
    id = "anim_denoise",
    label = "Strength (0.30)",
    min = 5,
    max = 100,
    value = 30,
    onchange = function()
      dlg:modify{ id = "anim_denoise",
        label = string.format("Strength (%.2f)", dlg.data.anim_denoise / 100.0) }
    end,
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
    min = 1,
    max = 3,
    value = 2,
    visible = false,
  }

  -- ══════════════════════════════════════════════════════════
  -- TAB: Live
  -- ══════════════════════════════════════════════════════════
  dlg:tab{ id = "tab_live", text = "Live" }

  dlg:slider{
    id = "live_strength",
    label = "Strength (0.50)",
    min = 5,
    max = 95,
    value = 50,
    onchange = function()
      dlg:modify{ id = "live_strength",
        label = string.format("Strength (%.2f)", dlg.data.live_strength / 100.0) }
      if live_mode then
        send({ action = "realtime_update", denoise_strength = dlg.data.live_strength / 100.0 })
      end
    end,
  }

  dlg:slider{
    id = "live_steps",
    label = "Steps",
    min = 2,
    max = 8,
    value = 4,
    onchange = function()
      if live_mode then
        send({ action = "realtime_update", steps = dlg.data.live_steps })
      end
    end,
  }

  dlg:slider{
    id = "live_cfg",
    label = "CFG (2.5)",
    min = 10,
    max = 100,
    value = 25,
    onchange = function()
      dlg:modify{ id = "live_cfg",
        label = string.format("CFG (%.1f)", dlg.data.live_cfg / 10.0) }
      if live_mode then
        send({ action = "realtime_update", cfg_scale = dlg.data.live_cfg / 10.0 })
      end
    end,
  }

  dlg:slider{
    id = "live_opacity",
    label = "Preview (70%)",
    min = 10,
    max = 100,
    value = 70,
    onchange = function()
      dlg:modify{ id = "live_opacity",
        label = string.format("Preview (%d%%)", dlg.data.live_opacity) }
      if live_preview_layer then
        live_preview_layer.opacity = math.floor(dlg.data.live_opacity * 255 / 100)
        app.refresh()
      end
    end,
  }

  -- ── End tabs ──
  dlg:endtabs{ id = "main_tabs", selected = "tab_gen" }

  -- ══════════════════════════════════════════════════════════
  -- ACTIONS (always visible, bottom)
  -- ══════════════════════════════════════════════════════════
  dlg:separator{ text = "Actions", hexpand = true }

  dlg:check{
    id = "loop_check",
    label = "Loop Mode",
    selected = false,
    onchange = function()
      dlg:modify{ id = "loop_seed_combo", visible = dlg.data.loop_check }
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
      if generating or animating then return end
      -- Initialize loop state
      if dlg.data.loop_check then
        loop_mode = true
        loop_counter = 0
        loop_seed_mode = dlg.data.loop_seed_combo or "random"
      end
      local gw, gh = parse_size()
      local req = {
        action = "generate",
        prompt = dlg.data.prompt,
        negative_prompt = dlg.data.negative_prompt,
        mode = dlg.data.mode,
        width = gw, height = gh,
        seed = parse_seed(),
        steps = dlg.data.steps,
        cfg_scale = dlg.data.cfg_scale / 10.0,
        clip_skip = dlg.data.clip_skip,
        denoise_strength = dlg.data.denoise / 100.0,
        post_process = build_post_process(),
      }
      attach_lora(req)
      attach_neg_ti(req)
      if not attach_source_image(req) then loop_mode = false; return end

      generating = true
      gen_step_start = os.clock()
      start_gen_timeout()
      dlg:modify{ id = "generate_btn", enabled = false }
      dlg:modify{ id = "cancel_btn", enabled = true }
      if loop_mode then
        loop_counter = loop_counter + 1
        update_status("Loop #" .. loop_counter .. " — Generating...")
      else
        update_status("Generating...")
      end
      send(req)
    end,
  }

  dlg:button{
    id = "cancel_btn",
    text = "CANCEL",
    enabled = false,
    hexpand = true,
    onclick = function()
      loop_mode = false
      if generating or animating then
        send({ action = "cancel" })
        -- H10: cancel debounce — disable generate temporarily
        cancel_pending = true
        dlg:modify{ id = "generate_btn", enabled = false }
        update_status("Cancelling...")
      end
    end,
  }

  dlg:button{
    id = "animate_btn",
    text = "ANIMATE",
    enabled = false,
    hexpand = true,
    onclick = function()
      if animating or generating then return end
      local gw, gh = parse_size()
      local tag_name = dlg.data.anim_tag or ""
      if tag_name == "" then tag_name = nil end

      local req = {
        action = "generate_animation",
        method = dlg.data.anim_method,
        prompt = dlg.data.prompt,
        negative_prompt = dlg.data.negative_prompt,
        mode = dlg.data.mode,
        width = gw, height = gh,
        seed = parse_seed(),
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
        post_process = build_post_process(),
      }
      attach_lora(req)
      attach_neg_ti(req)
      if not attach_source_image(req) then return end

      animating = true
      gen_step_start = os.clock()
      start_gen_timeout()
      dlg:modify{ id = "animate_btn", enabled = false }
      dlg:modify{ id = "cancel_btn", enabled = true }
      update_status("Animating...")
      send(req)
    end,
  }

  dlg:button{
    id = "live_btn",
    text = "START LIVE",
    enabled = false,
    hexpand = true,
    onclick = function()
      if live_mode then
        send({ action = "realtime_stop" })
        stop_live_timer()
        update_status("Stopping live...")
      else
        if generating or animating then return end
        local spr = app.sprite
        if spr == nil then
          app.alert("Open a sprite first to use Live mode.")
          return
        end
        local gw, gh = parse_size()
        local req = {
          action = "realtime_start",
          prompt = dlg.data.prompt,
          negative_prompt = dlg.data.negative_prompt,
          width = gw, height = gh,
          seed = parse_seed(),
          steps = dlg.data.live_steps,
          cfg_scale = dlg.data.live_cfg / 10.0,
          denoise_strength = dlg.data.live_strength / 100.0,
          clip_skip = dlg.data.clip_skip,
          post_process = build_post_process(),
        }
        attach_lora(req)
        attach_neg_ti(req)
        live_canvas_hash = nil
        live_frame_id = 0
        update_status("Starting live...")
        send(req)
      end
    end,
  }

  dlg:button{
    id = "live_accept_btn",
    text = "ACCEPT",
    visible = false,
    onclick = function()
      local spr = app.sprite
      if spr == nil or live_preview_layer == nil then return end
      local cel = live_preview_layer:cel(app.frame)
      if cel == nil or cel.image == nil then return end
      local new_layer = spr:newLayer()
      new_layer.name = "PixyToon Live"
      spr:newCel(new_layer, app.frame, cel.image:clone(), cel.position)
      app.refresh()
      update_status("Live result accepted")
    end,
  }

  dlg:show{ wait = false, autoscrollbars = true }
end

-- ─── LAUNCH ──────────────────────────────────────────────────

build_dialog()
apply_settings(load_settings())
