--
-- PixyToon — Aseprite Extension for AI Pixel Art Generation
--
-- Connects to the local PixyToon Python server via WebSocket
-- and provides a full GUI for generating pixel art sprites.
--

-- Load json.lua from same directory (works from scripts/ or extensions/)
local json
local scripts_path = app.fs.joinPath(app.fs.userConfigPath, "scripts", "json.lua")
local ext_path = app.fs.joinPath(app.fs.userConfigPath, "extensions", "pixytoon", "scripts", "json.lua")
if app.fs.isFile(scripts_path) then
  json = dofile(scripts_path)
else
  json = dofile(ext_path)
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
local available_loras = {}
local available_palettes = {}
local available_embeddings = {}
local resources_requested = false
local connect_timer = nil
local gen_step_start = nil
local _file_counter = 0

-- Forward declarations
local handle_response
local import_result
local request_resources

-- ─── WEBSOCKET ───────────────────────────────────────────────

local function update_status(text)
  if dlg then
    dlg:modify{ id = "status", text = text }
  end
end

local function set_connected(state)
  connected = state
  if dlg then
    if state then
      dlg:modify{ id = "connect_btn", text = "Disconnect" }
    else
      dlg:modify{ id = "connect_btn", text = "Connect" }
      -- Re-enable generate button if stuck
      if generating then
        generating = false
        dlg:modify{ id = "generate_btn", enabled = true }
      end
    end
  end
end

local function stop_connect_timer()
  if connect_timer and connect_timer.isRunning then
    connect_timer:stop()
  end
  connect_timer = nil
end

local function connect()
  if ws then
    ws:close()
    ws = nil
  end
  update_status("Connecting...")
  ws = WebSocket{
    url = SERVER_URL,
    onreceive = function(msg_type, data)
      -- Handle OPEN event
      if msg_type == WebSocketMessageType.OPEN then
        stop_connect_timer()
        set_connected(true)
        update_status("Connected")
        -- Send ping — pong handler will request resources
        pcall(function() ws:sendText(json.encode({ action = "ping" })) end)
        return
      end

      -- Handle CLOSE event
      if msg_type == WebSocketMessageType.CLOSE then
        set_connected(false)
        resources_requested = false
        update_status("Disconnected (server closed)")
        ws = nil
        return
      end

      -- Handle TEXT messages
      if msg_type == WebSocketMessageType.TEXT then
        -- If OPEN never fired, detect connection from first TEXT message
        if not connected then
          stop_connect_timer()
          set_connected(true)
          update_status("Connected")
        end
        local ok, response = pcall(json.decode, data)
        if not ok then return end
        handle_response(response)
      end
    end,
    deflate = false,
  }

  -- Explicit connect required by Aseprite WebSocket API
  ws:connect()

  -- Connection timeout — if not connected after 5s, inform user
  connect_timer = Timer{
    interval = 5.0,
    ontick = function()
      stop_connect_timer()
      if not connected then
        if ws then
          ws:close()
          ws = nil
        end
        update_status("Connection failed — is the server running?")
      end
    end,
  }
  connect_timer:start()
end

local function disconnect()
  stop_connect_timer()
  if ws then
    ws:close()
    ws = nil
  end
  set_connected(false)
  resources_requested = false
  update_status("Disconnected")
end

local function send(payload)
  if not ws or not connected then
    app.alert("Not connected to PixyToon server.\nStart the server and click Connect.")
    return false
  end
  ws:sendText(json.encode(payload))
  return true
end

request_resources = function()
  resources_requested = true
  send({ action = "list_loras" })
  send({ action = "list_palettes" })
  send({ action = "list_embeddings" })
end

-- ─── RESPONSE HANDLER ────────────────────────────────────────

handle_response = function(resp)
  if resp.type == "progress" then
    if dlg then
      local pct = math.floor((resp.step / resp.total) * 100)
      local eta_str = ""

      -- ETA calculation
      local now = os.clock()
      if gen_step_start and resp.step > 1 then
        local elapsed = now - gen_step_start
        local avg_per_step = elapsed / (resp.step - 1)
        local remaining = avg_per_step * (resp.total - resp.step)
        if remaining < 60 then
          eta_str = string.format(" — ~%.0fs left", remaining)
        else
          eta_str = string.format(" — ~%.1fmin left", remaining / 60)
        end
      end

      update_status("Generating... " .. resp.step .. "/" .. resp.total .. " (" .. pct .. "%)" .. eta_str)
    end

  elseif resp.type == "result" then
    generating = false
    gen_step_start = nil
    if dlg then
      update_status("Done (" .. resp.time_ms .. "ms, seed=" .. resp.seed .. ")")
      dlg:modify{ id = "generate_btn", enabled = true }
    end
    -- Import the generated image as a new layer
    import_result(resp)

  elseif resp.type == "error" then
    generating = false
    gen_step_start = nil
    if dlg then
      update_status("Error: " .. resp.message)
      dlg:modify{ id = "generate_btn", enabled = true }
    end
    app.alert("PixyToon Error: " .. resp.message)

  elseif resp.type == "list" then
    local list_type = resp.list_type or ""
    if list_type == "loras" and resp.items then
      available_loras = resp.items
      if dlg and #available_loras > 0 then
        -- Build options and auto-select first available LoRA
        local options = { "" }  -- empty = no LoRA
        for _, name in ipairs(available_loras) do
          options[#options + 1] = name
        end
        dlg:modify{ id = "lora_name", options = options, option = available_loras[1] }
        update_status("Loaded " .. #available_loras .. " LoRA(s) — " .. available_loras[1] .. " selected")
      end
    elseif list_type == "palettes" and resp.items then
      available_palettes = resp.items
      if dlg and #available_palettes > 0 then
        local options = {}
        for _, name in ipairs(available_palettes) do
          options[#options + 1] = name
        end
        dlg:modify{ id = "palette_name", options = options }
        update_status("Loaded " .. #available_palettes .. " palette(s)")
      end
    elseif list_type == "embeddings" and resp.items then
      available_embeddings = resp.items
      if dlg and #available_embeddings > 0 then
        local options = { "" }  -- empty = no TI
        for _, name in ipairs(available_embeddings) do
          options[#options + 1] = name
        end
        dlg:modify{ id = "neg_ti", options = options, option = available_embeddings[1] }
        update_status("Loaded " .. #available_embeddings .. " embedding(s)")
      end
    end

  elseif resp.type == "pong" then
    if not connected then
      set_connected(true)
    end
    -- Request resource lists on first pong
    if not resources_requested then
      request_resources()
    end
    update_status("Connected")
  end
end

-- ─── IMPORT RESULT ───────────────────────────────────────────

import_result = function(resp)
  local img_data = base64_decode(resp.image)

  _file_counter = _file_counter + 1
  local tmp_dir = app.fs.tempPath or os.getenv("TEMP") or os.getenv("TMP") or "."
  local tmp = app.fs.joinPath(tmp_dir, "pixytoon_" .. os.time() .. "_" .. _file_counter .. ".png")

  -- Write to temp file
  local f = io.open(tmp, "wb")
  if not f then
    app.alert("Failed to create temp file")
    return
  end
  f:write(img_data)
  f:close()

  local spr = app.sprite
  if spr == nil then
    spr = Sprite(resp.width, resp.height, ColorMode.RGB)
  end

  -- Create new layer
  local layer = spr:newLayer()
  layer.name = "PixyToon #" .. (resp.seed or "?")

  -- Load image and create cel
  local img = Image{ fromFile = tmp }
  if img then
    spr:newCel(layer, app.frame, img, Point(0, 0))
  end

  -- Cleanup temp
  os.remove(tmp)
  app.refresh()
end

-- ─── CAPTURE ACTIVE LAYER ────────────────────────────────────

local function capture_active_layer()
  local cel = app.cel
  if cel == nil then return nil end

  local img = cel.image
  if img == nil then return nil end

  _file_counter = _file_counter + 1
  local tmp_dir = app.fs.tempPath or os.getenv("TEMP") or os.getenv("TMP") or "."
  local tmp = app.fs.joinPath(tmp_dir, "pixytoon_src_" .. os.time() .. "_" .. _file_counter .. ".png")
  img:saveAs(tmp)

  local f = io.open(tmp, "rb")
  if not f then return nil end
  local data = f:read("*a")
  f:close()
  os.remove(tmp)

  return base64_encode(data)
end

-- ─── BUILD DIALOG ────────────────────────────────────────────

local function build_dialog()
  dlg = Dialog{
    title = "PixyToon — AI Pixel Art Generator",
  }

  -- ── Connection ──
  dlg:separator{ text = "Connection" }
  dlg:label{ id = "status", text = "Disconnected" }
  dlg:button{
    id = "connect_btn",
    text = "Connect",
    onclick = function()
      if connected then
        disconnect()
      else
        connect()
      end
    end
  }

  -- ── Generation Settings ──
  dlg:separator{ text = "Generation" }

  dlg:combobox{
    id = "mode",
    label = "Mode",
    options = {
      "txt2img", "img2img",
      "controlnet_openpose", "controlnet_canny",
      "controlnet_scribble", "controlnet_lineart",
    },
    option = "txt2img",
  }

  dlg:entry{
    id = "prompt",
    label = "Prompt",
    text = "pixel art, PixArFK, game sprite, 16-bit style, sharp pixels, clean edges",
  }

  dlg:entry{
    id = "neg_prompt",
    label = "Neg. Prompt",
    text = "blurry, antialiased, smooth gradient, photorealistic, 3d render, soft edges, anti-aliasing, bokeh, depth of field, low quality, worst quality, bad quality, jpeg artifacts, watermark, text, logo, deformed, disfigured, bad anatomy, bad proportions, extra limbs, missing limbs, extra fingers, fused fingers, poorly drawn hands, poorly drawn face, ugly, realistic, photo, high resolution, complex shading",
  }

  dlg:combobox{
    id = "output_size",
    label = "Gen Size",
    options = {
      "512x512", "512x768", "768x512",
      "384x384", "256x256", "128x128", "64x64",
    },
    option = "512x512",
  }

  dlg:slider{
    id = "steps",
    label = "Steps",
    min = 2,
    max = 20,
    value = 8,      -- Hyper-SD 8-step LoRA is optimized for 8 steps
  }

  dlg:slider{
    id = "cfg",
    label = "CFG (x10)",
    min = 10,        -- 1.0
    max = 100,       -- 10.0 — wider range for flexibility
    value = 50,      -- 5.0 — matches server default_cfg
  }

  dlg:entry{
    id = "seed",
    label = "Seed (-1=rand)",
    text = "-1",
  }

  dlg:slider{
    id = "denoise",
    label = "Denoise",
    min = 0,
    max = 100,
    value = 75,      -- matches server default_denoise
  }

  -- ── LoRA ──
  dlg:separator{ text = "LoRA" }

  dlg:combobox{
    id = "lora_name",
    label = "LoRA",
    options = { "" },  -- populated on connect, auto-selects first
    option = "",
  }

  dlg:slider{
    id = "lora_weight",
    label = "Weight (x100)",
    min = -200,
    max = 200,
    value = 100,     -- 1.0 — full LoRA strength default
  }

  -- ── Textual Inversion (Negative Embeddings) ──
  dlg:separator{ text = "Neg. Embeddings" }

  dlg:combobox{
    id = "neg_ti",
    label = "Embedding",
    options = { "" },  -- populated on connect
    option = "",
  }

  -- ── Post-Processing ──
  dlg:separator{ text = "Post-Processing" }

  dlg:check{
    id = "pixelate",
    label = "Pixelate",
    selected = true,
  }

  dlg:slider{
    id = "pixel_size",
    label = "Target Size",
    min = 8,
    max = 256,
    value = 128,
  }

  dlg:slider{
    id = "colors",
    label = "Colors",
    min = 2,
    max = 256,
    value = 32,
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
  }

  dlg:combobox{
    id = "palette_name",
    label = "Preset",
    options = { "pico8" },  -- populated on connect
    option = "pico8",
  }

  dlg:entry{
    id = "palette_custom_colors",
    label = "Custom Hex",
    text = "",  -- comma-separated: #ff0000,#00ff00,#0000ff
  }

  dlg:check{
    id = "remove_bg",
    label = "Remove BG",
    selected = false,
  }

  -- ── Actions ──
  dlg:separator{ text = "" }
  dlg:button{
    id = "generate_btn",
    text = "GENERATE",
    onclick = function()
      if generating then return end

      -- Parse size
      local size_str = dlg.data.output_size
      local gw, gh = size_str:match("(%d+)x(%d+)")
      gw, gh = tonumber(gw), tonumber(gh)

      -- Build request
      local req = {
        action = "generate",
        prompt = dlg.data.prompt,
        negative_prompt = dlg.data.neg_prompt,
        mode = dlg.data.mode,
        width = gw,
        height = gh,
        seed = tonumber(dlg.data.seed) or -1,
        steps = dlg.data.steps,
        cfg_scale = dlg.data.cfg / 10.0,
        denoise_strength = dlg.data.denoise / 100.0,
        post_process = {
          pixelate = {
            enabled = dlg.data.pixelate,
            target_size = dlg.data.pixel_size,
          },
          quantize_method = dlg.data.quantize_method,
          quantize_colors = dlg.data.colors,
          dither = dlg.data.dither,
          palette = {
            mode = dlg.data.palette_mode,
            name = dlg.data.palette_name,
          },
          remove_bg = dlg.data.remove_bg,
        },
      }

      -- Custom palette colors
      if dlg.data.palette_mode == "custom" then
        local colors_str = dlg.data.palette_custom_colors or ""
        if colors_str ~= "" then
          local colors = {}
          for hex in colors_str:gmatch("[^,%s]+") do
            colors[#colors + 1] = hex
          end
          if #colors > 0 then
            req.post_process.palette.colors = colors
          end
        end
      end

      -- LoRA
      local lora_name = dlg.data.lora_name
      if lora_name and lora_name ~= "" then
        req.lora = {
          name = lora_name,
          weight = dlg.data.lora_weight / 100.0,
        }
      end

      -- Negative TI embedding
      local ti_name = dlg.data.neg_ti
      if ti_name and ti_name ~= "" then
        req.negative_ti = {
          { name = ti_name, weight = 1.0 },
        }
      end

      -- Source image for img2img / ControlNet
      if req.mode == "img2img" or req.mode:find("controlnet_") then
        local b64 = capture_active_layer()
        if b64 == nil then
          app.alert("No active layer to use as source.")
          return
        end
        if req.mode == "img2img" then
          req.source_image = b64
        else
          req.control_image = b64
        end
      end

      -- Send
      generating = true
      gen_step_start = os.clock()
      dlg:modify{ id = "generate_btn", enabled = false }
      update_status("Generating...")
      send(req)
    end
  }

  dlg:button{
    id = "refresh_btn",
    text = "Refresh Lists",
    onclick = function()
      if connected then
        resources_requested = false
        request_resources()
        update_status("Refreshing resource lists...")
      else
        update_status("Not connected")
      end
    end
  }

  dlg:show{ wait = false }
end

-- ─── LAUNCH ──────────────────────────────────────────────────

build_dialog()
