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
local available_palettes = {}
local resources_requested = false
local connect_timer = nil
local heartbeat_timer = nil
local gen_step_start = nil
local _file_counter = 0

-- Animation state
local animating = false
local anim_layer = nil
local anim_start_frame = 0
local anim_frame_count = 0
local anim_base_seed = 0
local available_loras = {}
local available_embeddings = {}

-- Forward declarations
local handle_response
local import_result
local import_animation_frame
local request_resources
local start_heartbeat
local stop_heartbeat

-- ─── DRY HELPERS ─────────────────────────────────────────────

local function get_tmp_dir()
  return app.fs.tempPath or os.getenv("TEMP") or os.getenv("TMP") or "."
end

local function image_to_base64(img)
  local tmp_dir = get_tmp_dir()
  _file_counter = _file_counter + 1
  local tmp = tmp_dir .. "/pixytoon_b64_" .. os.time() .. "_" .. _file_counter .. ".png"
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
    palette = {
      mode = dlg.data.palette_mode,
    },
    remove_bg = dlg.data.remove_bg
  }
  if dlg.data.palette_mode == "preset" then
    pp.palette.name = dlg.data.palette_name
  elseif dlg.data.palette_mode == "custom" then
    local hex_str = dlg.data.palette_custom_colors or ""
    local colors = {}
    for hex in hex_str:gmatch("#?(%x%x%x%x%x%x)") do
      table.insert(colors, "#" .. hex)
    end
    if #colors > 0 then
      pp.palette.colors = colors
    end
  end
  return pp
end

-- ─── WEBSOCKET ───────────────────────────────────────────────

local function update_status(text)
  if dlg then
    dlg:modify{ id = "status", text = text }
  end
end

local function set_connected(state)
  connected = state
  if state then
    start_heartbeat()
  else
    stop_heartbeat()
  end
  if dlg then
    if state then
      dlg:modify{ id = "connect_btn", text = "Disconnect" }
      dlg:modify{ id = "generate_btn", enabled = true }
      dlg:modify{ id = "animate_btn", enabled = true }
    else
      dlg:modify{ id = "connect_btn", text = "Connect" }
      dlg:modify{ id = "generate_btn", enabled = false }
      dlg:modify{ id = "cancel_btn", enabled = false }
      dlg:modify{ id = "animate_btn", enabled = false }
      -- Re-enable buttons if stuck
      if generating then
        generating = false
      end
      if animating then
        animating = false
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

stop_heartbeat = function()
  if heartbeat_timer and heartbeat_timer.isRunning then
    heartbeat_timer:stop()
  end
  heartbeat_timer = nil
end

start_heartbeat = function()
  stop_heartbeat()
  heartbeat_timer = Timer{
    interval = 30.0,
    ontick = function()
      if connected and ws and not generating and not animating then
        pcall(function() ws:sendText('{"action":"ping"}') end)
      end
    end,
  }
  heartbeat_timer:start()
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
        local hok, herr = pcall(handle_response, response)
        if not hok then
          update_status("Error: " .. tostring(herr))
        end
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
  -- Reset animation state to prevent stale references
  anim_layer = nil
  anim_start_frame = 0
  anim_frame_count = 0
  anim_base_seed = 0
  generating = false
  animating = false
  update_status("Disconnected")
end

local function send(payload)
  if not connected or ws == nil then
    app.alert("Not connected to PixyToon server.")
    return false
  end
  local ok, err = pcall(function()
    ws:sendText(json.encode(payload))
  end)
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
end

-- ─── RESPONSE HANDLER ────────────────────────────────────────

handle_response = function(resp)
  if resp.type == "progress" then
    if resp.total == nil or resp.total <= 0 then return end
    if dlg then
      local pct = math.floor((resp.step / resp.total) * 100)
      local eta_str = ""

      local now = os.clock()
      if gen_step_start and resp.step > 1 and resp.total > 0 then
        local elapsed = now - gen_step_start
        local steps_done = resp.step - 1
        if steps_done > 0 then
          local avg_per_step = elapsed / steps_done
          local remaining = avg_per_step * (resp.total - resp.step)
          if remaining < 60 then
            eta_str = string.format(" — ~%.0fs left", remaining)
          else
            eta_str = string.format(" — ~%.1fmin left", remaining / 60)
          end
        end
      end

      -- Show frame context during animation
      local frame_ctx = ""
      if resp.frame_index ~= nil and resp.total_frames ~= nil then
        frame_ctx = " [Frame " .. (resp.frame_index + 1) .. "/" .. resp.total_frames .. "]"
      end
      update_status("Generating..." .. frame_ctx .. " " .. resp.step .. "/" .. resp.total .. " (" .. pct .. "%)" .. eta_str)
    end

  elseif resp.type == "result" then
    generating = false
    gen_step_start = nil
    if dlg then
      update_status("Done (" .. resp.time_ms .. "ms, seed=" .. resp.seed .. ")")
      dlg:modify{ id = "generate_btn", enabled = true }
      dlg:modify{ id = "cancel_btn", enabled = false }
    end
    import_result(resp)

  elseif resp.type == "animation_frame" then
    import_animation_frame(resp)

  elseif resp.type == "animation_complete" then
    animating = false
    gen_step_start = nil
    if dlg then
      local tag_str = ""
      if resp.tag_name and resp.tag_name ~= "" then
        tag_str = ", tag=" .. resp.tag_name
      end
      update_status("Animation done (" .. resp.total_frames .. " frames, " .. resp.total_time_ms .. "ms" .. tag_str .. ")")
      dlg:modify{ id = "animate_btn", enabled = true }
      dlg:modify{ id = "cancel_btn", enabled = false }
    end

    -- Set frame durations and create tag
    local spr = app.sprite
    if spr and anim_frame_count > 0 then
      local dur = (dlg and dlg.data.anim_duration or 100) / 1000.0
      for i = 0, anim_frame_count - 1 do
        local fn = anim_start_frame + i
        if spr.frames[fn] then
          spr.frames[fn].duration = dur
        end
      end

      -- Create animation tag
      local tag_start = anim_start_frame
      local tag_end = anim_start_frame + anim_frame_count - 1
      if resp.tag_name and resp.tag_name ~= "" and spr.frames[tag_start] and spr.frames[tag_end] then
        local tag = spr:newTag(tag_start, tag_end)
        tag.name = resp.tag_name
      end

      app.refresh()
    end

    -- Reset animation state
    anim_layer = nil
    anim_start_frame = 0
    anim_frame_count = 0
    anim_base_seed = 0

  elseif resp.type == "error" then
    local was_animating = animating
    generating = false
    animating = false
    gen_step_start = nil

    -- Finalize partial animation: set frame durations + tag for already-received frames
    if was_animating and anim_frame_count > 0 then
      local spr = app.sprite
      if spr then
        local dur = (dlg and dlg.data.anim_duration or 100) / 1000.0
        for i = 0, anim_frame_count - 1 do
          local fn = anim_start_frame + i
          if spr.frames[fn] then
            spr.frames[fn].duration = dur
          end
        end
      end
      anim_layer = nil
      anim_start_frame = 0
      anim_frame_count = 0
      anim_base_seed = 0
    end

    if dlg then
      update_status("Error: " .. resp.message)
      dlg:modify{ id = "generate_btn", enabled = true }
      dlg:modify{ id = "animate_btn", enabled = true }
      dlg:modify{ id = "cancel_btn", enabled = false }
    end
    -- Don't show popup for user-initiated cancellation
    if resp.code ~= "CANCELLED" then
      app.alert("PixyToon Error: " .. resp.message)
    end

  elseif resp.type == "list" then
    local list_type = resp.list_type or ""
    if list_type == "palettes" and resp.items then
      available_palettes = resp.items
      if dlg and #available_palettes > 0 then
        local options = {}
        for _, name in ipairs(available_palettes) do
          options[#options + 1] = name
        end
        dlg:modify{ id = "palette_name", options = options }
      end
    elseif list_type == "loras" and resp.items then
      available_loras = resp.items
      if dlg then
        local options = { "(default)" }
        for _, name in ipairs(available_loras) do
          options[#options + 1] = name
        end
        dlg:modify{ id = "lora_name", options = options }
      end
    elseif list_type == "embeddings" and resp.items then
      available_embeddings = resp.items
    end
    update_status("Resources loaded")

  elseif resp.type == "pong" then
    if not connected then
      set_connected(true)
    end
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
  local tmp_dir = get_tmp_dir()
  local tmp = app.fs.joinPath(tmp_dir, "pixytoon_" .. os.time() .. "_" .. _file_counter .. ".png")

  local ok, err = pcall(function()
    -- Write to temp file
    local f = io.open(tmp, "wb")
    if not f then
      error("Failed to create temp file")
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
  end)
  if not ok then
    pcall(os.remove, tmp)  -- cleanup temp file even on error
    update_status("Import error: " .. tostring(err))
  end
end

-- ─── IMPORT ANIMATION FRAME ─────────────────────────────────

import_animation_frame = function(resp)
  if not animating then return end  -- Ignore frames after animation_complete

  -- Guard: frame 0 must arrive first (it creates anim_layer)
  if resp.frame_index ~= 0 and anim_layer == nil then return end

  local img_data = base64_decode(resp.image)

  _file_counter = _file_counter + 1
  local tmp_dir = get_tmp_dir()
  local tmp = app.fs.joinPath(tmp_dir, "pixytoon_anim_" .. os.time() .. "_" .. _file_counter .. ".png")

  local ok, err = pcall(function()
    local f = io.open(tmp, "wb")
    if not f then return end
    f:write(img_data)
    f:close()

    local spr = app.sprite
    if spr == nil then
      spr = Sprite(resp.width, resp.height, ColorMode.RGB)
    end

    -- Create dedicated animation layer on first frame
    if resp.frame_index == 0 then
      anim_layer = spr:newLayer()
      anim_layer.name = "PixyToon Anim #" .. (resp.seed or "?")
      anim_base_seed = resp.seed or 0
      anim_frame_count = 0
      -- Anchor: use last existing frame number as the start position
      anim_start_frame = #spr.frames
    end

    -- Create frame at deterministic position
    local frame_num
    if resp.frame_index == 0 then
      -- Reuse the last existing frame for frame 0
      frame_num = anim_start_frame
    else
      -- Add new empty frame at the exact position after start
      local target_pos = anim_start_frame + resp.frame_index
      -- Ensure we don't exceed the frame array + 1
      if target_pos > #spr.frames then
        target_pos = #spr.frames + 1
      end
      local new_frame = spr:newEmptyFrame(target_pos)
      frame_num = new_frame.frameNumber
    end

    -- Load image and create cel on animation layer at the exact frame
    local img = Image{ fromFile = tmp }
    if img and anim_layer and spr.frames[frame_num] then
      spr:newCel(anim_layer, spr.frames[frame_num], img, Point(0, 0))
    end

    anim_frame_count = anim_frame_count + 1

    os.remove(tmp)
    app.refresh()

    if dlg then
      update_status("Frame " .. (resp.frame_index + 1) .. "/" .. resp.total_frames .. " (" .. resp.time_ms .. "ms)")
    end
  end)
  if not ok then
    pcall(os.remove, tmp)  -- cleanup temp file even on error
    update_status("Import error: " .. tostring(err))
  end
end

-- ─── CAPTURE ACTIVE LAYER ────────────────────────────────────

local function capture_active_layer()
  local spr = app.sprite
  if spr == nil then return nil end
  local cel = app.cel
  if cel == nil or cel.image == nil then return nil end
  -- Draw cel onto a full-size canvas to preserve correct positioning
  local full = Image(spr.spec)
  full:clear()
  full:drawImage(cel.image, cel.position)
  return image_to_base64(full)
end

-- ─── CAPTURE FLATTENED SPRITE ────────────────────────────────
-- Renders ALL visible layers into a single composite image.
-- Used as inpaint source so the model sees the full context.

local function capture_flattened()
  local spr = app.sprite
  if spr == nil then return nil end

  local flat_img = Image(spr.spec)
  flat_img:drawSprite(spr, app.frame)

  return image_to_base64(flat_img)
end

-- ─── CAPTURE MASK ────────────────────────────────────────────
-- Priority: Selection > "Mask" layer > Active layer alpha (auto)

local function capture_mask()
  local spr = app.sprite
  if spr == nil then return nil end

  -- Strategy A: use active selection as mask
  local sel = spr.selection
  if sel and not sel.isEmpty then
    local mask_img = Image(spr.width, spr.height, ColorMode.GRAY)
    mask_img:clear(Color{ gray = 0 })  -- black = keep
    for y = sel.bounds.y, sel.bounds.y + sel.bounds.height - 1 do
      for x = sel.bounds.x, sel.bounds.x + sel.bounds.width - 1 do
        if sel:contains(x, y) then
          mask_img:drawPixel(x, y, Color{ gray = 255 })  -- white = repaint
        end
      end
    end
    return image_to_base64(mask_img)
  end

  -- Strategy B: look for a layer named "Mask" or "mask"
  for _, layer in ipairs(spr.layers) do
    if layer.name == "Mask" or layer.name == "mask" then
      local cel = layer:cel(app.frame)
      if cel and cel.image then
        return image_to_base64(cel.image)
      end
    end
  end

  -- Strategy C: auto-derive mask from active layer's non-transparent pixels
  -- User draws on a layer → non-transparent pixels become the repaint area
  local cel = app.cel
  if cel and cel.image then
    local img = cel.image
    local mask_img = Image(spr.width, spr.height, ColorMode.GRAY)
    mask_img:clear(Color{ gray = 0 })
    local ox, oy = cel.position.x, cel.position.y
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
    return image_to_base64(mask_img)
  end

  return nil
end

-- ─── BUILD DIALOG ────────────────────────────────────────────

local function build_dialog()
  dlg = Dialog{
    title = "PixyToon — AI Pixel Art Generator",
    onclose = function()
      disconnect()
    end
  }

  -- ── Connection ──
  dlg:separator{ text = "Connection" }
  dlg:entry{
    id = "server_url",
    label = "Server",
    text = SERVER_URL,
    tooltip = "WebSocket server URL (e.g. ws://127.0.0.1:9876/ws)",
  }
  dlg:label{ id = "status", text = "Disconnected" }
  dlg:button{
    id = "connect_btn",
    text = "Connect",
    onclick = function()
      if connected then
        disconnect()
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
        update_status("Refreshing resources...")
      else
        update_status("Not connected")
      end
    end
  }

  -- ── Generation ──
  dlg:separator{ text = "Generation" }

  dlg:combobox{
    id = "mode",
    label = "Mode",
    options = {
      "txt2img", "img2img", "inpaint",
      "controlnet_openpose", "controlnet_canny",
      "controlnet_scribble", "controlnet_lineart",
    },
    option = "txt2img",
    tooltip = "Generation mode: text-to-image, image-to-image, inpaint, or ControlNet variants.",
  }

  dlg:combobox{
    id = "lora_name",
    label = "LoRA",
    options = { "(default)" },
    option = "(default)",
    tooltip = "Style LoRA to apply. Place .safetensors files in server/models/loras/.",
  }

  dlg:slider{
    id = "lora_weight",
    label = "LoRA Strength",
    min = -200,
    max = 200,
    value = 100,
    tooltip = "LoRA influence strength. 100 = full, 0 = none, negative = inverse."
  }

  dlg:entry{
    id = "prompt",
    label = "Prompt",
    text = "pixel art, PixArFK, game sprite, sharp pixels",
    tooltip = "Describe what to generate. Comma-separated tags work best."
  }

  dlg:entry{
    id = "negative_prompt",
    label = "Neg. Prompt",
    text = "blurry, antialiased, smooth gradient, photorealistic, 3d render, soft edges, low quality, worst quality",
    tooltip = "Terms to avoid. Pre-filled with pixel art optimized defaults."
  }

  dlg:combobox{
    id = "output_size",
    label = "Size",
    options = {
      "512x512", "512x768", "768x512",
      "384x384", "256x256", "128x128", "64x64",
    },
    option = "512x512",
    tooltip = "Generated image dimensions (must be multiple of 8)"
  }

  dlg:entry{
    id = "seed",
    label = "Seed (-1=rand)",
    text = "-1",
    tooltip = "Seed for reproducibility. -1 = random each time."
  }

  dlg:slider{
    id = "denoise",
    label = "Strength %",
    min = 0,
    max = 100,
    value = 100,
    tooltip = "Denoising strength. 100% = full redraw, lower = preserve more of source image."
  }

  dlg:slider{
    id = "steps",
    label = "Steps",
    min = 1,
    max = 50,
    value = 8,
    tooltip = "Inference steps. 6-12 for Hyper-SD. More steps = slower but sometimes better."
  }

  dlg:slider{
    id = "cfg_scale",
    label = "CFG Scale",
    min = 10,
    max = 200,
    value = 50,
    tooltip = "Classifier-Free Guidance. Higher = more prompt adherence. 3-7 typical."
  }

  dlg:slider{
    id = "clip_skip",
    label = "CLIP Skip",
    min = 1,
    max = 4,
    value = 2,
    tooltip = "CLIP skip layers. 2 recommended for stylized/pixel art."
  }

  -- ── Post-Processing ──
  dlg:separator{ text = "Post-Processing" }

  dlg:check{
    id = "pixelate",
    label = "Pixelate",
    selected = true,
    tooltip = "Enable pixel art downscaling to target size.",
  }

  dlg:slider{
    id = "pixel_size",
    label = "Target Size",
    min = 8,
    max = 256,
    value = 128,
    tooltip = "Pixel art resolution (before upscale). Lower = more pixelated."
  }

  dlg:slider{
    id = "colors",
    label = "Colors",
    min = 2,
    max = 256,
    value = 32,
    tooltip = "Max colors in final palette. Lower = more retro."
  }

  dlg:combobox{
    id = "quantize_method",
    label = "Quantize",
    options = { "kmeans", "median_cut", "octree" },
    option = "kmeans",
    tooltip = "Color quantization algorithm. kmeans=best quality, octree=fastest.",
  }

  dlg:combobox{
    id = "dither",
    label = "Dithering",
    options = { "none", "floyd_steinberg", "bayer_2x2", "bayer_4x4", "bayer_8x8" },
    option = "none",
    tooltip = "Dithering algorithm for color reduction."
  }

  dlg:combobox{
    id = "palette_mode",
    label = "Palette",
    options = { "auto", "preset", "custom" },
    option = "auto",
    tooltip = "auto=extract, preset=use palette file, custom=your hex colors.",
    onchange = function()
      local m = dlg.data.palette_mode
      dlg:modify{ id = "palette_name", visible = (m == "preset") }
      dlg:modify{ id = "palette_custom_colors", visible = (m == "custom") }
    end
  }

  dlg:combobox{
    id = "palette_name",
    label = "Preset",
    options = { "pico8" },  -- populated on connect
    option = "pico8",
    visible = false,
    tooltip = "Preset palette from server/palettes/.",
  }

  dlg:entry{
    id = "palette_custom_colors",
    label = "Custom Hex",
    text = "",  -- comma-separated: #ff0000,#00ff00,#0000ff
    visible = false,
    tooltip = "Comma-separated hex colors: #ff0000,#00ff00,#0000ff",
  }

  dlg:check{
    id = "remove_bg",
    label = "Remove BG",
    selected = false,
    tooltip = "Remove background using AI (adds ~3-4s, CPU)."
  }

  -- ── Animation Settings ──
  dlg:separator{ text = "Animation Settings" }

  dlg:combobox{
    id = "anim_method",
    label = "Method",
    options = { "chain", "animatediff" },
    option = "chain",
    tooltip = "chain=frame-by-frame img2img, animatediff=temporal coherence model.",
    onchange = function()
      local is_ad = dlg.data.anim_method == "animatediff"
      dlg:modify{ id = "anim_freeinit", visible = is_ad }
      dlg:modify{ id = "anim_freeinit_iters", visible = is_ad }
    end
  }

  dlg:slider{
    id = "anim_frames",
    label = "Frames",
    min = 2,
    max = 120,
    value = 8,
    tooltip = "Total frames to generate."
  }

  dlg:slider{
    id = "anim_duration",
    label = "Duration (ms)",
    min = 50,
    max = 500,
    value = 100,
    tooltip = "Milliseconds per frame for playback."
  }

  dlg:slider{
    id = "anim_denoise",
    label = "Denoise %",
    min = 5,
    max = 100,
    value = 30,
    tooltip = "Denoising per frame. Lower = more consistency between frames."
  }

  dlg:combobox{
    id = "anim_seed_strategy",
    label = "Seed Mode",
    options = { "increment", "fixed", "random" },
    option = "increment",
    tooltip = "How seed changes between frames."
  }

  dlg:entry{
    id = "anim_tag",
    label = "Tag Name",
    text = "",
    tooltip = "Aseprite tag name for the animation (e.g. walk, idle).",
  }

  dlg:check{
    id = "anim_freeinit",
    label = "FreeInit",
    selected = false,
    visible = false,
    tooltip = "FreeInit: improve temporal consistency (AnimateDiff only, slower)."
  }

  dlg:slider{
    id = "anim_freeinit_iters",
    label = "FreeInit Iters",
    min = 1,
    max = 3,
    value = 2,
    visible = false,
    tooltip = "FreeInit iterations (higher = better consistency, slower).",
  }

  -- ── Actions (all action buttons grouped together) ──
  dlg:separator{ text = "Actions" }
  dlg:button{
    id = "generate_btn",
    text = "GENERATE",
    enabled = false,
    onclick = function()
      if generating then return end
      if animating then return end

      -- Parse size
      local size_str = dlg.data.output_size
      local gw, gh = size_str:match("(%d+)x(%d+)")
      gw, gh = tonumber(gw), tonumber(gh)

      -- Seed validation
      local seed_text = dlg.data.seed
      local seed_val = tonumber(seed_text) or -1
      if seed_val ~= math.floor(seed_val) then seed_val = -1 end

      -- Build request
      local req = {
        action = "generate",
        prompt = dlg.data.prompt,
        negative_prompt = dlg.data.negative_prompt,
        mode = dlg.data.mode,
        width = gw,
        height = gh,
        seed = seed_val,
        steps = dlg.data.steps,
        cfg_scale = dlg.data.cfg_scale / 10.0,
        clip_skip = dlg.data.clip_skip,
        denoise_strength = dlg.data.denoise / 100.0,
        post_process = build_post_process(),
      }

      -- LoRA selection
      local lora_sel = dlg.data.lora_name
      if lora_sel and lora_sel ~= "(default)" then
        req.lora = { name = lora_sel, weight = dlg.data.lora_weight / 100.0 }
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

      -- Inpaint: source = flattened sprite (full context), mask = auto from active layer
      if req.mode == "inpaint" then
        local b64_source = capture_flattened()
        if b64_source == nil then
          app.alert("Inpaint requires an open sprite with at least one layer.")
          return
        end
        req.source_image = b64_source

        local b64_mask = capture_mask()
        if b64_mask == nil then
          app.alert("Inpaint requires a mask.\nEither:\n- Make a selection (rectangle/lasso/wand)\n- Create a layer named 'Mask' with white=repaint areas\n- Draw on the active layer (non-transparent pixels = repaint area)")
          return
        end
        req.mask_image = b64_mask
      end

      -- Send
      generating = true
      gen_step_start = os.clock()
      dlg:modify{ id = "generate_btn", enabled = false }
      dlg:modify{ id = "cancel_btn", enabled = true }
      update_status("Generating...")
      send(req)
    end
  }

  dlg:button{
    id = "cancel_btn",
    text = "CANCEL",
    enabled = false,
    onclick = function()
      if generating or animating then
        send({ action = "cancel" })
        update_status("Cancelling...")
      end
    end
  }

  dlg:button{
    id = "animate_btn",
    text = "ANIMATE",
    enabled = false,
    onclick = function()
      if animating then return end
      if generating then return end

      -- Parse size (reuse from main generation section)
      local size_str = dlg.data.output_size
      local gw, gh = size_str:match("(%d+)x(%d+)")
      gw, gh = tonumber(gw), tonumber(gh)

      local tag_name = dlg.data.anim_tag or ""
      if tag_name == "" then tag_name = nil end

      -- Seed validation
      local seed_text = dlg.data.seed
      local seed_val = tonumber(seed_text) or -1
      if seed_val ~= math.floor(seed_val) then seed_val = -1 end

      local req = {
        action = "generate_animation",
        method = dlg.data.anim_method,
        prompt = dlg.data.prompt,
        negative_prompt = dlg.data.negative_prompt,
        mode = dlg.data.mode,
        width = gw,
        height = gh,
        seed = seed_val,
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

      -- LoRA selection (same as single gen)
      local lora_sel = dlg.data.lora_name
      if lora_sel and lora_sel ~= "(default)" then
        req.lora = { name = lora_sel, weight = dlg.data.lora_weight / 100.0 }
      end

      -- Source image for img2img / ControlNet modes
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

      -- Inpaint: source = flattened sprite, mask = auto from active layer
      if req.mode == "inpaint" then
        local b64_source = capture_flattened()
        if b64_source == nil then
          app.alert("Inpaint requires an open sprite with at least one layer.")
          return
        end
        req.source_image = b64_source

        local b64_mask = capture_mask()
        if b64_mask == nil then
          app.alert("Inpaint requires a mask.\nEither:\n- Make a selection (rectangle/lasso/wand)\n- Create a layer named 'Mask' with white=repaint areas\n- Draw on the active layer (non-transparent pixels = repaint area)")
          return
        end
        req.mask_image = b64_mask
      end

      -- Send animation request
      animating = true
      gen_step_start = os.clock()
      dlg:modify{ id = "animate_btn", enabled = false }
      dlg:modify{ id = "cancel_btn", enabled = true }
      update_status("Animating...")
      send(req)
    end
  }

  dlg:show{ wait = false }
end

-- ─── LAUNCH ──────────────────────────────────────────────────

build_dialog()

-- Auto-connect on dialog open
connect()
