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
      -- Re-enable buttons if stuck
      if generating then
        generating = false
        dlg:modify{ id = "generate_btn", enabled = true }
      end
      if animating then
        animating = false
        dlg:modify{ id = "animate_btn", enabled = true }
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
  if not ws or not connected then
    app.alert("Not connected to PixyToon server.\nStart the server and click Connect.")
    return false
  end
  ws:sendText(json.encode(payload))
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
    generating = false
    animating = false
    gen_step_start = nil
    if dlg then
      update_status("Error: " .. resp.message)
      dlg:modify{ id = "generate_btn", enabled = true }
      dlg:modify{ id = "animate_btn", enabled = true }
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

-- ─── IMPORT ANIMATION FRAME ─────────────────────────────────

import_animation_frame = function(resp)
  local img_data = base64_decode(resp.image)

  _file_counter = _file_counter + 1
  local tmp_dir = app.fs.tempPath or os.getenv("TEMP") or os.getenv("TMP") or "."
  local tmp = app.fs.joinPath(tmp_dir, "pixytoon_anim_" .. os.time() .. "_" .. _file_counter .. ".png")

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

-- ─── CAPTURE MASK ────────────────────────────────────────────

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
    _file_counter = _file_counter + 1
    local tmp_dir = app.fs.tempPath or os.getenv("TEMP") or os.getenv("TMP") or "."
    local tmp = app.fs.joinPath(tmp_dir, "pixytoon_mask_" .. os.time() .. "_" .. _file_counter .. ".png")
    mask_img:saveAs(tmp)
    local f = io.open(tmp, "rb")
    if not f then return nil end
    local data = f:read("*a")
    f:close()
    os.remove(tmp)
    return base64_encode(data)
  end

  -- Strategy B: look for a layer named "Mask" or "mask"
  for _, layer in ipairs(spr.layers) do
    if layer.name == "Mask" or layer.name == "mask" then
      local cel = layer:cel(app.frame)
      if cel and cel.image then
        _file_counter = _file_counter + 1
        local tmp_dir = app.fs.tempPath or os.getenv("TEMP") or os.getenv("TMP") or "."
        local tmp = app.fs.joinPath(tmp_dir, "pixytoon_mask_" .. os.time() .. "_" .. _file_counter .. ".png")
        cel.image:saveAs(tmp)
        local f = io.open(tmp, "rb")
        if not f then return nil end
        local data = f:read("*a")
        f:close()
        os.remove(tmp)
        return base64_encode(data)
      end
    end
  end

  return nil
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
  }

  dlg:combobox{
    id = "lora_name",
    label = "LoRA",
    options = { "(default)" },
    option = "(default)",
  }

  dlg:entry{
    id = "prompt",
    label = "Prompt",
    text = "pixel art, PixArFK, game sprite, sharp pixels",
  }

  dlg:combobox{
    id = "output_size",
    label = "Size",
    options = {
      "512x512", "512x768", "768x512",
      "384x384", "256x256", "128x128", "64x64",
    },
    option = "512x512",
  }

  dlg:entry{
    id = "seed",
    label = "Seed (-1=rand)",
    text = "-1",
  }

  dlg:slider{
    id = "denoise",
    label = "Strength %",
    min = 0,
    max = 100,
    value = 100,
  }

  dlg:slider{
    id = "steps",
    label = "Steps",
    min = 1,
    max = 50,
    value = 8,
  }

  dlg:slider{
    id = "cfg_scale",
    label = "CFG (x10)",
    min = 10,
    max = 200,
    value = 50,
  }

  dlg:slider{
    id = "clip_skip",
    label = "CLIP Skip",
    min = 1,
    max = 4,
    value = 2,
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

  -- ── Animation Settings ──
  dlg:separator{ text = "Animation Settings" }

  dlg:combobox{
    id = "anim_method",
    label = "Method",
    options = { "chain", "animatediff" },
    option = "chain",
  }

  dlg:slider{
    id = "anim_frames",
    label = "Frames",
    min = 2,
    max = 60,
    value = 8,
  }

  dlg:slider{
    id = "anim_duration",
    label = "Duration (ms)",
    min = 50,
    max = 500,
    value = 100,
  }

  dlg:slider{
    id = "anim_denoise",
    label = "Denoise %",
    min = 5,
    max = 100,
    value = 30,
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
  }

  dlg:check{
    id = "anim_freeinit",
    label = "FreeInit",
    selected = false,
  }

  dlg:slider{
    id = "anim_freeinit_iters",
    label = "FreeInit Iters",
    min = 1,
    max = 3,
    value = 2,
  }

  -- ── Actions (all action buttons grouped together) ──
  dlg:separator{ text = "Actions" }
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
        mode = dlg.data.mode,
        width = gw,
        height = gh,
        seed = tonumber(dlg.data.seed) or -1,
        steps = dlg.data.steps,
        cfg_scale = dlg.data.cfg_scale / 10.0,
        clip_skip = dlg.data.clip_skip,
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

      -- LoRA selection
      local lora_sel = dlg.data.lora_name
      if lora_sel and lora_sel ~= "(default)" then
        req.lora = { name = lora_sel, weight = 1.0 }
      end

      -- Custom palette colors (with hex validation)
      if dlg.data.palette_mode == "custom" then
        local colors_str = dlg.data.palette_custom_colors or ""
        if colors_str ~= "" then
          local colors = {}
          local invalid = false
          for hex in colors_str:gmatch("[^,%s]+") do
            local clean = hex:match("^#?(%x%x%x%x%x%x)$") or hex:match("^#?(%x%x%x)$")
            if not clean then
              app.alert("Invalid hex color: " .. hex .. "\nExpected format: #RRGGBB or #RGB")
              invalid = true
              break
            end
            -- Normalize #RGB → #RRGGBB
            if #clean == 3 then
              clean = clean:sub(1,1):rep(2) .. clean:sub(2,2):rep(2) .. clean:sub(3,3):rep(2)
            end
            colors[#colors + 1] = "#" .. clean
          end
          if invalid then return end
          if #colors > 0 then
            req.post_process.palette.colors = colors
          end
        end
      end

      -- Source image for img2img / ControlNet / Inpaint
      if req.mode == "img2img" or req.mode == "inpaint" or req.mode:find("controlnet_") then
        local b64 = capture_active_layer()
        if b64 == nil then
          app.alert("No active layer to use as source.")
          return
        end
        if req.mode == "img2img" or req.mode == "inpaint" then
          req.source_image = b64
        else
          req.control_image = b64
        end
      end

      -- Mask image for inpaint mode
      if req.mode == "inpaint" then
        local b64_mask = capture_mask()
        if b64_mask == nil then
          app.alert("Inpaint requires a mask.\nEither:\n- Make a selection (rectangle/lasso/wand)\n- Create a layer named 'Mask' with white=repaint areas")
          return
        end
        req.mask_image = b64_mask
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
    id = "cancel_btn",
    text = "CANCEL",
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
    onclick = function()
      if animating then return end

      -- Parse size (reuse from main generation section)
      local size_str = dlg.data.output_size
      local gw, gh = size_str:match("(%d+)x(%d+)")
      gw, gh = tonumber(gw), tonumber(gh)

      local tag_name = dlg.data.anim_tag or ""
      if tag_name == "" then tag_name = nil end

      local req = {
        action = "generate_animation",
        method = dlg.data.anim_method,
        prompt = dlg.data.prompt,
        mode = dlg.data.mode,
        width = gw,
        height = gh,
        seed = tonumber(dlg.data.seed) or -1,
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

      -- LoRA selection (same as single gen)
      local lora_sel = dlg.data.lora_name
      if lora_sel and lora_sel ~= "(default)" then
        req.lora = { name = lora_sel, weight = 1.0 }
      end

      -- Custom palette colors
      if dlg.data.palette_mode == "custom" then
        local colors_str = dlg.data.palette_custom_colors or ""
        if colors_str ~= "" then
          local colors = {}
          local invalid = false
          for hex in colors_str:gmatch("[^,%s]+") do
            local clean = hex:match("^#?(%x%x%x%x%x%x)$") or hex:match("^#?(%x%x%x)$")
            if not clean then
              app.alert("Invalid hex color: " .. hex .. "\nExpected format: #RRGGBB or #RGB")
              invalid = true
              break
            end
            -- Normalize #RGB → #RRGGBB
            if #clean == 3 then
              clean = clean:sub(1,1):rep(2) .. clean:sub(2,2):rep(2) .. clean:sub(3,3):rep(2)
            end
            colors[#colors + 1] = "#" .. clean
          end
          if invalid then return end
          if #colors > 0 then
            req.post_process.palette.colors = colors
          end
        end
      end

      -- Source image for img2img / ControlNet / Inpaint modes
      if req.mode == "img2img" or req.mode == "inpaint" or req.mode:find("controlnet_") then
        local b64 = capture_active_layer()
        if b64 == nil then
          app.alert("No active layer to use as source.")
          return
        end
        if req.mode == "img2img" or req.mode == "inpaint" then
          req.source_image = b64
        else
          req.control_image = b64
        end
      end

      -- Mask image for inpaint mode
      if req.mode == "inpaint" then
        local b64_mask = capture_mask()
        if b64_mask == nil then
          app.alert("Inpaint requires a mask.\nEither:\n- Make a selection (rectangle/lasso/wand)\n- Create a layer named 'Mask' with white=repaint areas")
          return
        end
        req.mask_image = b64_mask
      end

      -- Send animation request
      animating = true
      gen_step_start = os.clock()
      dlg:modify{ id = "animate_btn", enabled = false }
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
