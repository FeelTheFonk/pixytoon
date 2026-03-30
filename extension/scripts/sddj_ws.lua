--
-- SDDj — WebSocket Transport & Connection Management
--

return function(PT)

-- ─── Status ─────────────────────────────────────────────────

function PT.update_status(text)
  if PT.dlg then
    pcall(PT.dlg.modify, PT.dlg, { id = "status", text = text })
  end
end

-- ─── Heartbeat ──────────────────────────────────────────────

function PT.stop_heartbeat()
  PT.timers.heartbeat = PT.stop_timer(PT.timers.heartbeat)
end

function PT.start_heartbeat()
  PT.stop_heartbeat()
  local ok, t = pcall(Timer, {
    interval = PT.cfg.HEARTBEAT_INTERVAL,
    ontick = function()
      if not PT.state.connected or not PT.ws_handle then return end
      -- Pong watchdog: detect unresponsive server
      if PT.state.last_pong
         and not PT.state.generating
         and not PT.state.animating
         and not PT.audio.generating
         and (os.clock() - PT.state.last_pong) > PT.cfg.HEARTBEAT_INTERVAL * 3 then
        PT.set_connected(false)
        PT.update_status("Server unresponsive — disconnected")
        if PT.ws_handle then pcall(function() PT.ws_handle:close() end); PT.ws_handle = nil end
        PT.schedule_reconnect()
        return
      end
      if not PT.state.generating and not PT.state.animating then
        pcall(function() PT.ws_handle:sendText('{"action":"ping"}') end)
      end
    end,
  })
  if ok and t then PT.timers.heartbeat = t; t:start() end
end

-- ─── Generation Timeout ─────────────────────────────────────

function PT.stop_gen_timeout()
  PT.timers.gen_timeout = PT.stop_timer(PT.timers.gen_timeout)
end

function PT.start_gen_timeout(override_seconds)
  PT.stop_gen_timeout()
  local ok, t = pcall(Timer, {
    interval = override_seconds or PT.cfg.GEN_TIMEOUT,
    ontick = function()
      PT.stop_gen_timeout()
      if PT.state.generating or PT.state.animating then
        -- Send cancel to server so GPU stops working
        pcall(function() PT.send({ action = "cancel" }) end)
        PT.state.generating = false
        PT.state.animating = false
        PT.audio.generating = false
        PT.audio.analyzing = false
        PT.state.pending_action = nil
        PT.state.cancel_pending = false
        PT.timers.cancel_safety = PT.stop_timer(PT.timers.cancel_safety)
        PT.reset_loop_state()
        PT.timers.loop = PT.stop_timer(PT.timers.loop)
        PT.state.gen_step_start = nil
        PT.stop_refresh_timer()
        PT.clear_response_queue()
        PT.reset_sequence()
        -- Reset anim tracking fields
        PT.anim.last_frame_index = -1
        PT.anim.decode_failures = 0
        if PT.dlg then
          PT.update_status("Timed out — no response from server")
          PT.reset_ui_buttons()
        end
      end
    end,
  })
  if ok and t then PT.timers.gen_timeout = t; t:start() end
end

-- ─── Connection State ───────────────────────────────────────

function PT.set_connected(is_connected)
  PT.state.connected = is_connected
  if is_connected then
    PT.start_heartbeat()
    PT.state.last_pong = os.clock()
    PT.reconnect.attempts = 0
  else
    PT.stop_heartbeat()
  end
  if not PT.dlg then return end

  if is_connected then
    PT.dlg:modify{ id = "connect_btn", text = "Disconnect" }
    PT.dlg:modify{ id = "action_btn", enabled = true }
  else
    PT.dlg:modify{ id = "connect_btn", text = "Connect" }
    PT.update_action_button(PT.dlg.data.main_tabs or "tab_gen")
    PT.dlg:modify{ id = "action_btn", enabled = false }
    PT.dlg:modify{ id = "cancel_btn", enabled = false }
    pcall(function() PT.dlg:modify{ id = "export_mp4_btn", enabled = false } end)
    if PT.state.generating then PT.state.generating = false end
    if PT.state.animating then PT.state.animating = false end
    PT.stop_gen_timeout()
    PT.timers.cancel_safety = PT.stop_timer(PT.timers.cancel_safety)
    PT.reset_loop_state()
    PT.loop.counter = 0
    PT.timers.loop = PT.stop_timer(PT.timers.loop)
    PT.state.cancel_pending = false
    PT.state.pending_action = nil
    PT.state.last_pong = nil
    -- Reset animation state (prevent orphan layer references)
    PT.anim.layer = nil
    PT.anim.start_frame = 0
    PT.anim.frame_count = 0
    PT.anim.base_seed = 0
    PT.anim.output_dir = nil
    PT.anim.output_count = 0
    PT.anim.last_saved_frame = nil
    PT.anim.last_frame_index = -1
    PT.anim.decode_failures = 0
    -- Reset sequence state
    PT.reset_sequence()
    -- Reset audio state
    PT.audio.analyzing = false
    PT.audio.generating = false
    PT.audio.analyzed = false
    PT.audio.last_output_dir = nil
    -- Stop refresh timer and clear queued messages
    if PT.stop_refresh_timer then PT.stop_refresh_timer() end
    if PT.clear_response_queue then PT.clear_response_queue() end
  end
end

-- ─── Connect / Disconnect ───────────────────────────────────

function PT.stop_connect_timer()
  PT.timers.connect = PT.stop_timer(PT.timers.connect)
end

function PT.connect()
  if PT.state.connecting then return end  -- prevent double-connect race
  PT.state.connecting = true
  PT.reconnect.manual_disconnect = false
  -- Validate URL before connecting (basic scheme check — allows paths, ports, IPs)
  local url = PT.cfg.DEFAULT_SERVER_URL or ""
  if not url:match("^wss?://[%w%.%-_]+") then
    PT.update_status("Invalid server URL: " .. url)
    PT.state.connecting = false
    return
  end
  if PT.ws_handle then pcall(function() PT.ws_handle:close() end); PT.ws_handle = nil end
  PT.update_status("Connecting...")
  PT.ws_handle = WebSocket{
    url = url,
    onreceive = function(msg_type, data)
      if msg_type == WebSocketMessageType.OPEN then
        PT.state.connecting = false
        PT.stop_connect_timer()
        PT.timers.reconnect = PT.stop_timer(PT.timers.reconnect)
        PT.reconnect.attempts = 0
        PT.set_connected(true)
        PT.update_status("Connected")
        pcall(function() PT.ws_handle:sendText(PT.json.encode({ action = "ping" })) end)
        -- Always re-request resources on (re)connect
        PT.request_resources()
        return
      end
      if msg_type == WebSocketMessageType.CLOSE then
        PT.state.connecting = false
        PT.set_connected(false)
        PT.res.requested = false
        PT.update_status("Disconnected (server closed)")
        PT.ws_handle = nil
        -- Auto-reconnect if not manual disconnect
        if not PT.reconnect.manual_disconnect then
          PT.schedule_reconnect()
        end
        return
      end
      if msg_type == WebSocketMessageType.BINARY then
        -- Binary frame: [uint32 LE json_len][JSON metadata][raw RGBA bytes]
        if #data < 4 then return end
        local b1, b2, b3, b4 = string.byte(data, 1, 4)
        local json_len = b1 + b2 * 256 + b3 * 65536 + b4 * 16777216
        if #data < 4 + json_len then return end
        local json_str = data:sub(5, 4 + json_len)
        local ok, response = pcall(PT.json.decode, json_str)
        if ok and type(response) ~= "table" then
          ok = false
          response = "Invalid binary JSON structure (not a table)"
        end
        if not ok then
          PT.update_status("Binary JSON error: " .. tostring(response))
          return
        end
        -- Attach raw image bytes directly (skip base64 entirely)
        response._raw_image = data:sub(5 + json_len)
        -- Any valid message from server proves it's alive — reset watchdog
        PT.state.last_pong = os.clock()
        local hok, herr = pcall(PT.handle_response, response)
        
        -- Force explicit nullification to aid Lua Garbage Collection of 1MB+ binary strings
        response._raw_image = nil 
        
        if not hok then PT.update_status("Error: " .. tostring(herr)) end
        return
      end
      if msg_type == WebSocketMessageType.TEXT then
        if #data > PT.cfg.MAX_WS_MESSAGE_SIZE then
          PT.update_status("Message rejected: too large (" .. #data .. " bytes)")
          return
        end
        local ok, response = pcall(PT.json.decode, data)
        if ok and type(response) ~= "table" then
          ok = false
          response = "Invalid JSON structure (not a table)"
        end
        if not ok then
          PT.update_status("JSON error: " .. tostring(response))
          return
        end
        -- Any valid JSON from server proves it's alive — reset watchdog
        PT.state.last_pong = os.clock()
        local hok, herr = pcall(PT.handle_response, response)
        if not hok then PT.update_status("Error: " .. tostring(herr)) end
      end
    end,
    deflate = false,
  }
  PT.ws_handle:connect()

  -- Connection timeout
  PT.stop_connect_timer()
  PT.timers.connect = Timer{
    interval = PT.cfg.CONNECT_TIMEOUT,
    ontick = function()
      PT.stop_connect_timer()
      PT.state.connecting = false
      if not PT.state.connected then
        if PT.ws_handle then pcall(function() PT.ws_handle:close() end); PT.ws_handle = nil end
        if not PT.reconnect.manual_disconnect then
          PT.schedule_reconnect()
        else
          PT.update_status("Connection failed - is the server running?")
        end
      end
    end,
  }
  PT.timers.connect:start()
end

function PT.disconnect()
  PT.state.connecting = false
  PT.reconnect.manual_disconnect = true
  PT.timers.reconnect = PT.stop_timer(PT.timers.reconnect)
  PT.stop_connect_timer()
  if PT.ws_handle then pcall(function() PT.ws_handle:close() end); PT.ws_handle = nil end
  PT.set_connected(false)
  PT.res.requested = false
  PT.state.generating = false
  PT.state.animating = false
  PT.reset_loop_state()
  PT.stop_gen_timeout()
  PT.update_status("Disconnected")
end

-- ─── Auto-Reconnect ───────────────────────────────────────

function PT.schedule_reconnect()
  PT.timers.reconnect = PT.stop_timer(PT.timers.reconnect)
  if PT.reconnect.manual_disconnect then return end
  PT.reconnect.attempts = PT.reconnect.attempts + 1
  local delay = math.min(
    PT.cfg.RECONNECT_BASE_DELAY * (2 ^ (PT.reconnect.attempts - 1)),
    PT.cfg.RECONNECT_MAX_DELAY
  )
  PT.update_status("Reconnecting in " .. string.format("%.0f", delay) .. "s (#" .. PT.reconnect.attempts .. ")...")
  PT.timers.reconnect = Timer{
    interval = delay,
    ontick = function()
      PT.timers.reconnect = PT.stop_timer(PT.timers.reconnect)
      if PT.state.connected or PT.reconnect.manual_disconnect then return end
      PT.update_status("Reconnecting (#" .. PT.reconnect.attempts .. ")...")
      PT.connect()
    end,
  }
  PT.timers.reconnect:start()
end

-- ─── UI Button Reset (factored) ──────────────────────────

function PT.reset_ui_buttons(opts)
  if not PT.dlg then return end
  opts = opts or {}
  local gen_enabled = opts.enabled ~= false
  PT.update_action_button(PT.dlg.data.main_tabs or "tab_gen")
  PT.dlg:modify{ id = "action_btn", enabled = gen_enabled }
  PT.dlg:modify{ id = "cancel_btn", enabled = opts.cancel or false }
end

-- ─── Send ───────────────────────────────────────────────────

function PT.send(payload)
  if not PT.state.connected or PT.ws_handle == nil then
    PT.update_status("Not connected")
    return false
  end
  -- Encode first: catches depth/cycle errors from json.encode separately
  local enc_ok, encoded = pcall(PT.json.encode, payload)
  if not enc_ok then
    PT.update_status("Encode error: " .. tostring(encoded))
    return false
  end
  local ok, err = pcall(function() PT.ws_handle:sendText(encoded) end)
  if not ok then
    PT.update_status("Send failed: " .. tostring(err))
    return false
  end
  return true
end

-- ─── Resource Requests ──────────────────────────────────────

function PT.request_resources()
  PT.res.requested = true
  PT.send({ action = "list_palettes" })
  PT.send({ action = "list_loras" })
  PT.send({ action = "list_embeddings" })
  PT.send({ action = "list_presets" })
  PT.send({ action = "list_modulation_presets" })
  PT.send({ action = "list_expression_presets" })
  PT.send({ action = "list_choreography_presets" })
end

end
