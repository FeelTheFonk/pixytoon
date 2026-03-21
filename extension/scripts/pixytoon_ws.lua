--
-- PixyToon — WebSocket Transport & Connection Management
--

return function(PT)

-- ─── Status ─────────────────────────────────────────────────

function PT.update_status(text)
  if PT.dlg then PT.dlg:modify{ id = "status", text = text } end
end

-- ─── Heartbeat ──────────────────────────────────────────────

function PT.stop_heartbeat()
  PT.timers.heartbeat = PT.stop_timer(PT.timers.heartbeat)
end

function PT.start_heartbeat()
  PT.stop_heartbeat()
  PT.timers.heartbeat = Timer{
    interval = PT.cfg.HEARTBEAT_INTERVAL,
    ontick = function()
      if PT.state.connected and PT.ws_handle
          and not PT.state.generating and not PT.state.animating then
        pcall(function() PT.ws_handle:sendText('{"action":"ping"}') end)
      end
    end,
  }
  PT.timers.heartbeat:start()
end

-- ─── Generation Timeout ─────────────────────────────────────

function PT.stop_gen_timeout()
  PT.timers.gen_timeout = PT.stop_timer(PT.timers.gen_timeout)
end

function PT.start_gen_timeout()
  PT.stop_gen_timeout()
  PT.timers.gen_timeout = Timer{
    interval = PT.cfg.GEN_TIMEOUT,
    ontick = function()
      PT.stop_gen_timeout()
      if PT.state.generating or PT.state.animating then
        PT.state.generating = false
        PT.state.animating = false
        PT.loop.mode = false
        PT.loop.random_mode = false
        PT.state.gen_step_start = nil
        if PT.dlg then
          PT.update_status("Timed out — no response from server")
          PT.dlg:modify{ id = "generate_btn", text = "GENERATE", enabled = not PT.live.mode }
          PT.dlg:modify{ id = "animate_btn", enabled = not PT.live.mode }
          PT.dlg:modify{ id = "cancel_btn", enabled = false }
        end
      end
    end,
  }
  PT.timers.gen_timeout:start()
end

-- ─── Connection State ───────────────────────────────────────

function PT.set_connected(is_connected)
  PT.state.connected = is_connected
  if is_connected then PT.start_heartbeat() else PT.stop_heartbeat() end
  if not PT.dlg then return end

  if is_connected then
    PT.dlg:modify{ id = "connect_btn", text = "Disconnect" }
    PT.dlg:modify{ id = "generate_btn", enabled = true }
    PT.dlg:modify{ id = "animate_btn", enabled = true }
    PT.dlg:modify{ id = "live_btn", enabled = true }
  else
    PT.dlg:modify{ id = "connect_btn", text = "Connect" }
    PT.dlg:modify{ id = "generate_btn", text = "GENERATE", enabled = false }
    PT.dlg:modify{ id = "cancel_btn", enabled = false }
    PT.dlg:modify{ id = "animate_btn", enabled = false }
    PT.dlg:modify{ id = "live_btn", enabled = false }
    PT.dlg:modify{ id = "live_btn", text = "START LIVE" }
    PT.dlg:modify{ id = "live_accept_btn", visible = false }
    PT.dlg:modify{ id = "live_send_btn", visible = false }
    if PT.state.generating then PT.state.generating = false end
    if PT.state.animating then PT.state.animating = false end
    PT.stop_live_timer()
    PT.stop_gen_timeout()
    PT.live.mode = false
    PT.live.request_inflight = false
    PT.loop.mode = false
    PT.loop.random_mode = false
    PT.state.cancel_pending = false
  end
end

-- ─── Connect / Disconnect ───────────────────────────────────

function PT.stop_connect_timer()
  PT.timers.connect = PT.stop_timer(PT.timers.connect)
end

function PT.connect()
  if PT.ws_handle then pcall(function() PT.ws_handle:close() end); PT.ws_handle = nil end
  PT.update_status("Connecting...")
  PT.ws_handle = WebSocket{
    url = PT.cfg.DEFAULT_SERVER_URL,
    onreceive = function(msg_type, data)
      if msg_type == WebSocketMessageType.OPEN then
        PT.stop_connect_timer()
        PT.set_connected(true)
        PT.update_status("Connected")
        pcall(function() PT.ws_handle:sendText(PT.json.encode({ action = "ping" })) end)
        -- Auto re-request resources on reconnect
        if PT.res.requested then
          PT.send({ action = "list_loras" })
          PT.send({ action = "list_palettes" })
          PT.send({ action = "list_embeddings" })
          PT.send({ action = "list_presets" })
        end
        return
      end
      if msg_type == WebSocketMessageType.CLOSE then
        PT.set_connected(false)
        PT.res.requested = false
        PT.update_status("Disconnected (server closed)")
        PT.ws_handle = nil
        return
      end
      if msg_type == WebSocketMessageType.TEXT then
        local ok, response = pcall(PT.json.decode, data)
        if not ok then
          PT.update_status("JSON error: " .. tostring(response))
          return
        end
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
      if not PT.state.connected then
        if PT.ws_handle then pcall(function() PT.ws_handle:close() end); PT.ws_handle = nil end
        PT.update_status("Connection failed - is the server running?")
      end
    end,
  }
  PT.timers.connect:start()
end

function PT.disconnect()
  PT.stop_connect_timer()
  if PT.ws_handle then pcall(function() PT.ws_handle:close() end); PT.ws_handle = nil end
  PT.set_connected(false)
  PT.res.requested = false
  PT.anim.layer = nil
  PT.anim.start_frame = 0
  PT.anim.frame_count = 0
  PT.anim.base_seed = 0
  PT.state.generating = false
  PT.state.animating = false
  PT.loop.mode = false
  PT.loop.random_mode = false
  PT.stop_live_timer()
  PT.stop_gen_timeout()
  PT.live.mode = false
  PT.live.request_inflight = false
  PT.live.pending_send = false
  -- Clean up preview layer from sprite
  if PT.live.preview_layer then
    local spr = app.sprite
    if spr then
      pcall(function()
        local cel = PT.live.preview_layer:cel(app.frame)
        if cel then spr:deleteCel(cel) end
        spr:deleteLayer(PT.live.preview_layer)
      end)
    end
  end
  PT.live.preview_layer = nil
  PT.live.preview_sprite = nil
  PT.update_status("Disconnected")
end

-- ─── Send ───────────────────────────────────────────────────

function PT.send(payload)
  if not PT.state.connected or PT.ws_handle == nil then
    PT.update_status("Not connected")
    return false
  end
  local ok, err = pcall(function() PT.ws_handle:sendText(PT.json.encode(payload)) end)
  if not ok then
    PT.update_status("Send failed: " .. tostring(err))
    return false
  end
  return true
end

-- Binary frame: 4-byte LE header length + JSON header + raw PNG data.
-- Eliminates Lua base64 encoding overhead (~200ms on 512×512).
function PT.send_live_binary(header_table, png_data)
  if not PT.state.connected or PT.ws_handle == nil then return false end
  local header_json = PT.json.encode(header_table)
  local header_len = #header_json
  local len_bytes = string.char(
    header_len % 256,
    math.floor(header_len / 256) % 256,
    math.floor(header_len / 65536) % 256,
    math.floor(header_len / 16777216) % 256
  )
  local ok, err = pcall(function()
    PT.ws_handle:sendBinary(len_bytes .. header_json .. png_data)
  end)
  if not ok then
    PT.update_status("Binary send failed: " .. tostring(err))
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
end

end
