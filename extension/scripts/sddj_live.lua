--
-- SDDj — Live Paint System (Event-Driven)
--
-- Uses sprite.events:on('change') instead of polling for zero-interference
-- live painting. Supports Auto (event-driven) and Manual (F5 hotkey) modes.
--

return function(PT)

-- ─── Dirty Region Detection ─────────────────────────────────

function PT.detect_dirty_region(prev, curr)
  local w, h = curr.width, curr.height
  local min_x, min_y = w, h
  local max_x, max_y = 0, 0
  local step = math.max(1, math.floor(math.min(w, h) / PT.cfg.DIRTY_STEP_DIVISOR))
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

-- ─── Sprite Event Listeners ──────────────────────────────────

function PT.register_sprite_listener()
  PT.unregister_sprite_listener()
  local spr = app.sprite
  if not spr then return end

  PT.live.monitored_sprite = spr
  PT.live.change_listener = spr.events:on('change', function(ev)
    -- Guard: ignore our own modifications (preview import)
    if PT.live.importing then return end
    -- Guard: ignore undo/redo
    if ev and ev.fromUndo then return end
    -- Guard: mode, connection
    if not PT.live.mode or not PT.state.connected then return end
    -- Auto mode only
    if not PT.live.auto_mode then return end

    PT.live.pending_send = true
    -- Debounce: wait for drawing to pause before sending
    PT.live.debounce_timer = PT.stop_timer(PT.live.debounce_timer)
    PT.live.debounce_timer = Timer{
      interval = PT.cfg.LIVE_STROKE_DEBOUNCE,
      ontick = function()
        PT.live.debounce_timer = PT.stop_timer(PT.live.debounce_timer)
        PT.live_send_now()
      end,
    }
    PT.live.debounce_timer:start()
  end)

  -- Listen for active sprite changes
  PT.live.site_listener = app.events:on('sitechange', function()
    if not PT.live.mode then return end
    if app.sprite ~= PT.live.monitored_sprite then
      PT.register_sprite_listener()
    end
  end)
end

function PT.unregister_sprite_listener()
  if PT.live.change_listener and PT.live.monitored_sprite then
    pcall(function()
      PT.live.monitored_sprite.events:off(PT.live.change_listener)
    end)
  end
  PT.live.change_listener = nil

  if PT.live.site_listener then
    pcall(function() app.events:off(PT.live.site_listener) end)
  end
  PT.live.site_listener = nil

  PT.live.monitored_sprite = nil
  PT.live.debounce_timer = PT.stop_timer(PT.live.debounce_timer)
  PT.live.pending_send = false
end

-- ─── Core: Send Live Frame ───────────────────────────────────

function PT.live_send_now()
  if not PT.live.mode or not PT.state.connected then return end
  if PT.live.request_inflight then
    PT.live.pending_send = true
    return
  end
  PT.live.pending_send = false

  local spr = app.sprite
  if not spr or not app.frame then return end

  -- Clean capture: hide preview layer
  local was_visible = true
  if PT.live.preview_layer then
    local ok_vis, vis = pcall(function() return PT.live.preview_layer.isVisible end)
    if ok_vis then
      was_visible = vis
      PT.live.preview_layer.isVisible = false
    else
      PT.live.preview_layer = nil
    end
  end

  local flat_img = Image(spr.spec)
  flat_img:drawSprite(spr, app.frame)

  if PT.live.preview_layer then
    pcall(function() PT.live.preview_layer.isVisible = was_visible end)
  end

  -- ROI detection
  local roi = nil
  if PT.live.prev_canvas
      and PT.live.prev_canvas.width == flat_img.width
      and PT.live.prev_canvas.height == flat_img.height then
    roi = PT.detect_dirty_region(PT.live.prev_canvas, flat_img)
  end
  PT.live.prev_canvas = flat_img:clone()

  -- Binary send
  local png_data = PT.image_to_png_bytes(flat_img)
  if not png_data then return end
  PT.live.frame_id = PT.live.frame_id + 1
  local header = {
    action = "realtime_frame",
    frame_id = PT.live.frame_id,
  }
  if roi then
    header.roi_x = roi.x
    header.roi_y = roi.y
    header.roi_w = roi.w
    header.roi_h = roi.h
  end

  local sent = PT.send_live_binary(header, png_data)
  if sent then
    PT.live.request_inflight = true
    PT.live.inflight_time = os.clock()
    if PT.dlg then PT.update_status("Live — processing...") end
  end
end

-- ─── Timer Lifecycle ────────────────────────────────────────

function PT.stop_live_timer()
  PT.unregister_sprite_listener()
  PT.live.timer = PT.stop_timer(PT.live.timer)
  PT.live.request_inflight = false
  PT.live.inflight_time = nil
  PT.live.debounce_timer = PT.stop_timer(PT.live.debounce_timer)
  PT.live.pending_send = false
  PT.live.slider_debounce = PT.stop_timer(PT.live.slider_debounce)
end

function PT.start_live_timer()
  PT.stop_live_timer()
  PT.live.prev_canvas = nil
  PT.live.pending_send = false

  -- Register sprite event listeners
  PT.register_sprite_listener()

  -- Minimal watchdog timer (prompt changes, inflight timeout, sprite closed)
  PT.live.timer = Timer{
    interval = PT.cfg.LIVE_WATCHDOG_INTERVAL,
    ontick = function()
      if not PT.live.mode then return end

      -- Disconnect guard: auto-stop live if connection lost
      if not PT.state.connected then
        pcall(function() PT.send({ action = "realtime_stop" }) end)
        PT.stop_live_mode()
        return
      end

      -- Inflight timeout guard
      if PT.live.request_inflight and PT.live.inflight_time then
        if (os.clock() - PT.live.inflight_time) > PT.cfg.LIVE_INFLIGHT_TIMEOUT then
          PT.live.request_inflight = false
          PT.live.inflight_time = nil
          if PT.dlg then PT.update_status("Live — timeout, retrying...") end
          if PT.live.pending_send then PT.live_send_now() end
        end
      end

      -- Prompt change detection
      if not PT.live.request_inflight and PT.dlg then
        local current_prompt = PT.dlg.data.prompt
        if current_prompt ~= PT.live.last_prompt then
          PT.live.last_prompt = current_prompt
          PT.send({ action = "realtime_update", prompt = current_prompt })
        end
      end

      -- Sprite closed detection
      if app.sprite == nil or app.frame == nil then
        pcall(function() PT.send({ action = "realtime_stop" }) end)
        PT.stop_live_mode()
        return
      end
    end,
  }
  PT.live.timer:start()
end

-- ─── Stop Live Mode (factored helper) ───────────────────────

function PT.stop_live_mode()
  PT.stop_live_timer()
  PT.live.mode = false
  PT.live.request_inflight = false
  PT.live.inflight_time = nil
  PT.live.preview_layer = nil
  PT.live.preview_sprite = nil
  PT.live.pending_send = false
  if PT.dlg then
    PT.update_status("Live stopped")
    PT.dlg:modify{ id = "live_btn", text = "START LIVE" }
    PT.dlg:modify{ id = "live_accept_btn", visible = false }
    PT.dlg:modify{ id = "live_send_btn", visible = false }
    PT.dlg:modify{ id = "generate_btn", enabled = true }
    PT.dlg:modify{ id = "animate_btn", enabled = true }
  end
end

end
