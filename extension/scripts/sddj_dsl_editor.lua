--
-- SDDj — Structured Keyframe Editor Popup
--
-- Opens a Dialog{ parent=PT.dlg } with per-keyframe controls for
-- frame, prompt, transition, and parameter overrides.
-- On Apply, serializes keyframes back to DSL text.
--

return function(PT)

-- ─── Transition choices ──────────────────────────────────────

local TRANSITIONS = {
  "hard_cut", "linear_blend", "ease_in", "ease_out",
  "ease_in_out", "cubic", "slerp",
}

-- ─── Parse DSL text into editable keyframe list ──────────────

local function dsl_to_keyframes(dsl_text, total_frames, fps)
  if not dsl_text or dsl_text:match("^%s*$") then
    return {{ frame = 0, prompt = "", negative = "", transition = "hard_cut",
              transition_frames = 0, weight = 100, denoise = nil, cfg = nil, steps = nil }}
  end
  if PT.dsl_parser then
    local ok, result = pcall(PT.dsl_parser.parse, dsl_text, total_frames or 100, fps or 24)
    if ok and result and result.keyframes and #result.keyframes > 0 then
      local kfs = {}
      for _, kf in ipairs(result.keyframes) do
        kfs[#kfs + 1] = {
          frame = kf.frame or 0,
          prompt = kf.prompt or "",
          negative = kf.negative_prompt or "",
          transition = kf.transition or "hard_cut",
          transition_frames = kf.transition_frames or 0,
          weight = math.floor((kf.weight or 1.0) * 100),
          denoise = kf.denoise_strength and math.floor(kf.denoise_strength * 100) or nil,
          cfg = kf.cfg_scale and math.floor(kf.cfg_scale * 10) or nil,
          steps = kf.steps or nil,
        }
      end
      return kfs
    end
  end
  -- Fallback: single empty keyframe
  return {{ frame = 0, prompt = dsl_text:match("^%s*(.-)%s*$") or "", negative = "",
            transition = "hard_cut", transition_frames = 0, weight = 100,
            denoise = nil, cfg = nil, steps = nil }}
end

-- ─── Serialize keyframes back to DSL ─────────────────────────

local function keyframes_to_dsl(kfs)
  local lines = {}
  for _, kf in ipairs(kfs) do
    lines[#lines + 1] = "[" .. kf.frame .. "]"
    if kf.prompt and kf.prompt ~= "" then
      lines[#lines + 1] = kf.prompt
    end
    if kf.negative and kf.negative ~= "" then
      lines[#lines + 1] = "-- " .. kf.negative
    end
    if kf.transition and kf.transition ~= "hard_cut" then
      lines[#lines + 1] = "transition: " .. kf.transition
    end
    if kf.transition_frames and kf.transition_frames > 0 then
      lines[#lines + 1] = "blend: " .. kf.transition_frames
    end
    if kf.weight and kf.weight ~= 100 then
      lines[#lines + 1] = string.format("weight: %.2f", kf.weight / 100.0)
    end
    if kf.denoise then
      lines[#lines + 1] = string.format("denoise: %.2f", kf.denoise / 100.0)
    end
    if kf.cfg then
      lines[#lines + 1] = string.format("cfg: %.1f", kf.cfg / 10.0)
    end
    if kf.steps then
      lines[#lines + 1] = "steps: " .. kf.steps
    end
    lines[#lines + 1] = ""
  end
  return table.concat(lines, "\n")
end

-- ─── Read keyframes from popup dialog ────────────────────────

local function read_keyframes_from_dialog(edlg, count)
  local kfs = {}
  local d = edlg.data
  for i = 1, count do
    local p = "kf" .. i .. "_"
    kfs[#kfs + 1] = {
      frame = d[p .. "frame"] or 0,
      prompt = d[p .. "prompt"] or "",
      negative = d[p .. "negative"] or "",
      transition = d[p .. "transition"] or "hard_cut",
      transition_frames = d[p .. "blend"] or 0,
      weight = d[p .. "weight"] or 100,
      denoise = (d[p .. "denoise_on"] and d[p .. "denoise"]) or nil,
      cfg = (d[p .. "cfg_on"] and d[p .. "cfg"]) or nil,
      steps = (d[p .. "steps_on"] and d[p .. "steps"]) or nil,
    }
  end
  -- Sort by frame
  table.sort(kfs, function(a, b) return a.frame < b.frame end)
  return kfs
end

-- ─── Build a single keyframe section ─────────────────────────

local function build_kf_section(edlg, i, kf)
  local p = "kf" .. i .. "_"
  edlg:separator{ text = "Keyframe " .. i }
  edlg:number{
    id = p .. "frame",
    label = "Frame",
    text = tostring(kf.frame),
    decimals = 0,
  }
  edlg:entry{
    id = p .. "prompt",
    label = "Prompt",
    text = kf.prompt or "",
    hexpand = true,
  }
  edlg:entry{
    id = p .. "negative",
    label = "Neg.",
    text = kf.negative or "",
    hexpand = true,
  }
  edlg:combobox{
    id = p .. "transition",
    label = "Transition",
    options = TRANSITIONS,
    option = kf.transition or "hard_cut",
  }
  edlg:slider{
    id = p .. "blend",
    label = "Blend Frames",
    min = 0, max = 120,
    value = kf.transition_frames or 0,
  }
  edlg:slider{
    id = p .. "weight",
    label = "Weight (%)",
    min = 10, max = 500,
    value = kf.weight or 100,
  }
  -- Per-keyframe overrides (optional, with enable toggle)
  edlg:newrow()
  edlg:check{
    id = p .. "denoise_on",
    text = "Denoise",
    selected = kf.denoise ~= nil,
  }
  edlg:slider{
    id = p .. "denoise",
    label = "(%)",
    min = 0, max = 100,
    value = kf.denoise or 50,
  }
  edlg:check{
    id = p .. "cfg_on",
    text = "CFG",
    selected = kf.cfg ~= nil,
  }
  edlg:slider{
    id = p .. "cfg",
    label = "(×10)",
    min = 10, max = 300,
    value = kf.cfg or 50,
  }
  edlg:check{
    id = p .. "steps_on",
    text = "Steps",
    selected = kf.steps ~= nil,
  }
  edlg:slider{
    id = p .. "steps",
    label = "",
    min = 1, max = 100,
    value = kf.steps or 8,
  }
end

-- ─── Open the editor popup ───────────────────────────────────

function PT.open_schedule_editor()
  if not PT.dlg then return end
  local d = PT.dlg.data
  local dsl_text = d.generate_prompt_schedule_dsl or ""
  local total_frames = d.anim_frames or 100
  local fps = 24

  local kfs = dsl_to_keyframes(dsl_text, total_frames, fps)
  local kf_count = #kfs
  -- Cap at 8 keyframes in the editor for sanity
  if kf_count > 8 then kf_count = 8 end

  -- Build the editor dialog
  local edlg = Dialog{
    title = "Prompt Schedule Editor",
    parent = PT.dlg,
  }

  edlg:label{ text = "Define keyframes for your animation prompt schedule." }
  edlg:label{ text = "Keyframes are ordered by frame number." }

  -- Populate keyframe sections
  for i = 1, kf_count do
    build_kf_section(edlg, i, kfs[i])
  end

  -- Add/Remove keyframe buttons
  edlg:separator{}
  edlg:button{
    id = "add_kf_btn",
    text = "+ Add Keyframe",
    onclick = function()
      -- Read current state, add a new KF, rebuild
      local cur = read_keyframes_from_dialog(edlg, kf_count)
      local last_frame = #cur > 0 and cur[#cur].frame or 0
      cur[#cur + 1] = {
        frame = last_frame + 10, prompt = "", negative = "",
        transition = "ease_in_out", transition_frames = 5,
        weight = 100, denoise = nil, cfg = nil, steps = nil,
      }
      -- Close and reopen with updated data
      edlg:close()
      -- Re-serialize and re-open
      local new_dsl = keyframes_to_dsl(cur)
      PT.dlg:modify{ id = "generate_prompt_schedule_dsl", text = new_dsl }
      PT.update_schedule_state(new_dsl)
      PT.open_schedule_editor()
    end,
  }
  edlg:button{
    id = "remove_kf_btn",
    text = "- Remove Last",
    onclick = function()
      if kf_count <= 1 then return end
      local cur = read_keyframes_from_dialog(edlg, kf_count)
      table.remove(cur)
      edlg:close()
      local new_dsl = keyframes_to_dsl(cur)
      PT.dlg:modify{ id = "generate_prompt_schedule_dsl", text = new_dsl }
      PT.update_schedule_state(new_dsl)
      PT.open_schedule_editor()
    end,
  }

  -- Import raw DSL button
  edlg:separator{ text = "Import / Export" }
  edlg:button{
    id = "import_dsl_btn",
    text = "Import from File...",
    onclick = function()
      local ok, path = pcall(app.fs.fileDialog, {
        title = "Import DSL File",
        open = true,
        filetypes = { "txt" },
      })
      if not ok or not path or path == "" then return end
      local fh = io.open(path, "r")
      if not fh then
        app.alert("Cannot open file: " .. path)
        return
      end
      local content = fh:read("*a")
      fh:close()
      edlg:close()
      PT.dlg:modify{ id = "generate_prompt_schedule_dsl", text = content }
      PT.update_schedule_state(content)
      PT.open_schedule_editor()
    end,
  }
  edlg:button{
    id = "export_dsl_btn",
    text = "Export to File...",
    onclick = function()
      local cur = read_keyframes_from_dialog(edlg, kf_count)
      local dsl = keyframes_to_dsl(cur)
      local ok, path = pcall(app.fs.fileDialog, {
        title = "Export DSL File",
        save = true,
        filetypes = { "txt" },
      })
      if not ok or not path or path == "" then return end
      local fh = io.open(path, "w")
      if not fh then
        app.alert("Cannot write to: " .. path)
        return
      end
      fh:write(dsl)
      fh:close()
      app.alert("Schedule exported to:\n" .. path)
    end,
  }

  -- Apply / Cancel
  edlg:separator{}
  edlg:button{
    id = "apply_btn",
    text = "Apply",
    focus = true,
    onclick = function()
      local cur = read_keyframes_from_dialog(edlg, kf_count)
      local dsl = keyframes_to_dsl(cur)
      PT.dlg:modify{ id = "generate_prompt_schedule_dsl", text = dsl }
      PT.update_schedule_state(dsl)
      edlg:close()
    end,
  }
  edlg:button{
    id = "cancel_btn",
    text = "Cancel",
    onclick = function()
      edlg:close()
    end,
  }

  edlg:show()
end

-- ─── Schedule state management ───────────────────────────────

-- Central state table
PT.schedule_data = PT.schedule_data or {
  keyframes = {},
  valid = true,
  error_count = 0,
  warning_count = 0,
  errors = {},
  warnings = {},
}

--- Re-parse the current DSL text and update state + timeline.
function PT.update_schedule_state(dsl_text)
  dsl_text = dsl_text or ""
  local total_frames = 100
  local fps = 24
  if PT.dlg then
    local d = PT.dlg.data
    total_frames = d.anim_frames or 100
    fps = 24
  end

  local kfs = dsl_to_keyframes(dsl_text, total_frames, fps)
  PT.schedule_data.keyframes = kfs
  PT.schedule_data.total_frames = total_frames

  -- Local validation via Lua parser
  if PT.dsl_parser and dsl_text ~= "" then
    local ok, result = pcall(PT.dsl_parser.validate, dsl_text, total_frames, fps)
    if ok and result then
      PT.schedule_data.valid = result.valid
      PT.schedule_data.error_count = #(result.errors or {})
      PT.schedule_data.warning_count = #(result.warnings or {})
      PT.schedule_data.errors = result.errors or {}
      PT.schedule_data.warnings = result.warnings or {}
    else
      PT.schedule_data.valid = true
      PT.schedule_data.error_count = 0
      PT.schedule_data.warning_count = 0
    end
  else
    PT.schedule_data.valid = true
    PT.schedule_data.error_count = 0
    PT.schedule_data.warning_count = 0
    PT.schedule_data.errors = {}
    PT.schedule_data.warnings = {}
  end

  -- Update status label
  PT.update_schedule_status()

  -- Repaint timeline
  if PT.dlg then
    pcall(function() PT.dlg:repaint() end)
  end
end

--- Update the parse status label in the main dialog.
function PT.update_schedule_status()
  if not PT.dlg then return end
  local sd = PT.schedule_data
  local kf_count = #sd.keyframes
  local status
  if kf_count == 0 then
    status = "No schedule"
  elseif sd.valid then
    status = string.format("OK — %d keyframe%s", kf_count, kf_count > 1 and "s" or "")
  else
    local first_err = sd.errors[1]
    local msg = first_err and (first_err.code .. ": " .. first_err.message) or "Error"
    status = string.format("ERR (%d) — %s", sd.error_count, msg)
  end
  pcall(function()
    PT.dlg:modify{ id = "schedule_status", text = status }
  end)
end

-- ─── Timeline canvas painter ─────────────────────────────────

--- Paint the timeline canvas.
function PT.paint_schedule_timeline(ev)
  local gc = ev.context
  local w = gc.width
  local h = gc.height
  local sd = PT.schedule_data
  local kfs = sd.keyframes
  local total = sd.total_frames or 100
  if total < 1 then total = 100 end

  -- Background
  gc.color = Color{ r = 32, g = 32, b = 38, a = 255 }
  gc:fillRect(Rectangle(0, 0, w, h))

  -- Track line
  local track_y = math.floor(h * 0.5)
  local track_h = math.max(2, math.floor(h * 0.15))
  gc.color = Color{ r = 60, g = 60, b = 70, a = 255 }
  gc:fillRect(Rectangle(2, track_y - track_h, w - 4, track_h * 2))

  if #kfs == 0 then
    gc.color = Color{ r = 120, g = 120, b = 130, a = 255 }
    gc:fillText("No schedule", 4, 4)
    return
  end

  -- Keyframe colors (cycling palette)
  local KF_COLORS = {
    Color{ r = 100, g = 180, b = 255, a = 255 },  -- blue
    Color{ r = 255, g = 140, b = 80,  a = 255 },   -- orange
    Color{ r = 120, g = 220, b = 140, a = 255 },  -- green
    Color{ r = 220, g = 120, b = 200, a = 255 },  -- pink
    Color{ r = 255, g = 220, b = 80,  a = 255 },   -- yellow
    Color{ r = 140, g = 160, b = 255, a = 255 },  -- indigo
    Color{ r = 80,  g = 220, b = 220, a = 255 },   -- cyan
    Color{ r = 255, g = 100, b = 100, a = 255 },  -- red
  }

  -- Draw transition zones between keyframes
  for i = 2, #kfs do
    local prev = kfs[i - 1]
    local cur = kfs[i]
    local tf = cur.transition_frames or 0
    if tf > 0 and cur.transition ~= "hard_cut" then
      local blend_start = cur.frame - tf
      if blend_start < prev.frame then blend_start = prev.frame end
      local x1 = math.floor((blend_start / total) * (w - 4)) + 2
      local x2 = math.floor((cur.frame / total) * (w - 4)) + 2
      -- Transition zone: gradient-like fill using dithered blend
      gc.color = Color{ r = 70, g = 70, b = 90, a = 180 }
      gc:fillRect(Rectangle(x1, track_y - track_h - 2, x2 - x1, track_h * 2 + 4))

      -- Transition type label
      local label = cur.transition:sub(1, 3):upper()
      gc.color = Color{ r = 180, g = 180, b = 200, a = 200 }
      local lx = math.floor((x1 + x2) / 2) - 4
      gc:fillText(label, lx, track_y - track_h - 8)
    end
  end

  -- Draw keyframe markers
  for i, kf in ipairs(kfs) do
    local x = math.floor((kf.frame / total) * (w - 4)) + 2
    local col = KF_COLORS[((i - 1) % #KF_COLORS) + 1]

    -- Vertical marker
    gc.color = col
    gc:fillRect(Rectangle(x - 1, 2, 3, h - 4))

    -- Diamond marker at track center
    gc:beginPath()
    gc:moveTo(x, track_y - 4)
    gc:lineTo(x + 4, track_y)
    gc:lineTo(x, track_y + 4)
    gc:lineTo(x - 4, track_y)
    gc:closePath()
    gc:fill()

    -- Frame number label
    gc.color = Color{ r = 200, g = 200, b = 210, a = 255 }
    gc:fillText(tostring(kf.frame), x - 3, h - 10)
  end

  -- Border
  gc.color = Color{ r = 80, g = 80, b = 90, a = 255 }
  gc:strokeRect(Rectangle(0, 0, w, h))
end

-- ─── Timeline click handler ──────────────────────────────────

function PT.on_timeline_click(ev)
  local w = ev.context and ev.context.width or 300
  local sd = PT.schedule_data
  local kfs = sd.keyframes
  local total = sd.total_frames or 100
  if #kfs == 0 or total < 1 then return end

  -- Find nearest keyframe to click position
  local click_frame = math.floor((ev.x / w) * total)
  local nearest_i = 1
  local nearest_dist = math.abs(kfs[1].frame - click_frame)
  for i = 2, #kfs do
    local dist = math.abs(kfs[i].frame - click_frame)
    if dist < nearest_dist then
      nearest_i = i
      nearest_dist = dist
    end
  end

  -- Show info in status label
  local kf = kfs[nearest_i]
  local info = string.format("KF %d [%d]: %s | %s",
    nearest_i, kf.frame,
    (kf.prompt or ""):sub(1, 30),
    kf.transition or "hard_cut")
  if kf.denoise then info = info .. string.format(" | D=%.0f%%", kf.denoise) end
  if kf.cfg then info = info .. string.format(" | CFG=%.1f", (kf.cfg or 50) / 10.0) end
  if kf.steps then info = info .. " | S=" .. kf.steps end

  pcall(function()
    PT.dlg:modify{ id = "schedule_status", text = info }
  end)
end

-- ─── Preset management popup ─────────────────────────────────

function PT.open_schedule_presets()
  if not PT.dlg then return end

  local pdlg = Dialog{
    title = "Prompt Schedule Presets",
    parent = PT.dlg,
  }

  pdlg:combobox{
    id = "preset_name",
    label = "Preset",
    options = PT.schedule_preset_list or { "(none)" },
    option = PT.schedule_preset_list and PT.schedule_preset_list[1] or "(none)",
  }

  pdlg:button{
    id = "load_btn",
    text = "Load",
    onclick = function()
      local name = pdlg.data.preset_name
      if name and name ~= "(none)" and PT.state and PT.state.connected then
        PT.send({ action = "get_prompt_schedule", prompt_schedule_name = name })
        pdlg:close()
      end
    end,
  }

  pdlg:entry{
    id = "save_name",
    label = "Save As",
    text = "",
    hexpand = true,
  }

  pdlg:button{
    id = "save_btn",
    text = "Save",
    onclick = function()
      local name = pdlg.data.save_name or ""
      if name == "" then
        app.alert("Enter a name for the preset.")
        return
      end
      if PT.state and PT.state.connected then
        local dsl_text = PT.dlg.data.generate_prompt_schedule_dsl or ""
        local kfs_data = dsl_to_keyframes(dsl_text, 100, 24)
        -- Convert to server format
        local server_kfs = {}
        for _, kf in ipairs(kfs_data) do
          server_kfs[#server_kfs + 1] = {
            frame = kf.frame,
            prompt = kf.prompt,
            negative_prompt = kf.negative,
            transition = kf.transition,
            transition_frames = kf.transition_frames,
            weight = (kf.weight or 100) / 100.0,
          }
        end
        PT.send({
          action = "save_prompt_schedule",
          prompt_schedule_name = name,
          prompt_schedule_data = { keyframes = server_kfs },
        })
        pdlg:close()
      end
    end,
  }

  pdlg:button{
    id = "delete_btn",
    text = "Delete",
    onclick = function()
      local name = pdlg.data.preset_name
      if name and name ~= "(none)" and PT.state and PT.state.connected then
        PT.send({ action = "delete_prompt_schedule", prompt_schedule_name = name })
        pdlg:close()
      end
    end,
  }

  pdlg:button{
    id = "refresh_btn",
    text = "Refresh",
    onclick = function()
      if PT.state and PT.state.connected then
        PT.send({ action = "list_prompt_schedules" })
      end
    end,
  }

  pdlg:button{
    id = "close_btn",
    text = "Close",
    onclick = function() pdlg:close() end,
  }

  -- Auto-refresh on open
  if PT.state and PT.state.connected then
    PT.send({ action = "list_prompt_schedules" })
  end

  pdlg:show()
end

end
