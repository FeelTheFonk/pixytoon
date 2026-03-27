--
-- SDDj DSL Parser v2.0 — Prompt Schedule DSL
--
-- Full rewrite: proper tokenizer, structured error accumulation,
-- multi-line prompt support, validated ranges, file: sandboxing.
-- Follows grammar in docs/PROMPT_SCHEDULE_DSL.md
--

local M = {}

-- ─── Transition Types ──────────────────────────────────────

local VALID_TRANSITIONS = {
  hard_cut = true,
  blend = true,
  linear_blend = true,
  ease_in = true,
  ease_out = true,
  ease_in_out = true,
  cubic = true,
  slerp = true,
}

-- ─── Pattern Definitions ───────────────────────────────────

-- Time markers: [0], [50%], [5s], [5.5s]
local function match_time_marker(line)
  -- Percentage: [50%] or [50.5%]
  local pct = line:match("^%s*%[%s*([%d%.]+)%s*%%%s*%]%s*$")
  if pct then return "percent", tonumber(pct) end
  -- Seconds: [5s] or [2.5s]
  local secs = line:match("^%s*%[%s*([%d%.]+)%s*s%s*%]%s*$")
  if secs then return "seconds", tonumber(secs) end
  -- Absolute frame: [0], [24]
  local frame = line:match("^%s*%[%s*(%d+)%s*%]%s*$")
  if frame then return "frame", tonumber(frame) end
  return nil, nil
end

-- Directives
local function match_directive(line)
  -- Transition
  local tr = line:match("^%s*transition:%s*(%S+)%s*$")
  if tr then return "transition", tr:lower() end
  -- Blend frames
  local bf = line:match("^%s*blend:%s*(%d+)%s*$")
  if bf then return "blend", tonumber(bf) end
  -- Weight (static or animated)
  local w1, w2 = line:match("^%s*weight:%s*([%d%.]+)%s*%->%s*([%d%.]+)%s*$")
  if w1 then return "weight_animated", { tonumber(w1), tonumber(w2) } end
  local w = line:match("^%s*weight:%s*([%d%.]+)%s*$")
  if w then return "weight", tonumber(w) end
  -- Denoise
  local dn = line:match("^%s*denoise:%s*([%d%.]+)%s*$")
  if dn then return "denoise", tonumber(dn) end
  -- CFG
  local cfg = line:match("^%s*cfg:%s*([%d%.]+)%s*$")
  if cfg then return "cfg", tonumber(cfg) end
  -- Steps
  local st = line:match("^%s*steps:%s*(%d+)%s*$")
  if st then return "steps", tonumber(st) end
  -- Negative prompt
  local neg = line:match("^%s*%-%-%s*(.*)$")
  if neg then return "negative", neg end
  return nil, nil
end

-- ─── Time Resolution ───────────────────────────────────────

local function resolve_time(time_type, value, total_frames, fps, line_num, errors)
  if time_type == "percent" then
    if value < 0 or value > 100 then
      errors[#errors + 1] = {
        line = line_num, code = "E001",
        message = string.format("Percentage %.1f%% out of range [0, 100]", value),
      }
      value = math.max(0, math.min(100, value))
    end
    return math.min(total_frames - 1, math.max(0, math.floor(value / 100 * total_frames)))
  elseif time_type == "seconds" then
    if value < 0 then
      errors[#errors + 1] = {
        line = line_num, code = "E001",
        message = string.format("Time %.2fs is negative", value),
      }
      value = 0
    end
    return math.min(total_frames - 1, math.max(0, math.floor(value * fps)))
  else -- "frame"
    if value < 0 then
      errors[#errors + 1] = {
        line = line_num, code = "E001",
        message = string.format("Frame %d is negative", value),
      }
      value = 0
    end
    if value >= total_frames then
      errors[#errors + 1] = {
        line = line_num, code = "E001",
        message = string.format("Frame %d exceeds total frames %d", value, total_frames),
      }
      value = math.min(total_frames - 1, value)
    end
    return value
  end
end

-- ─── Builder Finalization ──────────────────────────────────

local function finalize_builder(builder)
  local prompt = table.concat(builder.prompt_lines, ", ")
  local negative = table.concat(builder.negative_lines, ", ")
  return {
    frame = builder.frame,
    prompt = prompt,
    negative_prompt = negative,
    weight = builder.weight,
    weight_end = builder.weight_end,
    transition = builder.transition,
    transition_frames = builder.transition_frames,
    denoise_strength = builder.denoise_strength,
    cfg_scale = builder.cfg_scale,
    steps = builder.steps,
  }
end

-- ─── File Reference Handling ───────────────────────────────

local function resolve_file_ref(path, base_dir, errors)
  -- Security: reject path traversal
  if path:find("%.%.") then
    errors[#errors + 1] = {
      line = 1, code = "E010",
      message = "Path traversal rejected: " .. path,
    }
    return nil
  end
  -- Reject absolute paths
  if path:sub(1, 1) == "/" or path:sub(1, 1) == "\\" then
    errors[#errors + 1] = {
      line = 1, code = "E010",
      message = "Absolute path rejected: " .. path,
    }
    return nil
  end
  -- Windows absolute path
  if #path >= 2 and path:sub(2, 2) == ":" then
    errors[#errors + 1] = {
      line = 1, code = "E010",
      message = "Absolute path rejected: " .. path,
    }
    return nil
  end

  local full_path
  if base_dir then
    -- Use app.fs if available (Aseprite)
    if app and app.fs and app.fs.joinPath then
      full_path = app.fs.joinPath(base_dir, path)
    else
      full_path = base_dir .. "/" .. path
    end
  else
    full_path = path
  end

  local f = io.open(full_path, "r")
  if not f then
    errors[#errors + 1] = {
      line = 1, code = "E011",
      message = "File not found: " .. path,
    }
    return nil
  end
  local content = f:read("*a")
  f:close()
  return content
end

-- ─── Main Parser ───────────────────────────────────────────

--- Parse a DSL text into a prompt schedule.
--- @param dsl_text string  The raw DSL text
--- @param total_frames number  Total frames in the animation
--- @param fps number  Frames per second (default 24)
--- @param base_dir string|nil  Base directory for file: references
--- @return table schedule  {keyframes={...}, has_auto=bool, errors={...}, warnings={...}}
function M.parse(dsl_text, total_frames, fps, base_dir)
  fps = fps or 24
  total_frames = math.max(1, total_frames or 1)

  local errors = {}
  local warnings = {}
  local has_auto = false

  if type(dsl_text) ~= "string" or dsl_text:match("^%s*$") then
    return {
      keyframes = {},
      has_auto = false,
      errors = {},
      warnings = { { line = nil, code = "W002", message = "Empty schedule" } },
    }
  end

  -- Handle file: reference (single-line DSL)
  local trimmed = dsl_text:match("^%s*(.-)%s*$")
  local file_path = trimmed:match("^file:%s*(.+)$")
  if file_path then
    file_path = file_path:match("^%s*(.-)%s*$")
    local content = resolve_file_ref(file_path, base_dir, errors)
    if not content then
      return { keyframes = {}, has_auto = false, errors = errors, warnings = warnings }
    end
    dsl_text = content
  end

  -- Split into lines
  local lines = {}
  for line in (dsl_text .. "\n"):gmatch("([^\n]*)\n") do
    lines[#lines + 1] = line
  end

  -- Parse
  local builders = {}
  local current = nil

  for line_num, raw_line in ipairs(lines) do
    local line = raw_line:gsub("\r$", "")  -- strip CR

    -- Blank
    if line:match("^%s*$") then
      goto continue
    end

    -- Comment
    if line:match("^%s*#") then
      goto continue
    end

    -- Auto directive
    if line:match("^%s*{auto}%s*$") then
      has_auto = true
      goto continue
    end

    -- Time marker
    local time_type, time_value = match_time_marker(line)
    if time_type then
      -- Finalize previous builder
      if current then
        builders[#builders + 1] = current
      end
      local frame = resolve_time(time_type, time_value, total_frames, fps, line_num, errors)
      current = {
        line = line_num,
        frame = frame,
        prompt_lines = {},
        negative_lines = {},
        transition = "hard_cut",
        transition_frames = 0,
        weight = 1.0,
        weight_end = nil,
        denoise_strength = nil,
        cfg_scale = nil,
        steps = nil,
      }
      goto continue
    end

    -- Must be inside a keyframe block
    if not current then
      local stripped = line:match("^%s*(.-)%s*$")
      if stripped ~= "" then
        errors[#errors + 1] = {
          line = line_num, code = "E012",
          message = "Content before first time marker: " .. stripped:sub(1, 50),
        }
      end
      goto continue
    end

    -- Try directives
    local dir_type, dir_value = match_directive(line)
    if dir_type then
      if dir_type == "transition" then
        if not VALID_TRANSITIONS[dir_value] then
          errors[#errors + 1] = {
            line = line_num, code = "E005",
            message = "Invalid transition type: " .. dir_value,
          }
        else
          current.transition = dir_value
        end
      elseif dir_type == "blend" then
        if dir_value > 120 then
          errors[#errors + 1] = {
            line = line_num, code = "E004",
            message = string.format("Transition frames %d exceeds maximum (120)", dir_value),
          }
        else
          current.transition_frames = dir_value
        end
      elseif dir_type == "weight" then
        if dir_value < 0.1 or dir_value > 5.0 then
          errors[#errors + 1] = {
            line = line_num, code = "E006",
            message = string.format("Weight %.2f out of range [0.1, 5.0]", dir_value),
          }
        else
          current.weight = dir_value
          if dir_value > 2.0 then
            warnings[#warnings + 1] = {
              line = line_num, code = "W004",
              message = string.format("Weight %.2f > 2.0 may cause artifacts", dir_value),
              severity = "warning",
            }
          end
        end
      elseif dir_type == "weight_animated" then
        local w1, w2 = dir_value[1], dir_value[2]
        local valid = true
        if w1 < 0.1 or w1 > 5.0 then
          errors[#errors + 1] = { line = line_num, code = "E006",
            message = string.format("Weight start %.2f out of range [0.1, 5.0]", w1) }
          valid = false
        end
        if w2 < 0.1 or w2 > 5.0 then
          errors[#errors + 1] = { line = line_num, code = "E006",
            message = string.format("Weight end %.2f out of range [0.1, 5.0]", w2) }
          valid = false
        end
        if valid then
          current.weight = w1
          current.weight_end = w2
        end
      elseif dir_type == "denoise" then
        if dir_value < 0.0 or dir_value > 1.0 then
          errors[#errors + 1] = { line = line_num, code = "E007",
            message = string.format("Denoise %.3f out of range [0.0, 1.0]", dir_value) }
        else
          current.denoise_strength = dir_value
        end
      elseif dir_type == "cfg" then
        if dir_value < 1.0 or dir_value > 30.0 then
          errors[#errors + 1] = { line = line_num, code = "E008",
            message = string.format("CFG %.1f out of range [1.0, 30.0]", dir_value) }
        else
          current.cfg_scale = dir_value
        end
      elseif dir_type == "steps" then
        if dir_value < 1 or dir_value > 150 then
          errors[#errors + 1] = { line = line_num, code = "E009",
            message = string.format("Steps %d out of range [1, 150]", dir_value) }
        else
          current.steps = dir_value
        end
      elseif dir_type == "negative" then
        local neg_text = dir_value:match("^%s*(.-)%s*$")
        if neg_text ~= "" then
          current.negative_lines[#current.negative_lines + 1] = neg_text
        end
      end
      goto continue
    end

    -- Prompt text (anything else)
    local stripped = line:match("^%s*(.-)%s*$")
    if stripped ~= "" then
      current.prompt_lines[#current.prompt_lines + 1] = stripped
    end

    ::continue::
  end

  -- Finalize last builder
  if current then
    builders[#builders + 1] = current
  end

  -- Build keyframes
  local keyframes = {}
  for _, b in ipairs(builders) do
    keyframes[#keyframes + 1] = finalize_builder(b)
  end

  -- Validate chronological order
  for i = 2, #keyframes do
    if keyframes[i].frame <= keyframes[i - 1].frame then
      errors[#errors + 1] = {
        line = builders[i].line, code = "E003",
        message = string.format("Keyframe at frame %d is not after previous frame %d",
          keyframes[i].frame, keyframes[i - 1].frame),
      }
    end
  end

  -- Check first keyframe
  if #keyframes > 0 and keyframes[1].frame ~= 0 then
    warnings[#warnings + 1] = {
      line = builders[1].line, code = "W001",
      message = string.format("First keyframe at frame %d, not frame 0", keyframes[1].frame),
      severity = "warning",
    }
  end

  -- Validate transition windows
  for i = 2, #keyframes do
    local kf = keyframes[i]
    if kf.transition_frames > 0 then
      local gap = kf.frame - keyframes[i - 1].frame
      if kf.transition_frames > gap then
        errors[#errors + 1] = {
          line = builders[i].line, code = "E004",
          message = string.format("Transition frames (%d) exceeds gap to previous keyframe (%d frames)",
            kf.transition_frames, gap),
        }
      end
    end
  end

  return {
    keyframes = keyframes,
    has_auto = has_auto,
    errors = errors,
    warnings = warnings,
  }
end


--- Validate DSL text without parsing into a full schedule.
--- Returns {valid=bool, error_count=int, warning_count=int, keyframe_count=int, errors={}, warnings={}}
function M.validate(dsl_text, total_frames, fps, base_dir)
  local result = M.parse(dsl_text, total_frames, fps, base_dir)
  return {
    valid = #result.errors == 0,
    error_count = #result.errors,
    warning_count = #result.warnings,
    keyframe_count = #result.keyframes,
    errors = result.errors,
    warnings = result.warnings,
    has_auto = result.has_auto,
  }
end

return M
