-- sddj_dsl_parser.lua
local M = {}

-- Helper to trim string
local function trim(s)
  return (s:gsub("^%s*(.-)%s*$", "%1"))
end

function M.parse(input_str, total_frames, fps)
  local schedule = {
    keyframes = {},
    default_prompt = ""
  }
  
  if not input_str or input_str == "" then
    return schedule
  end
  
  if input_str:match("{auto}") then
    schedule.auto_fill = true
  end
  
  -- Support file loading transparently. Note: in Aseprite env 'app.fs' or 'io.open' could be used.
  if input_str:match("^%s*file:%s*(.+)") then
     local path = trim(input_str:match("^%s*file:%s*(.+)"))
     local f = io.open(path, "r")
     if f then
       input_str = f:read("*a")
       f:close()
     else
       print("Warning: unable to read scheduling file " .. path)
       input_str = ""
     end
  end

  local segments = {}
  -- Split by |
  for segment in string.gmatch(input_str .. "|", "(.-)|") do
    if trim(segment) ~= "" then
      table.insert(segments, trim(segment))
    end
  end
  
  for _, seg in ipairs(segments) do
    -- Match generic structure: [time][options]: [prompt] [-] [negative]
    local time_part, rest = seg:match("^([^:]+):(.*)$")
    if time_part and rest then
      time_part = trim(time_part)
      rest = trim(rest)
      
      -- Extract options like (blend:5, w:1.2)
      local options_str = time_part:match("%((.-)%)")
      local absolute_time_str = time_part:gsub("%(.-%)", "")
      absolute_time_str = trim(absolute_time_str)
      
      -- Parse time
      local frame_val = 0
      if absolute_time_str:match("%%$") then
        local pct = tonumber(absolute_time_str:sub(1, -2)) or 0
        frame_val = math.floor((pct / 100.0) * total_frames)
      elseif absolute_time_str:match("s$") then
        local sec = tonumber(absolute_time_str:sub(1, -2)) or 0
        frame_val = math.floor(sec * fps)
      else
        frame_val = tonumber(absolute_time_str) or 0
      end
      
      -- Parse prompt and negative
      local prompt_pt = rest
      local neg_prompt = ""
      local neg_idx = rest:find("%[%-%]")
      if neg_idx then
        prompt_pt = trim(rest:sub(1, neg_idx - 1))
        neg_prompt = trim(rest:sub(neg_idx + 3))
      end
      
      -- Parse options
      local transition = "hard_cut"
      local transition_frames = 0
      local weight = 1.0
      
      if options_str then
        if options_str:match("blend") then
            transition = "blend"
            local tf = options_str:match("blend:(%d+)")
            if tf then transition_frames = tonumber(tf) end
        end
        if options_str:match("hard_cut") then
            transition = "hard_cut"
        end
        local w = options_str:match("w:([%d%.]+)")
        if w then weight = tonumber(w) end
      end
      
      table.insert(schedule.keyframes, {
        frame = frame_val,
        prompt = prompt_pt,
        negative_prompt = neg_prompt,
        weight = weight,
        transition = transition,
        transition_frames = transition_frames
      })
    end
  end
  
  return schedule
end

return M
