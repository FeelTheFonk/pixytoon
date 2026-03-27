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
  
  -- SOTA robust empty check
  if not input_str or input_str:match("^%s*$") then
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

  if not input_str or input_str:match("^%s*$") then
    return schedule
  end

  local blocks = {}
  local current_block = nil
  
  -- Parse multiple lines into block segments
  for line in input_str:gmatch("[^\r\n]+") do
    local l = trim(line)
    -- Ignore comments
    if not (l:match("^#") or l:match("^//")) then
      -- Match time markers: [0], [50%], [5s] or @0, @50%, @5s
      local time_str, rest = l:match("^%[(.-)%]%s*(.*)")
      if not time_str then
          time_str, rest = l:match("^@([^%s]+)%s*(.*)")
      end
      
      if time_str then
         current_block = {
             time_str = trim(time_str),
             lines = {}
         }
         if rest and trim(rest) ~= "" then
             table.insert(current_block.lines, trim(rest))
         end
         table.insert(blocks, current_block)
      else
         if not current_block then
             current_block = {
                 time_str = "0",
                 lines = {}
             }
             table.insert(blocks, current_block)
         end
         if l ~= "" then
             table.insert(current_block.lines, l)
         end
      end
    end
  end
  
  for _, block in ipairs(blocks) do
    local absolute_time_str = block.time_str
    
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
    
    local prompt_lines = {}
    local negative_prompt = ""
    local in_negative = false
    local weight = 1.0
    local transition = "hard_cut"
    local transition_frames = 0
    
    for _, l in ipairs(block.lines) do
      if l:match("^%-%-") then
          in_negative = true
          negative_prompt = trim(l:sub(3))
      elseif l:match("^blend:%s*(%d+)") then
          transition = "blend"
          transition_frames = tonumber(l:match("^blend:%s*(%d+)"))
      elseif l:match("^transition:%s*(%w+)") then
          transition = trim(l:match("^transition:%s*(%w+)"))
      elseif l:match("^weight:%s*([%d%.]+)") or l:match("^w:%s*([%d%.]+)") then
          weight = tonumber(l:match("^w.-:%s*([%d%.]+)")) or weight
      else
          if l ~= "" then
              if in_negative then
                  negative_prompt = negative_prompt .. " " .. l
              else
                  table.insert(prompt_lines, l)
              end
          end
      end
    end
    
    local prompt_pt = trim(table.concat(prompt_lines, " "))
    negative_prompt = trim(negative_prompt)
    
    table.insert(schedule.keyframes, {
      frame = frame_val,
      prompt = prompt_pt,
      negative_prompt = negative_prompt,
      weight = weight,
      transition = transition,
      transition_frames = transition_frames
    })
  end
  
  -- Sort ascending by frame
  table.sort(schedule.keyframes, function(a, b) return a.frame < b.frame end)
  
  return schedule
end

return M
