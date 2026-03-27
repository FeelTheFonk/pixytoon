-- extension/scripts/test_dsl_parser.lua
local parser = require("sddj_dsl_parser")

local input = [[
# This is a comment
[0]
base

[50%] blend: 5
evolving
-- bad
w: 1.2

@5s
climax
]]

local fps = 24
local total_frames = 240

local schedule = parser.parse(input, total_frames, fps)

assert(schedule ~= nil, "Schedule parsing failed")
assert(#schedule.keyframes == 3, "Expected 3 keyframes, got " .. #schedule.keyframes)

-- Keyframe 1 (0) -> frame 0
assert(schedule.keyframes[1].frame == 0)
assert(schedule.keyframes[1].prompt == "base", "Expected 'base', got " .. tostring(schedule.keyframes[1].prompt))
assert(schedule.keyframes[1].transition == "hard_cut")

-- Keyframe 2 (50%) -> frame 120
assert(schedule.keyframes[2].frame == 120)
assert(schedule.keyframes[2].prompt == "evolving")
assert(schedule.keyframes[2].negative_prompt == "bad")
assert(schedule.keyframes[2].transition == "blend")
assert(schedule.keyframes[2].transition_frames == 5)
assert(schedule.keyframes[2].weight == 1.2)

-- Keyframe 3 (5s) -> frame 120
assert(schedule.keyframes[3].frame == 120)
assert(schedule.keyframes[3].prompt == "climax")
assert(schedule.keyframes[3].transition == "hard_cut")

-- Test empty safe fallback
local empty_schedule = parser.parse("   \n  \t  ", total_frames, fps)
assert(#empty_schedule.keyframes == 0, "Empty parse should return 0 keyframes")

-- Test implicit frame 0 fallback
local implicit = parser.parse("beautiful scenery\n-- blurry\nblend: 10", total_frames, fps)
assert(#implicit.keyframes == 1, "Expected 1 keyframe for implicit text")
assert(implicit.keyframes[1].frame == 0)
assert(implicit.keyframes[1].prompt == "beautiful scenery")
assert(implicit.keyframes[1].negative_prompt == "blurry")
assert(implicit.keyframes[1].transition == "blend")
assert(implicit.keyframes[1].transition_frames == 10)

print("PASS")
