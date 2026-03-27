-- extension/scripts/test_dsl_parser.lua
local parser = require("sddj_dsl_parser")

local input = "0%: base | 50%(blend:5, w:1.2): evolving [-] bad | 5s: climax (hard_cut)"
local fps = 24
local total_frames = 240

local schedule = parser.parse(input, total_frames, fps)
assert(schedule ~= nil, "Schedule parsing failed")
assert(#schedule.keyframes == 3, "Expected 3 keyframes")

-- Keyframe 1 (0%)
assert(schedule.keyframes[1].frame == 0)
assert(schedule.keyframes[1].prompt == "base")
assert(schedule.keyframes[1].transition == "hard_cut")

-- Keyframe 2 (50%) -> frame 120
assert(schedule.keyframes[2].frame == 120)
assert(schedule.keyframes[2].prompt == "evolving")
assert(schedule.keyframes[2].negative_prompt == "bad")
assert(schedule.keyframes[2].transition == "blend")
assert(schedule.keyframes[2].transition_frames == 5)
assert(schedule.keyframes[2].weight == 1.2)

-- Keyframe 3 (5s) -> frame 120 (5 * 24 = 120)
assert(schedule.keyframes[3].frame == 120)
assert(schedule.keyframes[3].prompt == "climax")
assert(schedule.keyframes[3].transition == "hard_cut")

print("PASS")
