--
-- SDDj — Constants & Shared State
--

return function(PT)

-- ─── Constants ──────────────────────────────────────────────

PT.VERSION = "0.9.61"

PT.cfg = {
  DEFAULT_SERVER_URL      = "ws://127.0.0.1:9876/ws",
  SETTINGS_FILE           = app.fs.joinPath(app.fs.userConfigPath, "sddj_settings.json"),
  CONNECT_TIMEOUT         = 5.0,
  HEARTBEAT_INTERVAL      = 30.0,
  GEN_TIMEOUT             = 660,
  CANCEL_TIMEOUT          = 30,
  LOOP_DELAY              = 0.1,
  DIRTY_STEP_DIVISOR      = 32,
  RECONNECT_BASE_DELAY    = 2.0,
  RECONNECT_MAX_DELAY     = 30.0,
}

-- ─── Mutable State ──────────────────────────────────────────

math.randomseed(os.time())

PT.ws_handle = nil
PT.dlg       = nil

PT.state = {
  connected      = false,
  generating     = false,
  animating      = false,
  cancel_pending = false,
  gen_step_start = nil,
  file_counter   = 0,
  session_id     = tostring(os.time()) .. "_" .. tostring(math.random(1000, 9999)),
  last_pong      = nil,
  pending_action = nil,  -- "generate" | "animate" | "qr_generate" | "audio" | nil
}

PT.anim = {
  layer            = nil,
  start_frame      = 0,
  frame_count      = 0,
  base_seed        = 0,
  output_dir       = nil,   -- incremental output directory (set on first frame)
  output_count     = 0,     -- frames written to output_dir
  last_saved_frame = nil,   -- path to last written frame (for gap-fill fallback)
  last_frame_index = -1,    -- last received frame_index (gap detection)
  decode_failures  = 0,     -- frames that failed to decode (diagnostics)
}

PT.seq = {
  layer       = nil,
  start_frame = 0,
  frame_count = 0,
  active      = false,
}

PT.loop = {
  mode          = false,
  counter       = 0,
  seed_mode     = "random",
  random_mode   = false,
  locked_fields = {},
  target        = nil,  -- "generate" | "animate" | "audio" | nil
}

PT.res = {
  requested  = false,
  palettes   = {},
  loras      = {},
  embeddings = {},
  presets    = {},
}

PT.timers = {
  connect        = nil,
  heartbeat      = nil,
  gen_timeout    = nil,
  loop           = nil,
  cancel_safety  = nil,
  reconnect      = nil,
}

PT.reconnect = {
  attempts = 0,
  manual_disconnect = false,
}

PT.audio = {
  analyzing        = false,
  analyzed         = false,
  generating       = false,
  duration         = 0,
  total_frames     = 0,
  features         = {},
  stems_available  = false,
  stems            = {},
  mod_presets       = {},
  bpm              = 0,
  lufs             = -24,
  sample_rate      = 44100,
  hop_length       = 256,
  recommended_preset = "",
  waveform         = {},
  last_output_dir  = nil,
}

-- ─── Metadata Tracking ────────────────────────────────────────

PT.last_request     = nil   -- last generation request (deep copy, no images)
PT.last_result_meta = nil   -- metadata built from last_request + response

-- ─── Output Config ────────────────────────────────────────────

PT.output = {
  enabled = true,   -- save to output dir by default
}

end
