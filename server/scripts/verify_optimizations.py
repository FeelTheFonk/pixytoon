import asyncio
import hashlib
import time
import sys
import logging
from PIL import Image
import torch
from pathlib import Path

# Setup SDDj context
sys.path.insert(0, str(Path(__file__).parent.parent))

from sddj.engine.core import DiffusionEngine
from sddj.protocol import AudioReactiveRequest, GenerationMode
from sddj.config import settings

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("sddj.verify")

def hash_image(img: Image.Image) -> str:
    """Hash the raw RGBA bytes of the PIL image."""
    img = img.convert("RGBA")
    return hashlib.sha256(img.tobytes()).hexdigest()

async def run_verification():
    log.info("SDDj SOTA 2026 Optimization Verification Harness")
    log.info("Initializing DiffusionEngine (weights loading)...")
    t0 = time.perf_counter()
    engine = DiffusionEngine()
    engine.load_models()
    log.info(f"Engine loaded in {time.perf_counter() - t0:.2f}s")
    
    req = AudioReactiveRequest(
        mode=GenerationMode.IMG2IMG,
        steps=8,
        denoise_strength=0.6,
        cfg_scale=5.0,
        width=512,
        height=512,
        prompt="A beautiful futuristic cyberpunk city masterwork, uncompressed, SOTA quality",
        negative_prompt="blurry, low quality, artifact",
        seed=1337,
        frame_count=20,
        audio_fps=10.0,
        # Force temporal coherence and flow to test those paths
        prompt_schedule={} 
    )
    
    log.info("Warming up engine (compilation)...")
    engine._warmup()

    frame_hashes = []
    
    def on_progress(p):
        pass

    def on_frame(f):
        img_raw = f._raw_bytes
        frame_hash = hashlib.sha256(img_raw).hexdigest()
        frame_hashes.append(frame_hash)
        log.info(f"Frame {f.frame_index}/{f.total_frames} | {f.time_ms}ms | Hash: {frame_hash[:8]}")

    log.info("Starting Audio-Reactive Chain test (20 frames)...")
    t1 = time.perf_counter()
    engine.generate_audio_reactive(req, on_progress=on_progress, on_frame=on_frame)
    chain_time = time.perf_counter() - t1
    
    log.info(f"Chain test completed in {chain_time:.2f}s ({req.frame_count/chain_time:.2f} fps avg)")
    log.info(f"Final Image Hashes: {[h[:8] for h in frame_hashes]}")

if __name__ == "__main__":
    if not torch.cuda.is_available():
        log.error("CUDA not available. Cannot perform determinism verification.")
        sys.exit(1)
        
    asyncio.run(run_verification())
