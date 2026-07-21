"""
ANIQUEN wake-word server
-------------------------
Runs the full mel-spectrogram + embedding + wake-word pipeline on the
SERVER instead of the client device. The phone just streams raw audio
over a WebSocket and receives back a wake probability / detection flag.

Setup on your always-on VM (e.g. Oracle Cloud Free Tier):
    pip install fastapi uvicorn onnxruntime numpy websockets

Put these three files in the SAME folder as this script:
    melspectrogram.onnx
    embedding_model.onnx   (quantized version is fine and faster)
    hey_Aniquen.onnx       (quantized version is fine and faster)

Run it:
    uvicorn server:app --host 0.0.0.0 --port 8000

Then from any device, connect a WebSocket to:
    ws://<your-vm-public-ip>:8000/ws
(use wss:// with a reverse proxy + TLS cert if you want it secure —
 recommended before exposing this on the public internet long-term)

Protocol (very simple):
  - Client sends BINARY WebSocket messages containing raw Float32 PCM
    samples at 16000 Hz, already scaled to int16 range (same convention
    as the original browser code: sample * 32768).
  - Server replies with a JSON TEXT message after each chunk it
    processes:
        {"prob": 0.734, "detected": true, "rms": 812.3}
  - Client just watches for "detected": true to flip the UI green.
"""

import asyncio
import json
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

# ----- configuration (mirrors the browser version) ----------------------
SAMPLE_RATE = 16000
FRAMES_PER_WINDOW = 76
MEL_BINS = 32
STRIDE = 8
CONTEXT_EMBEDDINGS = 16

MAX_BUFFER_SAMPLES = 48000
MIN_BUFFER_SAMPLES = 32000

WAKE_THRESHOLD = 0.3
CYCLE_HISTORY_LEN = 4
EMBEDDING_HISTORY_MAX = 40

TARGET_RMS = 600.0
MIN_RMS_TO_NORMALIZE = 30.0
MAX_GAIN = 8.0

VAD_MULTIPLIER = 2.0
ACTIVE_HOLD_CYCLES = 6
NOISE_FLOOR_ALPHA = 0.05

MEL_MODEL_PATH = "./melspectrogram.onnx"
EMBED_MODEL_PATH = "./embedding_model.onnx"
MULTI_MODEL_PATH = "./hey_Aniquen.onnx"

# ----- load models once, shared across all connections -------------------
mel_session = ort.InferenceSession(MEL_MODEL_PATH, providers=["CPUExecutionProvider"])
embed_session = ort.InferenceSession(EMBED_MODEL_PATH, providers=["CPUExecutionProvider"])
multi_session = ort.InferenceSession(MULTI_MODEL_PATH, providers=["CPUExecutionProvider"])


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def compute_mel_spectrogram(samples: np.ndarray) -> np.ndarray:
    # samples: shape (1, N) float32
    result = mel_session.run(None, {"input": samples})[0]  # (1,1,frames,32)
    mel2d = (result.reshape(-1, MEL_BINS) / 10.0 + 2.0).astype(np.float32)
    return mel2d


def compute_embeddings(mel2d: np.ndarray) -> list:
    total_frames = mel2d.shape[0]
    embeddings = []
    if total_frames < FRAMES_PER_WINDOW:
        return embeddings
    for start in range(0, total_frames - FRAMES_PER_WINDOW + 1, STRIDE):
        window = mel2d[start:start + FRAMES_PER_WINDOW].reshape(1, 76, 32, 1).astype(np.float32)
        out = embed_session.run(None, {"input_1": window})[0].reshape(96)
        embeddings.append(out)
    return embeddings


def compute_wake_probability(embedding_stack: np.ndarray) -> float:
    # embedding_stack: (16, 96)
    flat = embedding_stack.reshape(1, 16, 96).astype(np.float32)
    result = multi_session.run(None, {"onnx::Flatten_0": flat})[0]
    score = float(result[0][0])
    return score if 0 <= score <= 1 else float(sigmoid(score))


def normalize_gain(samples: np.ndarray):
    rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2))) if len(samples) else 0.0
    if rms < MIN_RMS_TO_NORMALIZE:
        return samples.astype(np.float32), rms, 1.0
    gain = TARGET_RMS / rms
    gain = max(0.1, min(MAX_GAIN, gain))
    return (samples * gain).astype(np.float32), rms, gain


class SessionState:
    """Per-connection rolling state, since each phone gets its own buffer/history."""
    def __init__(self):
        self.ring = np.zeros(MAX_BUFFER_SAMPLES, dtype=np.float32)
        self.write_pos = 0
        self.filled = 0
        self.embedding_history = []
        self.cycle_max_history = []
        self.noise_floor = 200.0
        self.active_cycles_remaining = 0
        self.is_wake_active = False

    def push(self, chunk: np.ndarray):
        n = len(chunk)
        end = self.write_pos + n
        if end <= MAX_BUFFER_SAMPLES:
            self.ring[self.write_pos:end] = chunk
        else:
            first = MAX_BUFFER_SAMPLES - self.write_pos
            self.ring[self.write_pos:] = chunk[:first]
            self.ring[:end - MAX_BUFFER_SAMPLES] = chunk[first:]
        self.write_pos = end % MAX_BUFFER_SAMPLES
        self.filled = min(MAX_BUFFER_SAMPLES, self.filled + n)

    def get_latest(self, count: int) -> np.ndarray:
        count = min(count, self.filled)
        read_pos = (self.write_pos - count) % MAX_BUFFER_SAMPLES
        if read_pos + count <= MAX_BUFFER_SAMPLES:
            return self.ring[read_pos:read_pos + count].copy()
        first = MAX_BUFFER_SAMPLES - read_pos
        out = np.empty(count, dtype=np.float32)
        out[:first] = self.ring[read_pos:]
        out[first:] = self.ring[:count - first]
        return out


def process_chunk(state: SessionState) -> dict:
    """Runs one detection cycle for this session's current buffer. Returns a result dict or None if skipped."""
    if state.filled < MIN_BUFFER_SAMPLES:
        return None

    recent = state.get_latest(4096)
    recent_rms = float(np.sqrt(np.mean(recent.astype(np.float64) ** 2))) if len(recent) else 0.0

    is_active = recent_rms > state.noise_floor * VAD_MULTIPLIER
    if is_active:
        state.active_cycles_remaining = ACTIVE_HOLD_CYCLES
    else:
        state.noise_floor = state.noise_floor * (1 - NOISE_FLOOR_ALPHA) + recent_rms * NOISE_FLOOR_ALPHA

    if state.active_cycles_remaining <= 0:
        return {"idle": True, "rms": round(recent_rms, 1), "floor": round(state.noise_floor, 1)}
    state.active_cycles_remaining -= 1

    raw_chunk = state.get_latest(MAX_BUFFER_SAMPLES)
    chunk, rms, gain = normalize_gain(raw_chunk)

    mel2d = compute_mel_spectrogram(chunk.reshape(1, -1))
    if mel2d.shape[0] < FRAMES_PER_WINDOW:
        return None

    new_embeddings = compute_embeddings(mel2d)
    if not new_embeddings:
        return None

    state.embedding_history.extend(new_embeddings)
    if len(state.embedding_history) > EMBEDDING_HISTORY_MAX:
        state.embedding_history = state.embedding_history[-EMBEDDING_HISTORY_MAX:]
    if len(state.embedding_history) < CONTEXT_EMBEDDINGS:
        return None

    cycle_max_prob = 0.0
    triggered = False
    last_start = len(state.embedding_history) - CONTEXT_EMBEDDINGS
    # Cap how many positions we check per cycle — checking all ~28 possible
    # positions every cycle is too expensive for Render's free-tier CPU.
    # We'll still re-check nearby positions on the next cycle anyway (VAD
    # keeps us "active" for several cycles), so this trades a bit of
    # exhaustiveness for actually being able to keep up in real time.
    MAX_POSITIONS_PER_CYCLE = 6
    check_from = max(0, last_start - min(len(new_embeddings), MAX_POSITIONS_PER_CYCLE))
    for start in range(check_from, last_start + 1):
        ctx = np.array(state.embedding_history[start:start + CONTEXT_EMBEDDINGS])
        prob = compute_wake_probability(ctx)
        if prob > cycle_max_prob:
            cycle_max_prob = prob
        if prob >= WAKE_THRESHOLD:
            triggered = True
            break

    state.cycle_max_history.append(cycle_max_prob)
    if len(state.cycle_max_history) > CYCLE_HISTORY_LEN:
        state.cycle_max_history = state.cycle_max_history[-CYCLE_HISTORY_LEN:]
    rolling_max = max(state.cycle_max_history)

    detected = triggered or (rolling_max >= WAKE_THRESHOLD)

    return {
        "rms": round(rms, 1),
        "gain": round(gain, 2),
        "prob": round(cycle_max_prob, 3),
        "rolling_max": round(rolling_max, 3),
        "detected": bool(detected),
    }


app = FastAPI()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    state = SessionState()
    try:
        while True:
            message = await websocket.receive_bytes()
            # message is raw float32 PCM bytes (little-endian), already scaled to int16 range
            chunk = np.frombuffer(message, dtype=np.float32)
            state.push(chunk)

            # Run the heavy (blocking) ML pipeline in a background thread instead
            # of directly on the event loop. Without this, a slow inference call
            # (very likely on Render's free-tier CPU) freezes the ENTIRE websocket
            # connection — no new audio can be received, no keepalive frames can
            # be answered — which is what was causing the periodic disconnects
            # and "bursty" log delivery.
            result = await asyncio.to_thread(process_chunk, state)
            if result is not None:
                await websocket.send_text(json.dumps(result))
    except WebSocketDisconnect:
        pass
