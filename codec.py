from dataclasses import dataclass
import math
import wave

import numpy as np

try:
    import scipy.signal as _signal
    _HAVE_SCIPY = True
except ImportError:  
    _HAVE_SCIPY = False


@dataclass
class Config:
    

    img_width: int = 320
    img_height: int = 240
    channels: int = 4          
    sample_rate: int = 3500
    header_freq: float = 1200.0
    header_duration: float = 1.5
    footer_freq: float = 800.0
    footer_duration: float = 1.0
    sync_base: tuple = (0, 128, 255, 128)
    sync_repeats: int = 8
    sync_corr_threshold: float = 150000.0
    tint_r: float = 0.90
    tint_g: float = 1.05
    tint_b: float = 1.10
    x_fill: int = 128

    @property
    def bytes_per_line(self) -> int:
        return self.img_width * self.channels

    @property
    def sync_tone(self) -> np.ndarray:
        return np.array(list(self.sync_base) * self.sync_repeats, dtype=np.uint8)

    @property
    def sync_len(self) -> int:
        return len(self.sync_base) * self.sync_repeats

    @property
    def line_len(self) -> int:
        return self.sync_len + self.bytes_per_line




def generate_tone(freq: float, duration_sec: float, cfg: Config) -> np.ndarray:
    t = np.linspace(0, duration_sec, int(cfg.sample_rate * duration_sec), endpoint=False)
    wave_data = 0.5 * np.sin(2 * np.pi * freq * t)
    return (wave_data * 127.5 + 128).astype(np.uint8)

def encode_image_to_payload(img, cfg: Config) -> np.ndarray:
    img = img.convert("RGBA").resize((cfg.img_width, cfg.img_height))
    rgba_data = np.array(img)

    sync_tone = cfg.sync_tone
    payload_bytes = bytearray()

    for row in reversed(range(cfg.img_height)):
        row_pixels = rgba_data[row]
        bgrx_row = np.zeros_like(row_pixels)
        bgrx_row[:, 0] = row_pixels[:, 2]   
        bgrx_row[:, 1] = row_pixels[:, 1]   
        bgrx_row[:, 2] = row_pixels[:, 0]   
        bgrx_row[:, 3] = cfg.x_fill         

        payload_bytes.extend(sync_tone.tobytes())
        payload_bytes.extend(bgrx_row.flatten().tobytes())

    return np.frombuffer(bytes(payload_bytes), dtype=np.uint8)


def wrap_payload_with_tones(payload: np.ndarray, cfg: Config) -> np.ndarray:
    header_tone = generate_tone(cfg.header_freq, cfg.header_duration, cfg)
    footer_tone = generate_tone(cfg.footer_freq, cfg.footer_duration, cfg)
    return np.concatenate((header_tone, payload, footer_tone))


def encode_image_to_audio(img, cfg: Config) -> np.ndarray:
    payload = encode_image_to_payload(img, cfg)
    return wrap_payload_with_tones(payload, cfg)


def save_wav(filepath: str, audio_u8: np.ndarray, cfg: Config) -> None:
    with wave.open(filepath, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(1)
        wf.setframerate(cfg.sample_rate)
        wf.writeframes(audio_u8.tobytes())




def apply_tint_correction(line_pixels_bgrx: np.ndarray, cfg: Config) -> np.ndarray:
    r = line_pixels_bgrx[:, 2].astype(np.float32) * cfg.tint_r
    g = line_pixels_bgrx[:, 1].astype(np.float32) * cfg.tint_g
    b = line_pixels_bgrx[:, 0].astype(np.float32) * cfg.tint_b

    rgb = np.zeros((cfg.img_width, 3), dtype=np.uint8)
    rgb[:, 0] = np.clip(r, 0, 255).astype(np.uint8)
    rgb[:, 1] = np.clip(g, 0, 255).astype(np.uint8)
    rgb[:, 2] = np.clip(b, 0, 255).astype(np.uint8)
    return rgb


class LineDecoder:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._buffer = bytearray()
        self._sync_float = cfg.sync_tone.astype(np.float32) - 128.0

    def _try_decode_one(self, min_buffer_len: int):
        cfg = self.cfg
        if len(self._buffer) < min_buffer_len:
            return None

        search_len = min(cfg.line_len + cfg.sync_len, len(self._buffer))
        search_chunk = np.frombuffer(bytes(self._buffer[:search_len]), dtype=np.uint8)
        search_float = search_chunk.astype(np.float32) - 128.0

        corr = np.correlate(search_float, self._sync_float, mode="valid")
        if len(corr) == 0:
            return None
        best_idx = int(np.argmax(corr))
        max_corr = corr[best_idx]

        if max_corr > cfg.sync_corr_threshold:
            start_idx = best_idx + cfg.sync_len

            if start_idx + cfg.bytes_per_line <= len(self._buffer):
                line_bytes = np.frombuffer(
                    bytes(self._buffer[start_idx:start_idx + cfg.bytes_per_line]),
                    dtype=np.uint8,
                )
                del self._buffer[: start_idx + cfg.bytes_per_line]

                line_pixels = line_bytes.reshape((cfg.img_width, cfg.channels))
                return apply_tint_correction(line_pixels, cfg)
            else:
                return None  
        else:
            del self._buffer[: cfg.sync_len]
            return None

    def feed(self, byte_array: np.ndarray):
        self._buffer.extend(np.asarray(byte_array, dtype=np.uint8).tobytes())

        cfg = self.cfg
        rows = []
        while len(self._buffer) >= cfg.line_len * 2:
            row = self._try_decode_one(cfg.line_len * 2)
            if row is not None:
                rows.append(row)
            elif len(self._buffer) >= cfg.line_len * 2:
                continue
        return rows

    def flush(self):
        cfg = self.cfg
        rows = []
        while len(self._buffer) >= cfg.sync_len:
            row = self._try_decode_one(cfg.sync_len)
            if row is not None:
                rows.append(row)
            else:
                if len(self._buffer) < cfg.sync_len:
                    break
                if len(self._buffer) < cfg.line_len + cfg.sync_len:
                    break
        return rows


def decode_all(payload: np.ndarray, cfg: Config):
    decoder = LineDecoder(cfg)
    rows = decoder.feed(payload)
    rows.extend(decoder.flush())
    return rows


def rows_to_image(rows, cfg: Config):
    from PIL import Image
    if not rows:
        raise ValueError("no rows to assemble into an image")

    stacked = np.stack(rows[::-1], axis=0)  
    return Image.fromarray(stacked, mode="RGB")


def resample_hw_chunk(chunk_f32: np.ndarray, hw_rate: int, cfg: Config,
                       trim_len: int) -> np.ndarray:
    if not _HAVE_SCIPY:
        raise RuntimeError("resample_hw_chunk requires scipy (pip install scipy)")

    gcd = math.gcd(cfg.sample_rate, hw_rate)
    up_rate = cfg.sample_rate // gcd
    down_rate = hw_rate // gcd

    resampled = _signal.resample_poly(chunk_f32, up_rate, down_rate)
    valid = resampled[trim_len:]
    return np.clip((valid * 127.5) + 128, 0, 255).astype(np.uint8)
