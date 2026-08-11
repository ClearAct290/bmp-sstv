import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import pyaudio
import wave
import threading
import math
import scipy.signal as signal
from PIL import Image, ImageTk


SAMPLE_RATE = 3500
IMG_WIDTH = 320
IMG_HEIGHT = 240
DECODE_HEIGHT = 256
CHANNELS = 4  
BYTES_PER_LINE = IMG_WIDTH * CHANNELS
DISPLAY_SCALE = 2  
SYNC_TONE = np.array([0, 128, 255, 128] * 8, dtype=np.uint8)
SYNC_FLOAT = SYNC_TONE.astype(np.float32) - 128.0
SYNC_LEN = len(SYNC_TONE)
LINE_LEN = SYNC_LEN + BYTES_PER_LINE
HEADER_FREQ = 1200.0  
FOOTER_FREQ = 800.0   
CHUNK_SIZE = 8192     

class BGRXApp:
    def __init__(self, root):
        self.root = root
        self.root.title("bitmap sstv test rev 2")
        self.root.geometry("800x800") 
        
        self.p = pyaudio.PyAudio()
        self.is_decoding = False
        self.decode_thread = None
        
        self.display_buffer = np.full((DECODE_HEIGHT, IMG_WIDTH, CHANNELS), 128, dtype=np.uint8)
        self.audio_byte_buffer = bytearray()
        
        self.setup_gui()

    def setup_gui(self):
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        
        encode_frame = ttk.Frame(notebook)
        notebook.add(encode_frame, text="Encode")
        
        self.btn_load = ttk.Button(encode_frame, text="Load", command=self.load_image)
        self.btn_load.pack(pady=10)
        
        self.lbl_img_preview = tk.Label(encode_frame, text="No Image")
        self.lbl_img_preview.pack(pady=10)
        
        self.btn_export = ttk.Button(encode_frame, text="Export Wave", command=self.export_wav, state=tk.DISABLED)
        self.btn_export.pack(pady=5)

        self.btn_play = ttk.Button(encode_frame, text="Play", command=self.play_audio, state=tk.DISABLED)
        self.btn_play.pack(pady=5)
        
        
        decode_frame = ttk.Frame(notebook)
        notebook.add(decode_frame, text="Decode")
        
        mic_frame = ttk.Frame(decode_frame)
        mic_frame.pack(pady=5)
        ttk.Label(mic_frame, text="Input:").pack(side=tk.LEFT, padx=5)
        
        self.mic_devices = self.get_input_devices()
        self.mic_var = tk.StringVar()
        if self.mic_devices:
            self.mic_var.set(self.mic_devices[0][1])
            
        self.mic_dropdown = ttk.Combobox(mic_frame, textvariable=self.mic_var, values=[d[1] for d in self.mic_devices], width=40, state="readonly")
        self.mic_dropdown.pack(side=tk.LEFT, padx=5)
        
        ctrl_frame = ttk.Frame(decode_frame)
        ctrl_frame.pack(pady=5)
        
        self.btn_start_decode = ttk.Button(ctrl_frame, text="Decode", command=self.start_decoding)
        self.btn_start_decode.pack(side=tk.LEFT, padx=5)
        
        self.btn_stop_decode = ttk.Button(ctrl_frame, text="Stop Decode", command=self.stop_decoding, state=tk.DISABLED)
        self.btn_stop_decode.pack(side=tk.LEFT, padx=5)
        
        self.canvas = tk.Canvas(decode_frame, width=IMG_WIDTH * DISPLAY_SCALE, height=DECODE_HEIGHT * DISPLAY_SCALE, bg="black")
        self.canvas.pack(pady=5)
        self.decode_photo = None
        
        
        self.btn_save_bmp = ttk.Button(decode_frame, text="Save .BMP", command=self.save_bmp, state=tk.DISABLED)
        self.btn_save_bmp.pack(pady=5)

    def get_input_devices(self):
        devices = []
        for i in range(self.p.get_device_count()):
            try:
                info = self.p.get_device_info_by_index(i)
                if info["maxInputChannels"] > 0:
                    devices.append((i, info["name"]))
            except Exception:
                pass
        return devices

    
    
    def generate_tone(self, freq, duration_sec):
        t = np.linspace(0, duration_sec, int(SAMPLE_RATE * duration_sec), endpoint=False)
        wave_data = 0.5 * np.sin(2 * np.pi * freq * t)
        return (wave_data * 127.5 + 128).astype(np.uint8)

    def load_image(self):
        filepath = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")])
        if not filepath:
            return
            
        img = Image.open(filepath).convert("RGBA").resize((IMG_WIDTH, IMG_HEIGHT))
        self.preview_img = ImageTk.PhotoImage(img)
        self.lbl_img_preview.config(image=self.preview_img, text="")
        
        rgba_data = np.array(img)
        payload_bytes = bytearray()
        
        for row in reversed(range(IMG_HEIGHT)):
            row_pixels = rgba_data[row]
            bgrx_row = np.zeros_like(row_pixels)
            bgrx_row[:, 0] = row_pixels[:, 2] # Blue
            bgrx_row[:, 1] = row_pixels[:, 1] # Green
            bgrx_row[:, 2] = row_pixels[:, 0] # Red
            bgrx_row[:, 3] = 128              # X
            
            payload_bytes.extend(SYNC_TONE.tobytes())
            payload_bytes.extend(bgrx_row.flatten().tobytes())
            
        self.encoded_bytes = np.frombuffer(payload_bytes, dtype=np.uint8)
        self.btn_export.config(state=tk.NORMAL)
        self.btn_play.config(state=tk.NORMAL)

    def build_audio_payload(self):
        header_tone = self.generate_tone(HEADER_FREQ, 1.5)
        footer_tone = self.generate_tone(FOOTER_FREQ, 1.0)
        return np.concatenate((header_tone, self.encoded_bytes, footer_tone))

    def export_wav(self):
        filepath = filedialog.asksaveasfilename(defaultextension=".wav", filetypes=[("WAV Files", "*.wav")])
        if not filepath:
            return
            
        payload = self.build_audio_payload()
        with wave.open(filepath, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(1) 
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(payload.tobytes())
            
        messagebox.showinfo("Success", f"Saved to {filepath}")

    def play_audio(self):
        payload = self.build_audio_payload()
        def play():
            stream = self.p.open(format=pyaudio.paUInt8, channels=1, rate=SAMPLE_RATE, output=True)
            stream.write(payload.tobytes())
            stream.stop_stream()
            stream.close()
        threading.Thread(target=play, daemon=True).start()

    
    
    def start_decoding(self):
        selected_name = self.mic_var.get()
        device_index = next(d[0] for d in self.mic_devices if d[1] == selected_name)
        
        self.is_decoding = True
        self.btn_start_decode.config(state=tk.DISABLED)
        self.btn_stop_decode.config(state=tk.NORMAL)
        self.btn_save_bmp.config(state=tk.NORMAL)
        
        self.audio_byte_buffer = bytearray()
        self.decode_thread = threading.Thread(target=self.decode_loop, args=(device_index,), daemon=True)
        self.decode_thread.start()

    def stop_decoding(self):
        self.is_decoding = False
        self.btn_start_decode.config(state=tk.NORMAL)
        self.btn_stop_decode.config(state=tk.DISABLED)

    def decode_loop(self, device_index):
        hw_rate = 44100 
        
        gcd = math.gcd(SAMPLE_RATE, hw_rate)
        up_rate = SAMPLE_RATE // gcd   
        down_rate = hw_rate // gcd     
        
        overlap_samples = 63 * 32  
        trim_len = int(overlap_samples * up_rate / down_rate) 
        process_chunk_size = 63 * 128 
        
        try:
            stream = self.p.open(format=pyaudio.paFloat32,
                                 channels=1,
                                 rate=hw_rate,
                                 input=True,
                                 input_device_index=device_index,
                                 frames_per_buffer=CHUNK_SIZE)
        except Exception as e:
            messagebox.showerror("Audio Error", f"Could not open microphone: {e}")
            self.stop_decoding()
            return

        raw_audio_buffer = np.array([], dtype=np.float32)

        while self.is_decoding:
            try:
                data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                new_chunk = np.frombuffer(data, dtype=np.float32)
                raw_audio_buffer = np.concatenate((raw_audio_buffer, new_chunk))
                
                while len(raw_audio_buffer) >= process_chunk_size + overlap_samples:
                    to_process = raw_audio_buffer[:process_chunk_size + overlap_samples]
                    raw_audio_buffer = raw_audio_buffer[process_chunk_size:]
                    
                    resampled = signal.resample_poly(to_process, up_rate, down_rate)
                    valid_resampled = resampled[trim_len:]
                    
                    resampled_uint8 = np.clip((valid_resampled * 127.5) + 128, 0, 255).astype(np.uint8)
                    self.process_incoming_bytes(resampled_uint8)
                
            except Exception as e:
                print(f"Stream error: {e}")
                break
                
        stream.stop_stream()
        stream.close()

    def process_incoming_bytes(self, byte_array):
        self.audio_byte_buffer.extend(byte_array)
        
        while len(self.audio_byte_buffer) >= LINE_LEN * 2:
            search_chunk = np.array(self.audio_byte_buffer[:LINE_LEN + SYNC_LEN], dtype=np.uint8)
            search_float = search_chunk.astype(np.float32) - 128.0

            corr = np.correlate(search_float, SYNC_FLOAT, mode='valid')
            best_idx = np.argmax(corr)
            max_corr = corr[best_idx]

            if max_corr > 150000: 
                start_idx = best_idx + SYNC_LEN
                
                if start_idx + BYTES_PER_LINE <= len(self.audio_byte_buffer):
                    line_bytes = np.array(self.audio_byte_buffer[start_idx : start_idx + BYTES_PER_LINE], dtype=np.uint8)
                    del self.audio_byte_buffer[:start_idx + BYTES_PER_LINE]
                    self.render_line(line_bytes)
                else:
                    break 
            else:
                del self.audio_byte_buffer[:SYNC_LEN]

    def render_line(self, line_bytes):
        line_pixels = line_bytes.reshape((IMG_WIDTH, CHANNELS))
        
        
        r = line_pixels[:, 2].astype(np.float32) * 0.90  
        g = line_pixels[:, 1].astype(np.float32) * 1.05  
        b = line_pixels[:, 0].astype(np.float32) * 1.10  
        
        rgba_pixels = np.zeros_like(line_pixels)
        rgba_pixels[:, 0] = np.clip(r, 0, 255).astype(np.uint8) 
        rgba_pixels[:, 1] = np.clip(g, 0, 255).astype(np.uint8) 
        rgba_pixels[:, 2] = np.clip(b, 0, 255).astype(np.uint8) 
        rgba_pixels[:, 3] = 255                                 
        
        self.display_buffer = np.roll(self.display_buffer, 1, axis=0)
        self.display_buffer[0] = rgba_pixels
        
        self.update_canvas()

    def update_canvas(self):
        img = Image.fromarray(self.display_buffer, mode='RGBA')
        scaled_img = img.resize((IMG_WIDTH * DISPLAY_SCALE, DECODE_HEIGHT * DISPLAY_SCALE), Image.Resampling.NEAREST)
        self.decode_photo = ImageTk.PhotoImage(image=scaled_img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.decode_photo)

    def save_bmp(self):
        filepath = filedialog.asksaveasfilename(defaultextension=".bmp", filetypes=[("Bitmap Image", "*.bmp")])
        if not filepath:
            return
        
        
        img = Image.fromarray(self.display_buffer, mode='RGBA').convert("RGB")
        img.save(filepath, "BMP")
        messagebox.showinfo("Success", f"Image successfully saved to {filepath}")

if __name__ == "__main__":
    root = tk.Tk()
    app = BGRXApp(root)
    root.protocol("WM_DELETE_WINDOW", root.destroy)
    root.mainloop()