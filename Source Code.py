import tkinter as tk
import customtkinter as ctk
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import time

# Set UI Theme and Color
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class SignalGeneratorApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("Analog Communication Tool & Signal Generator")
        self.geometry("1100x700")
        self.resizable(True, True)
        
        # Audio state variables
        self.sample_rate = 44100
        self.is_playing = False
        self.stream = None
        
        # Thread-safe control variables (using simple floats/strings/bools)
        self.carrier_freq = 440.0
        self.message_freq = 5.0
        self.mod_type = "AM"
        self.mod_index = 0.5
        self.volume = 0.3
        
        # Phase accumulators for real-time audio generation
        self.phase_carrier = 0.0
        self.phase_message = 0.0
        
        # Build UI layout
        self.setup_ui()
        
        # Start initial plot
        self.update_plots()
        
    def setup_ui(self):
        # Configure Grid layout (1 row, 2 columns for sidebar and main plot)
        self.grid_columnconfigure(0, weight=1, minsize=350)
        self.grid_columnconfigure(1, weight=2)
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0) # Bottom bar
        
        # ==========================================
        # LEFT PANEL: Controls (Sidebar)
        # ==========================================
        self.sidebar = ctk.CTkFrame(self, corner_radius=15, fg_color="#1E1E24")
        self.sidebar.grid(row=0, column=0, padx=15, pady=15, sticky="nsew")
        
        # Title/Logo Area
        title_label = ctk.CTkLabel(self.sidebar, text="SIGNAL CONTROL", font=ctk.CTkFont(size=20, weight="bold"))
        title_label.pack(padx=20, pady=(20, 10))
        
        # Divider Line
        divider = ctk.CTkFrame(self.sidebar, height=2, fg_color="#2D2D34")
        divider.pack(fill="x", padx=20, pady=(0, 20))
        
        # Scrollable container for control controls
        self.scroll_container = ctk.CTkScrollableFrame(self.sidebar, fg_color="transparent")
        self.scroll_container.pack(fill="both", expand=True, padx=10, pady=5)
        
        # 1. Modulation Type
        mod_label = ctk.CTkLabel(self.scroll_container, text="Modulation Type", font=ctk.CTkFont(size=14, weight="bold"))
        mod_label.pack(anchor="w", padx=15, pady=(5, 5))
        
        self.mod_type_segmented = ctk.CTkSegmentedButton(
            self.scroll_container,
            values=["AM", "FM"],
            command=self.on_mod_type_change,
            font=ctk.CTkFont(weight="bold")
        )
        self.mod_type_segmented.set("AM")
        self.mod_type_segmented.pack(fill="x", padx=15, pady=(0, 15))
        
        # 2. Carrier Frequency
        self.fc_label = ctk.CTkLabel(self.scroll_container, text="Carrier Frequency (fc): 440 Hz", font=ctk.CTkFont(size=13))
        self.fc_label.pack(anchor="w", padx=15, pady=(5, 2))
        self.fc_slider = ctk.CTkSlider(self.scroll_container, from_=50, to=2000, number_of_steps=1950, command=self.on_fc_change)
        self.fc_slider.set(440)
        self.fc_slider.pack(fill="x", padx=15, pady=(0, 15))
        
        # 3. Message Frequency
        self.fm_label = ctk.CTkLabel(self.scroll_container, text="Message Frequency (fm): 5.0 Hz", font=ctk.CTkFont(size=13))
        self.fm_label.pack(anchor="w", padx=15, pady=(5, 2))
        self.fm_slider = ctk.CTkSlider(self.scroll_container, from_=0.5, to=100.0, number_of_steps=995, command=self.on_fm_change)
        self.fm_slider.set(5.0)
        self.fm_slider.pack(fill="x", padx=15, pady=(0, 15))
        
        # 4. Modulation Index / Deviation
        self.m_label = ctk.CTkLabel(self.scroll_container, text="Modulation Index (m): 50%", font=ctk.CTkFont(size=13))
        self.m_label.pack(anchor="w", padx=15, pady=(5, 2))
        self.m_slider = ctk.CTkSlider(self.scroll_container, from_=0.0, to=1.0, number_of_steps=100, command=self.on_m_change)
        self.m_slider.set(0.5)
        self.m_slider.pack(fill="x", padx=15, pady=(0, 15))
        
        # 5. Volume Control
        self.vol_label = ctk.CTkLabel(self.scroll_container, text="Volume: 30%", font=ctk.CTkFont(size=13))
        self.vol_label.pack(anchor="w", padx=15, pady=(5, 2))
        self.vol_slider = ctk.CTkSlider(self.scroll_container, from_=0.0, to=1.0, number_of_steps=100, command=self.on_volume_change)
        self.vol_slider.set(0.3)
        self.vol_slider.pack(fill="x", padx=15, pady=(0, 20))
        
        # ==========================================
        # RIGHT PANEL: Visualizer Display
        # ==========================================
        self.visualizer_frame = ctk.CTkFrame(self, corner_radius=15, fg_color="#18181C")
        self.visualizer_frame.grid(row=0, column=1, padx=15, pady=15, sticky="nsew")
        
        # Embed Matplotlib Figure
        self.fig, (self.ax_msg, self.ax_mod) = plt.subplots(2, 1, figsize=(6, 5))
        self.fig.patch.set_facecolor("#18181C")
        
        for ax in (self.ax_msg, self.ax_mod):
            ax.set_facecolor("#121214")
            ax.tick_params(colors="#8E8E93")
            ax.spines['bottom'].set_color('#2D2D34')
            ax.spines['top'].set_color('#2D2D34')
            ax.spines['left'].set_color('#2D2D34')
            ax.spines['right'].set_color('#2D2D34')
            ax.grid(True, color="#2D2D34", linestyle="--")
            
        self.ax_msg.set_title("Message / Modulation Signal m(t)", color="#E5E5EA", fontname="sans-serif", fontsize=11, fontweight="bold")
        self.ax_mod.set_title("Modulated Waveform s(t)", color="#E5E5EA", fontname="sans-serif", fontsize=11, fontweight="bold")
        
        self.fig.tight_layout(pad=3.0)
        
        # Canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.visualizer_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill="both", expand=True, padx=15, pady=15)
        
        # ==========================================
        # BOTTOM BAR: Playback Controls & Status
        # ==========================================
        self.bottom_bar = ctk.CTkFrame(self, height=80, corner_radius=15, fg_color="#1E1E24")
        self.bottom_bar.grid(row=1, column=0, columnspan=2, padx=15, pady=(0, 15), sticky="ew")
        
        # Grid inside bottom bar
        self.bottom_bar.grid_columnconfigure(0, weight=1)
        self.bottom_bar.grid_columnconfigure(1, weight=1)
        self.bottom_bar.grid_columnconfigure(2, weight=1)
        
        # Play/Stop Button
        self.play_btn = ctk.CTkButton(
            self.bottom_bar,
            text="Generate & Play",
            fg_color="#0A84FF",
            hover_color="#0056B3",
            font=ctk.CTkFont(size=15, weight="bold"),
            command=self.toggle_playback,
            height=45,
            corner_radius=8
        )
        self.play_btn.grid(row=0, column=0, padx=20, pady=15, sticky="w")
        
        # Real-time update checkbox
        self.live_checkbox = ctk.CTkCheckBox(
            self.bottom_bar,
            text="Live Waveform Updates",
            font=ctk.CTkFont(size=13),
            command=self.update_plots
        )
        self.live_checkbox.select()
        self.live_checkbox.grid(row=0, column=1, padx=20, pady=15)
        
        # Status Area (Indicator + Text)
        self.status_container = ctk.CTkFrame(self.bottom_bar, fg_color="transparent")
        self.status_container.grid(row=0, column=2, padx=20, pady=15, sticky="e")
        
        # Glowing LED status circle
        self.led_canvas = tk.Canvas(self.status_container, width=16, height=16, bg="#1E1E24", highlightthickness=0)
        self.led_canvas.pack(side="left", padx=(0, 8))
        self.led_circle = self.led_canvas.create_oval(2, 2, 14, 14, fill="#8E8E93", width=0)
        
        self.status_label = ctk.CTkLabel(self.status_container, text="System: Idle", font=ctk.CTkFont(size=14, weight="bold"))
        self.status_label.pack(side="left")

    # ==========================================
    # Callbacks and Value Updaters
    # ==========================================
    def on_fc_change(self, val):
        self.carrier_freq = float(val)
        self.fc_label.configure(text=f"Carrier Frequency (fc): {int(self.carrier_freq)} Hz")
        if self.live_checkbox.get():
            self.update_plots()
            
    def on_fm_change(self, val):
        self.message_freq = float(val)
        self.fm_label.configure(text=f"Message Frequency (fm): {self.message_freq:.1f} Hz")
        if self.live_checkbox.get():
            self.update_plots()
            
    def on_m_change(self, val):
        self.mod_index = float(val)
        if self.mod_type == "AM":
            self.m_label.configure(text=f"Modulation Index (m): {int(self.mod_index * 100)}%")
        else:
            # For FM, modulation index maps to frequency deviation (e.g. 0 to 500 Hz)
            deviation = self.mod_index * 500.0
            self.m_label.configure(text=f"Frequency Dev (Δf): {int(deviation)} Hz")
        if self.live_checkbox.get():
            self.update_plots()
            
    def on_volume_change(self, val):
        self.volume = float(val)
        self.vol_label.configure(text=f"Volume: {int(self.volume * 100)}%")

    def on_mod_type_change(self, val):
        self.mod_type = val
        # Trigger slider updates because meaning of slider changes
        self.on_m_change(self.m_slider.get())
        if self.live_checkbox.get():
            self.update_plots()

    # ==========================================
    # Matplotlib Visualization
    # ==========================================
    def update_plots(self):
        # Generate plot time steps (showing ~2 full periods of the message wave)
        duration = 2.0 / max(self.message_freq, 0.5)
        t = np.linspace(0, duration, 1000)
        
        # Calculate waves
        # Message signal
        y_msg = np.cos(2 * np.pi * self.message_freq * t)
        
        # Modulated signal
        if self.mod_type == "AM":
            y_mod = (1 + self.mod_index * y_msg) * np.cos(2 * np.pi * self.carrier_freq * t)
        else:
            # FM formulation: FM dev = mod_index * 500
            dev = self.mod_index * 500.0
            # Phase integral of cos(2pi fm t) is sin(2pi fm t) / (2pi fm)
            # FM equation: cos(2pi fc t + (dev / fm) * sin(2pi fm t))
            y_mod = np.cos(2 * np.pi * self.carrier_freq * t + (dev / self.message_freq) * np.sin(2 * np.pi * self.message_freq * t))
            
        # Draw on Message axis
        self.ax_msg.clear()
        self.ax_msg.set_facecolor("#121214")
        self.ax_msg.grid(True, color="#2D2D34", linestyle="--")
        self.ax_msg.plot(t * 1000, y_msg, color="#30D158", linewidth=2) # Neon Green
        self.ax_msg.set_title("Message / Modulation Signal m(t)", color="#E5E5EA", fontname="sans-serif", fontsize=11, fontweight="bold")
        self.ax_msg.set_xlabel("Time (ms)", color="#8E8E93", fontsize=9)
        self.ax_msg.set_ylabel("Amplitude", color="#8E8E93", fontsize=9)
        self.ax_msg.tick_params(colors="#8E8E93")
        
        # Draw on Modulated axis
        self.ax_mod.clear()
        self.ax_mod.set_facecolor("#121214")
        self.ax_mod.grid(True, color="#2D2D34", linestyle="--")
        self.ax_mod.plot(t * 1000, y_mod, color="#0A84FF", linewidth=1.5) # Neon Blue
        self.ax_mod.set_title(f"{self.mod_type} Modulated Waveform s(t)", color="#E5E5EA", fontname="sans-serif", fontsize=11, fontweight="bold")
        self.ax_mod.set_xlabel("Time (ms)", color="#8E8E93", fontsize=9)
        self.ax_mod.set_ylabel("Amplitude", color="#8E8E93", fontsize=9)
        self.ax_mod.tick_params(colors="#8E8E93")
        
        self.fig.tight_layout(pad=2.0)
        self.canvas.draw()

    # ==========================================
    # Audio Player & Stream
    # ==========================================
    def audio_callback(self, outdata, frames, time_info, status):
        """Callback to generate audio samples dynamically on-the-fly."""
        # Local snapshots of values to avoid mid-buffer changes
        fc = self.carrier_freq
        fm = self.message_freq
        m_index = self.mod_index
        m_type = self.mod_type
        vol = self.volume
        
        t = (np.arange(frames) / self.sample_rate)
        
        # Message Phase step
        fm_step = 2 * np.pi * fm / self.sample_rate
        # Create sequence of message phases starting from current accumulator
        msg_phases = self.phase_message + np.arange(frames) * fm_step
        
        # Message wave
        msg_wave = np.cos(msg_phases)
        
        if m_type == "AM":
            # Carrier phase step
            fc_step = 2 * np.pi * fc / self.sample_rate
            carrier_phases = self.phase_carrier + np.arange(frames) * fc_step
            out = (1.0 + m_index * msg_wave) * np.cos(carrier_phases)
            
            # Save accumulators
            self.phase_message = (msg_phases[-1] + fm_step) % (2 * np.pi)
            self.phase_carrier = (carrier_phases[-1] + fc_step) % (2 * np.pi)
        else:
            # FM: inst freq = fc + dev * cos(2 pi fm t)
            dev = m_index * 500.0
            inst_freq = fc + dev * msg_wave
            
            # Accumulate carrier phases sample by sample
            carrier_phases = np.zeros(frames)
            current_carrier_phase = self.phase_carrier
            for i in range(frames):
                carrier_phases[i] = current_carrier_phase
                current_carrier_phase += 2 * np.pi * inst_freq[i] / self.sample_rate
            
            out = np.cos(carrier_phases)
            
            # Save accumulators
            self.phase_message = (msg_phases[-1] + fm_step) % (2 * np.pi)
            self.phase_carrier = current_carrier_phase % (2 * np.pi)
            
        # Write to outputs
        outdata[:, 0] = out * vol
        if outdata.shape[1] > 1:
            outdata[:, 1] = out * vol # Stereo duplicate

    def toggle_playback(self):
        if self.is_playing:
            self.stop_playback()
        else:
            self.start_playback()
            
    def start_playback(self):
        try:
            self.phase_carrier = 0.0
            self.phase_message = 0.0
            self.stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=2,
                callback=self.audio_callback,
                blocksize=512
            )
            self.stream.start()
            self.is_playing = True
            
            # UI Updates
            self.play_btn.configure(text="Stop Playback", fg_color="#FF453A", hover_color="#D63027")
            self.led_canvas.itemconfig(self.led_circle, fill="#30D158") # Glowing green LED
            self.status_label.configure(text=f"Generating & Streaming: {self.mod_type}")
            
        except Exception as e:
            self.status_label.configure(text=f"Error: {str(e)}")
            self.led_canvas.itemconfig(self.led_circle, fill="#FF453A") # Red LED
            
    def stop_playback(self):
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
            
        self.is_playing = False
        
        # UI Updates
        self.play_btn.configure(text="Generate & Play", fg_color="#0A84FF", hover_color="#0056B3")
        self.led_canvas.itemconfig(self.led_circle, fill="#8E8E93") # Grey LED
        self.status_label.configure(text="System: Idle")

if __name__ == "__main__":
    app = SignalGeneratorApp()
    app.mainloop()
