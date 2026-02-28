# sim/renderer.py
import cv2
import numpy as np
import imageio.v2 as imageio
import os


class VideoRenderer:
    """
    MP4 renderer guaranteed to open in:
      - Windows Media Player
      - VLC
      - Chrome / Firefox

    Uses:
      - FFmpeg backend
      - H.264 (libx264)
      - yuv420p pixel format
    """

    def __init__(self, out_path, fps, base_img, scale=4):
        if not out_path.lower().endswith(".mp4"):
            raise ValueError("This renderer ONLY supports .mp4 output")

        if not hasattr(self, "_dbg_first"):
            self._dbg_first = True
            print("[DEBUG] write_frame called (first frame)")
        
        self.out_path = out_path
        self.fps = fps
        self.scale = scale

        # Enlarge map for visibility
        self.base = cv2.resize(
            base_img,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_NEAREST
        )

        # Ensure even dimensions (some codecs hate odd sizes)
        h, w, _ = self.base.shape
        if h % 2 == 1:
            self.base = self.base[:-1, :, :]
        if w % 2 == 1:
            self.base = self.base[:, :-1, :]

        # --- HARDENED MP4 WRITER ---
        self.writer = imageio.get_writer(
            out_path,
            fps=fps,
            format="FFMPEG",
            codec="libx264",
            macro_block_size=None,
            ffmpeg_params=[
                "-pix_fmt", "yuv420p",
                "-profile:v", "baseline",
                "-movflags", "+faststart"
            ]
        )

        self.frames_written = 0
        print(f"[Renderer] Writing MP4 to {os.path.abspath(out_path)}")

    def write_frame(self, agents, exits, fire_mask, time_seconds):
        frame = self.base.copy()

        # Fire (red)
        fire_cells = np.where(fire_mask)
        for r, c in zip(fire_cells[0], fire_cells[1]):
            cv2.circle(
                frame,
                (int(c * self.scale), int(r * self.scale)),
                self.scale,
                (0, 0, 255),
                -1
            )

        # Exits (green)
        for r, c in exits:
            cv2.rectangle(
                frame,
                (int(c * self.scale - 4), int(r * self.scale - 4)),
                (int(c * self.scale + 4), int(r * self.scale + 4)),
                (0, 255, 0),
                -1
            )

        # Agents (yellow)
        for a in agents:
            r, c = a["pos"]
            cv2.circle(
                frame,
                (int(c * self.scale), int(r * self.scale)),
                self.scale,
                (0, 255, 255),
                -1
            )

        # Timestamp
        cv2.putText(
            frame,
            f"t = {time_seconds:.1f}s",
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        # Convert BGR → RGB (imageio expects RGB)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.writer.append_data(rgb)
        self.frames_written += 1
        print("[DEBUG] writer type:", type(self.writer))

    def close(self):
        if self.writer is not None:
            self.writer.close()
            self.writer = None
            print(f"[Renderer] Finalized MP4 ({self.frames_written} frames)")