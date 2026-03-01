# sim/renderer.py
import os
import cv2
import numpy as np
import imageio.v2 as imageio


class VideoRenderer:
    """
    Fast MP4 renderer with smoke overlay.

    Call:
      write_frame(agents, exits, fire_mask, time_seconds, smoke_mask=fire.smoke, alarm_active=False)

    Notes:
      - fire_mask: bool grid (H,W) -> drawn red (solid)
      - smoke_mask: bool OR float grid (H,W); float is treated as density [0..1]
    """

    def __init__(self, out_path, fps, base_img, scale=4, exits=None):
        self.out_path = out_path
        self.fps = int(fps)
        self.scale = int(scale)

        # Upscale base image for visibility
        self.base = cv2.resize(
            base_img,
            None,
            fx=self.scale,
            fy=self.scale,
            interpolation=cv2.INTER_NEAREST,
        )

        # Ensure even dimensions (helps codecs)
        h, w, _ = self.base.shape
        if h % 2 == 1:
            self.base = self.base[:-1, :, :]
        if w % 2 == 1:
            self.base = self.base[:, :-1, :]

        self.writer = imageio.get_writer(
            out_path,
            fps=self.fps,
            format="FFMPEG",
            codec="libx264",
            macro_block_size=None,
            ffmpeg_params=[
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                "-preset", "ultrafast",
                "-crf", "30",
            ],
        )
        self._static_with_exits = None
        self._exits_signature = None
        if exits is not None:
            self._build_static_with_exits(exits)

        self.frames_written = 0
        print(f"[Renderer] Writing MP4 to {os.path.abspath(out_path)}")

    def write_frame(
        self,
        agents,
        exits,
        fire_mask,
        time_seconds,
        smoke_mask=None,
        alarm_active=False,
        death_count=None,
        survival_rate=None,
        doors=None,
        open_doors=None,
    ):
        if self._static_with_exits is None:
            self._build_static_with_exits(exits)
        frame = self._static_with_exits.copy()

        # ---- SMOKE OVERLAY (vectorized) ----
        if smoke_mask is not None:
            if smoke_mask.dtype == np.bool_:
                sm = smoke_mask.astype(np.float32)
            else:
                sm = np.asarray(smoke_mask, dtype=np.float32)
                sm = np.clip(sm, 0.0, 1.0)

            sm_big = cv2.resize(sm, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
            sm_big = np.clip(sm_big, 0.0, 1.0)

            if np.any(sm_big > 0.01):
                # Denser smoke -> darker, more opaque haze (more visible on bright maps).
                alpha = 0.12 + 0.70 * sm_big
                alpha3 = alpha[..., None]
                haze = np.full_like(frame, 70)
                frame = (frame * (1.0 - alpha3) + haze * alpha3).astype(np.uint8)

        # ---- FAST FIRE DRAW (vectorized) ----
        fm = (fire_mask.astype(np.uint8) * 255)
        fm_big = cv2.resize(fm, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
        frame[fm_big > 0] = (0, 0, 255)  # red (BGR)

        # Draw doors: closed=magenta, open=cyan
        if doors:
            open_set = set(open_doors or [])
            for r, c in doors:
                x = int(c * self.scale)
                y = int(r * self.scale)
                color = (255, 255, 0) if (r, c) in open_set else (255, 0, 255)
                cv2.rectangle(frame, (x - 2, y - 2), (x + 2, y + 2), color, -1)

        # Agents (small count, loop is fine)
        for a in agents:
            r, c = a["pos"]
            x = int(c * self.scale)
            y = int(r * self.scale)
            cv2.circle(frame, (x, y), self.scale, (0, 255, 255), -1)

        # Timestamp
        cv2.putText(
            frame,
            f"t = {time_seconds:.1f}s",
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        if death_count is not None:
            cv2.putText(
                frame,
                f"Deaths: {int(death_count)}",
                (10, 86),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (255, 255, 255),
                2,
            )
        if survival_rate is not None:
            cv2.putText(
                frame,
                f"Survival: {float(survival_rate):.1f}%",
                (10, 108),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (255, 255, 255),
                2,
            )

        if alarm_active:
            cv2.rectangle(frame, (10, 34), (280, 64), (0, 0, 255), -1)
            cv2.putText(
                frame,
                "FIRE ALARM ON",
                (16, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.writer.append_data(rgb)
        self.frames_written += 1

    def _build_static_with_exits(self, exits):
        sig = tuple(exits)
        if self._exits_signature == sig and self._static_with_exits is not None:
            return

        frame = self.base.copy()
        for r, c in exits:
            x = int(c * self.scale)
            y = int(r * self.scale)
            cv2.rectangle(frame, (x - 4, y - 4), (x + 4, y + 4), (0, 255, 0), -1)

        self._static_with_exits = frame
        self._exits_signature = sig

    def close(self):
        if self.writer is not None:
            self.writer.close()
            self.writer = None
            print(f"[Renderer] Finalized MP4 ({self.frames_written} frames)")
