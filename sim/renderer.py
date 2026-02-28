import cv2
import imageio
import numpy as np


class VideoRenderer:
    def __init__(self, out_path, fps, base_img, scale=4):
        self.scale = scale
        self.base = cv2.resize(
            base_img, None, fx=scale, fy=scale,
            interpolation=cv2.INTER_NEAREST
        )
        self.writer = imageio.get_writer(out_path, fps=fps)

    def write_frame(self, agents, exits, fire_mask, time_seconds):
        frame = self.base.copy()

        for r,c in zip(*np.where(fire_mask)):
            cv2.circle(frame, (c*self.scale, r*self.scale),
                       self.scale, (0,0,255), -1)

        for r,c in exits:
            cv2.rectangle(
                frame,
                (c*self.scale-4, r*self.scale-4),
                (c*self.scale+4, r*self.scale+4),
                (0,255,0), -1
            )

        for a in agents:
            r,c = a["pos"]
            cv2.circle(frame, (c*self.scale, r*self.scale),
                       self.scale, (0,255,255), -1)

        cv2.putText(
            frame, f"t = {time_seconds:.1f}s",
            (10,20), cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (255,255,255), 2
        )

        self.writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    def close(self):
        self.writer.close()