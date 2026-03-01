# sim/fire.py
import numpy as np


class FireModel:
    """
    Fire + independent smoke diffusion.

    Public:
      - burning: bool mask of active flames (blocks agents)
      - collapsed: bool mask of collapsed floor (permanent block)
      - smoke_field: float32 intensity [0..1], diffuses outward over time
      - smoke: bool mask (smoke_field >= smoke_threshold)
      - blocked: bool mask (burning OR collapsed)
    """

    def __init__(
        self,
        walkable: np.ndarray,
        fire_start: tuple[int, int],
        *,
        rng=None,

        # Fire spread (SLOW)
        p_spread_floor: float = 0.090,
        p_spread_wall: float = 0.010,
        max_new_ignitions_per_step: int = 25,

        # Flame lifetime (keeps flame region from exploding)
        burn_duration_floor: int = 120,
        burn_duration_wall: int = 200,

        # Smoke diffusion (INDEPENDENT expansion)
        smoke_emission: float = 0.7,     # added at burning cells per step
        smoke_decay: float = 0.985,       # smoke persistence
        smoke_diffusion: float = 2,    # higher => expands faster
        smoke_threshold: float = 0.12,    # lower => more visible smoke

        # Collapse (rare)
        p_collapse_floor: float = 0.0012,
        p_collapse_near_wall: float = 0.006,

        # Start snapping
        snap_start_to_walkable: bool = True,
        snap_radius: int = 10,
    ):
        self.walkable = walkable.astype(bool)
        self.H, self.W = self.walkable.shape
        self.wall = ~self.walkable

        self._rng = rng if rng is not None else np.random.default_rng()

        self.p_spread_floor = float(p_spread_floor)
        self.p_spread_wall = float(p_spread_wall)
        self.max_new_ignitions_per_step = int(max_new_ignitions_per_step)

        self.burn_duration_floor = int(burn_duration_floor)
        self.burn_duration_wall = int(burn_duration_wall)

        self.smoke_emission = float(smoke_emission)
        self.smoke_decay = float(smoke_decay)
        self.smoke_diffusion = float(smoke_diffusion)
        self.smoke_threshold = float(smoke_threshold)

        self.p_collapse_floor = float(p_collapse_floor)
        self.p_collapse_near_wall = float(p_collapse_near_wall)

        self.burning = np.zeros((self.H, self.W), dtype=bool)
        self.collapsed = np.zeros((self.H, self.W), dtype=bool)
        self._burn_age = np.zeros((self.H, self.W), dtype=np.int32)

        self.smoke_field = np.zeros((self.H, self.W), dtype=np.float32)

        # Start
        sr, sc = int(fire_start[0]), int(fire_start[1])
        sr = int(np.clip(sr, 0, self.H - 1))
        sc = int(np.clip(sc, 0, self.W - 1))

        if snap_start_to_walkable and not self.walkable[sr, sc]:
            found = False
            for rad in range(1, snap_radius + 1):
                for dr in range(-rad, rad + 1):
                    rr = sr + dr
                    if rr < 0 or rr >= self.H:
                        continue
                    rem = rad - abs(dr)
                    for dc in range(-rem, rem + 1):
                        cc = sc + dc
                        if 0 <= cc < self.W and self.walkable[rr, cc]:
                            sr, sc = rr, cc
                            found = True
                            break
                    if found:
                        break
                if found:
                    break

        self.burning[sr, sc] = True
        self._burn_age[sr, sc] = 1

    @property
    def smoke(self) -> np.ndarray:
        return self.smoke_field >= self.smoke_threshold

    @property
    def blocked(self) -> np.ndarray:
        return self.burning | self.collapsed

    def update(self, step_idx: int):
        # 1) Age & burn out
        if self.burning.any():
            self._burn_age[self.burning] += 1
            burn_out = (
                (self.burning & self.walkable & (self._burn_age >= self.burn_duration_floor))
                | (self.burning & self.wall & (self._burn_age >= self.burn_duration_wall))
            )
            if burn_out.any():
                self.burning[burn_out] = False
                self._burn_age[burn_out] = 0

        # 2) Fire spread (slow + capped)
        burning_cells = np.argwhere(self.burning)
        new_ignitions: list[tuple[int, int]] = []

        for r, c in burning_cells:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if nr < 0 or nr >= self.H or nc < 0 or nc >= self.W:
                    continue
                if self.collapsed[nr, nc] or self.burning[nr, nc]:
                    continue

                p = self.p_spread_floor if self.walkable[nr, nc] else self.p_spread_wall

                # smoke slightly increases ignition likelihood (heat), but not too much
                p = min(1.0, p + 0.01 * float(self.smoke_field[nr, nc]))

                if self._rng.random() < p:
                    new_ignitions.append((nr, nc))
                    if len(new_ignitions) >= self.max_new_ignitions_per_step:
                        break
            if len(new_ignitions) >= self.max_new_ignitions_per_step:
                break

        for nr, nc in new_ignitions:
            self.burning[nr, nc] = True
            self._burn_age[nr, nc] = 1

        # 3) Smoke: emit at fire cells, diffuse outward, decay
        if self.burning.any():
            self.smoke_field[self.burning] = np.clip(
                self.smoke_field[self.burning] + self.smoke_emission, 0.0, 1.0
            )

# 3) Smoke diffusion THROUGH HALLWAYS ONLY (walkable air cells)
        a = self.smoke_diffusion
        if a > 0:
            s = self.smoke_field
            air = self.walkable  # smoke only exists/moves in "air" cells

            # shift smoke AND air masks (for neighbor contributions)
            s_up = np.roll(s, 1, axis=0);    air_up = np.roll(air, 1, axis=0)
            s_dn = np.roll(s, -1, axis=0);   air_dn = np.roll(air, -1, axis=0)
            s_lt = np.roll(s, 1, axis=1);    air_lt = np.roll(air, 1, axis=1)
            s_rt = np.roll(s, -1, axis=1);   air_rt = np.roll(air, -1, axis=1)

            # prevent wrap-around artifacts at borders
            air_up[0, :] = False
            air_dn[-1, :] = False
            air_lt[:, 0] = False
            air_rt[:, -1] = False

            # Only take smoke from neighbors that are air
            neigh_sum = (s_up * air_up) + (s_dn * air_dn) + (s_lt * air_lt) + (s_rt * air_rt)
            neigh_cnt = air_up.astype(np.int32) + air_dn.astype(np.int32) + air_lt.astype(np.int32) + air_rt.astype(np.int32)

            # average of available air-neighbors (avoid divide-by-zero)
            neigh_avg = np.zeros_like(s, dtype=np.float32)
            mask = neigh_cnt > 0
            neigh_avg[mask] = neigh_sum[mask] / neigh_cnt[mask]

            # Diffuse only in air cells; keep walls at 0
            s2 = s.copy()
            s2[air] = (1.0 - a) * s[air] + a * neigh_avg[air]
            s2[~air] = 0.0

            self.smoke_field = np.clip(s2, 0.0, 1.0)

        # Decay after diffusion
        self.smoke_field *= self.smoke_decay

        # 4) Collapse (rare)
        burning_floor = np.argwhere(self.burning & self.walkable & (~self.collapsed))
        for r, c in burning_floor:
            p = self.p_collapse_floor
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.H and 0 <= nc < self.W:
                    if self.wall[nr, nc] and self.burning[nr, nc]:
                        p += self.p_collapse_near_wall
                        break
            if self._rng.random() < p:
                self.collapsed[r, c] = True
                self.burning[r, c] = False
                self._burn_age[r, c] = 0