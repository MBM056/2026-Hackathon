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
        smoke_emission: float = 1.6,      # added at burning cells per step
        smoke_decay: float = 0.985,       # higher persistence (closer to original behavior)
        smoke_diffusion: float = 0.86,    # 0..1; higher => expands faster
        smoke_diffusion_passes: int = 9,  # ~1.5x faster spread per tick (vs 6)
        smoke_front_gain: float = 0.96,   # stronger front propagation
        smoke_front_steps: int = 2,       # multi-step front push/tick
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
        self.smoke_diffusion_passes = int(smoke_diffusion_passes)
        self.smoke_front_gain = float(smoke_front_gain)
        self.smoke_front_steps = int(smoke_front_steps)
        self.smoke_threshold = float(smoke_threshold)

        self.p_collapse_floor = float(p_collapse_floor)
        self.p_collapse_near_wall = float(p_collapse_near_wall)

        self.burning = np.zeros((self.H, self.W), dtype=bool)
        self.collapsed = np.zeros((self.H, self.W), dtype=bool)
        self._burn_age = np.zeros((self.H, self.W), dtype=np.int32)

        self.smoke_field = np.zeros((self.H, self.W), dtype=np.float32)
        self.smoke_age = np.zeros((self.H, self.W), dtype=np.int16)

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

        # 3) Smoke: emit at fire cells, diffuse outward, and evolve over time.
        if self.burning.any():
            # If much of the air is already smoky, fresh source emission lingers more.
            smoky_frac = float(np.mean(self.smoke_field[self.walkable] > (self.smoke_threshold * 0.7)))
            source_boost = 1.0 + 0.8 * smoky_frac
            self.smoke_field[self.burning] = np.clip(
                self.smoke_field[self.burning] + self.smoke_emission * source_boost, 0.0, 1.0
            )
            # Freshly emitted smoke near flames should be dense.
            self.smoke_field[self.burning] = np.maximum(self.smoke_field[self.burning], 0.85)

            # Push smoke ahead of flames so it visibly leads the fire front.
            burning_cells = np.argwhere(self.burning)
            for r, c in burning_cells:
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1),
                               (1, 1), (1, -1), (-1, 1), (-1, -1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.H and 0 <= nc < self.W and self.walkable[nr, nc]:
                        self.smoke_field[nr, nc] = max(
                            self.smoke_field[nr, nc],
                            0.42 + 0.16 * source_boost
                        )

                # Second ring for "smoke ahead of fire" effect in corridors.
                for dr, dc in ((2, 0), (-2, 0), (0, 2), (0, -2),
                               (2, 2), (2, -2), (-2, 2), (-2, -2)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.H and 0 <= nc < self.W and self.walkable[nr, nc]:
                        self.smoke_field[nr, nc] = max(self.smoke_field[nr, nc], 0.45)

        # 3a) Smoke front propagation + multi-pass diffusion THROUGH HALLWAYS ONLY.
        a = float(np.clip(self.smoke_diffusion, 0.0, 1.0))
        passes = max(1, self.smoke_diffusion_passes)
        if a > 0:
            s = self.smoke_field
            air = self.walkable

            # Front propagation: move current smoke frontier outward quickly.
            gain = float(np.clip(self.smoke_front_gain, 0.0, 1.0))
            if gain > 0:
                for _ in range(max(1, self.smoke_front_steps)):
                    s_up = np.roll(s, 1, axis=0)
                    s_dn = np.roll(s, -1, axis=0)
                    s_lt = np.roll(s, 1, axis=1)
                    s_rt = np.roll(s, -1, axis=1)
                    s_ul = np.roll(s_up, 1, axis=1)
                    s_ur = np.roll(s_up, -1, axis=1)
                    s_dl = np.roll(s_dn, 1, axis=1)
                    s_dr = np.roll(s_dn, -1, axis=1)

                    # prevent wrap-around artifacts
                    s_up[0, :] = 0.0
                    s_dn[-1, :] = 0.0
                    s_lt[:, 0] = 0.0
                    s_rt[:, -1] = 0.0
                    s_ul[0, :] = 0.0; s_ul[:, 0] = 0.0
                    s_ur[0, :] = 0.0; s_ur[:, -1] = 0.0
                    s_dl[-1, :] = 0.0; s_dl[:, 0] = 0.0
                    s_dr[-1, :] = 0.0; s_dr[:, -1] = 0.0

                    front = np.maximum.reduce((s_up, s_dn, s_lt, s_rt, s_ul, s_ur, s_dl, s_dr))
                    s2 = s.copy()
                    s2[air] = np.maximum(s2[air], front[air] * gain)
                    s = np.clip(s2, 0.0, 1.0)

            # Multi-pass diffusion with 8-neighbor weighted average.
            for _ in range(passes):
                s_up = np.roll(s, 1, axis=0);    air_up = np.roll(air, 1, axis=0)
                s_dn = np.roll(s, -1, axis=0);   air_dn = np.roll(air, -1, axis=0)
                s_lt = np.roll(s, 1, axis=1);    air_lt = np.roll(air, 1, axis=1)
                s_rt = np.roll(s, -1, axis=1);   air_rt = np.roll(air, -1, axis=1)
                s_ul = np.roll(s_up, 1, axis=1); air_ul = np.roll(air_up, 1, axis=1)
                s_ur = np.roll(s_up, -1, axis=1);air_ur = np.roll(air_up, -1, axis=1)
                s_dl = np.roll(s_dn, 1, axis=1); air_dl = np.roll(air_dn, 1, axis=1)
                s_dr = np.roll(s_dn, -1, axis=1);air_dr = np.roll(air_dn, -1, axis=1)

                air_up[0, :] = False
                air_dn[-1, :] = False
                air_lt[:, 0] = False
                air_rt[:, -1] = False
                air_ul[0, :] = False; air_ul[:, 0] = False
                air_ur[0, :] = False; air_ur[:, -1] = False
                air_dl[-1, :] = False; air_dl[:, 0] = False
                air_dr[-1, :] = False; air_dr[:, -1] = False

                neigh_sum = (
                    (s_up * air_up + s_dn * air_dn + s_lt * air_lt + s_rt * air_rt)
                    + 0.7 * (s_ul * air_ul + s_ur * air_ur + s_dl * air_dl + s_dr * air_dr)
                )
                neigh_cnt = (
                    (air_up.astype(np.float32) + air_dn.astype(np.float32) + air_lt.astype(np.float32) + air_rt.astype(np.float32))
                    + 0.7 * (air_ul.astype(np.float32) + air_ur.astype(np.float32) + air_dl.astype(np.float32) + air_dr.astype(np.float32))
                )

                neigh_avg = np.zeros_like(s, dtype=np.float32)
                mask = neigh_cnt > 0
                neigh_avg[mask] = neigh_sum[mask] / neigh_cnt[mask]

                s2 = s.copy()
                s2[air] = (1.0 - a) * s[air] + a * neigh_avg[air]
                s2[~air] = 0.0
                s = np.clip(s2, 0.0, 1.0)

            self.smoke_field = s

        # 3b) Smoke aging:
        # Older smoke thins naturally, but if the space is saturated it retains density longer.
        present = self.smoke_field > 0.02
        self.smoke_age[present] = np.minimum(self.smoke_age[present] + 1, np.iinfo(np.int16).max)
        self.smoke_age[~present] = 0

        # As smoky coverage increases, decay weakens (room filling / poor dispersal).
        coverage = float(np.mean(self.smoke_field[self.walkable] > (self.smoke_threshold * 0.7)))
        retention = float(np.clip(self.smoke_decay + 0.22 * coverage, 0.80, 0.998))
        self.smoke_field[self.walkable] *= retention

        # Local thinning of old smoke pockets unless renewed by fresh source.
        # Use a small additive drain so smoke remains visible over time.
        age_norm = np.clip(self.smoke_age.astype(np.float32) / 700.0, 0.0, 1.0)
        self.smoke_field[self.walkable] = np.maximum(
            0.0,
            self.smoke_field[self.walkable] - 0.003 * age_norm[self.walkable]
        )
        if self.burning.any():
            self.smoke_field[self.burning] = np.maximum(self.smoke_field[self.burning], 0.85)
        self.smoke_field = np.clip(self.smoke_field, 0.0, 1.0)

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
