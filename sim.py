"""
Echos — 2D Simulation Environment
==================================
Deterministic Commander (SCOUT/HUNT/RTH) + correct SFCW sensing + maze navigation.

Observation (10 floats, all normalized [-1,1]):
    0  distance to waypoint        [0, 1]
    1  relative angle to waypoint  [-1, 1]
    2  Path B SNR                  [0, 1]
    3  SFCW ray 1 (leftmost)       [0, 1]
    4  SFCW ray 2                  [0, 1]
    5  SFCW ray 3 (center)         [0, 1]
    6  SFCW ray 4                  [0, 1]
    7  SFCW ray 5 (rightmost)      [0, 1]
    8  beam mode (1=throw, -1=flood)
    9  min wall distance           [0, 1]

Action (2 floats, [-1, 1]):
    0  forward thrust
    1  yaw rate command
"""
import numpy as np
import math
import gymnasium as gym
from gymnasium import spaces
from shapely.geometry import Polygon
import matplotlib.pyplot as plt

from acoustic_physics import (
    SfcwWaveform, AcousticBeam, MaterialDb,
    TdoaProcessor2D, DopplerCompensator2D, SPEED_OF_SOUND,
)


# ═══════════════════════════════════════════════════════════════
# Dynamics
# ═══════════════════════════════════════════════════════════════

class TiltRotorDynamics2D:
    def __init__(self, mass=1.8, drag=0.6, inertia=0.05, max_speed=3.0):
        self.mass = mass
        self.drag = drag
        self.inertia = inertia
        self.max_speed = max_speed
        self.velocity = np.zeros(2)
        self.yaw_rate = 0.0

    def reset(self):
        self.velocity = np.zeros(2)
        self.yaw_rate = 0.0

    def step(self, pos, yaw, action, dt=0.1):
        thrust_cmd, yaw_cmd = action
        target_rate = yaw_cmd * 3.0
        self.yaw_rate += (target_rate - self.yaw_rate) * 0.3
        next_yaw = yaw + self.yaw_rate * dt
        force = thrust_cmd * 15.0
        thrust_vec = np.array([math.cos(next_yaw), math.sin(next_yaw)]) * force
        speed = np.linalg.norm(self.velocity)
        drag_vec = -self.drag * self.velocity * speed
        accel = (thrust_vec + drag_vec) / self.mass
        self.velocity += accel * dt
        speed = np.linalg.norm(self.velocity)
        if speed > self.max_speed:
            self.velocity *= self.max_speed / speed
        next_pos = pos + self.velocity * dt
        return next_pos, next_yaw


# ═══════════════════════════════════════════════════════════════
# Maze
# ═══════════════════════════════════════════════════════════════

class MazeGenerator:
    def __init__(self, width=12.0, height=9.0):
        self.width = width
        self.height = height
        self.walls = []
        self.wall_materials = {}
        self.scout_waypoints = []

    def generate(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.walls = []
        self.wall_materials = {}

        # Boundary (concrete, 0.3 m thick)
        t = 0.3
        outer = Polygon([(-t, -t), (self.width + t, -t),
                         (self.width + t, self.height + t), (-t, self.height + t)])
        inner = Polygon([(0, 0), (self.width, 0),
                         (self.width, self.height), (0, self.height)])
        boundary = outer.difference(inner)
        self._add_poly(boundary, 'concrete')

        # Internal walls with gaps (zigzag path: up → right → down → right → up → right)
        self._add_poly(Polygon([(2.5, 0), (3.0, 0), (3.0, 5.0), (2.5, 5.0)]), 'concrete')
        self._add_poly(Polygon([(5.5, 4), (6.0, 4), (6.0, 9), (5.5, 9)]), 'concrete')
        self._add_poly(Polygon([(8.0, 0), (8.5, 0), (8.5, 5.0), (8.0, 5.0)]), 'concrete')

        # Small obstacles
        self._add_poly(Polygon([(4.3, 6.3), (5.0, 6.3), (4.6, 7.0)]), 'rubble')
        self._add_poly(Polygon([(7.3, 6.3), (8.0, 6.3), (8.0, 7.1), (7.3, 7.1)]), 'metal')
        self._add_poly(Polygon([(4.3, 2.2), (5.0, 2.2), (5.0, 2.8), (4.3, 2.8)]), 'drywall')

        # Scout waypoints — follow navigable path
        self.scout_waypoints = [
            np.array([1.5, 7.0]),
            np.array([4.2, 7.0]),
            np.array([4.2, 2.0]),
            np.array([7.2, 2.0]),
            np.array([7.2, 7.0]),
            np.array([10.0, 7.0]),
            np.array([11.0, 4.5]),
        ]
        return self.walls, self.wall_materials

    def _add_poly(self, poly, material):
        if poly.geom_type == 'Polygon':
            self.walls.append(poly)
            self.wall_materials[id(poly)] = material
        elif poly.geom_type == 'MultiPolygon':
            for p in poly.geoms:
                self.walls.append(p)
                self.wall_materials[id(p)] = material


# ═══════════════════════════════════════════════════════════════
# Acoustic Ray Tracer
# ═══════════════════════════════════════════════════════════════

class AcousticRayTracer:
    def __init__(self, walls, wall_materials):
        self.walls = walls
        self.wall_materials = wall_materials
        self._segments = []
        for wall in walls:
            coords = list(wall.exterior.coords)
            for i in range(len(coords) - 1):
                self._segments.append((
                    np.array(coords[i], dtype=float),
                    np.array(coords[i + 1], dtype=float),
                    wall,
                ))

    def trace_ray(self, origin, direction, max_range=10.0):
        origin = np.array(origin, dtype=float)
        direction = np.array(direction, dtype=float)
        norm = np.linalg.norm(direction)
        if norm < 1e-10:
            return max_range, None, None
        direction = direction / norm

        nearest_t = max_range
        nearest_mat = None
        nearest_pt = None

        for seg_a, seg_b, wall in self._segments:
            t = self._ray_seg_dist(origin, direction, seg_a, seg_b)
            if t is not None and 0.01 < t < nearest_t:
                nearest_t = t
                nearest_mat = self.wall_materials.get(id(wall), 'concrete')
                nearest_pt = origin + direction * t

        return nearest_t, nearest_mat, nearest_pt

    @staticmethod
    def _ray_seg_dist(O, D, A, B):
        BAx, BAy = B[0] - A[0], B[1] - A[1]
        AOx, AOy = A[0] - O[0], A[1] - O[1]
        denom = D[0] * (-BAy) - D[1] * (-BAx)
        if abs(denom) < 1e-10:
            return None
        t = (AOx * (-BAy) - AOy * (-BAx)) / denom
        s = (D[0] * AOy - D[1] * AOx) / denom
        if t > 0.01 and -0.001 <= s <= 1.001:
            return t
        return None

    def trace_beam(self, origin, yaw, mode='throw', num_rays=5, max_range=10.0):
        beam = AcousticBeam()
        half = beam.half_angle(mode)
        angles = np.linspace(-half, half, num_rays)
        results = []
        for a_off in angles:
            ray_angle = yaw + a_off
            direction = np.array([math.cos(ray_angle), math.sin(ray_angle)])
            dist, mat_name, hit_pt = self.trace_ray(origin, direction, max_range)
            gain = beam.gain(abs(a_off), mode)
            results.append({
                'distance': dist,
                'material_name': mat_name,
                'gain': gain,
                'angle_offset': float(a_off),
                'hit_point': hit_pt,
            })
        return results


# ═══════════════════════════════════════════════════════════════
# SFCW Sensor
# ═══════════════════════════════════════════════════════════════

class SfcwSensor:
    def __init__(self, sfcw, beam, ray_tracer, noise_std=0.01):
        self.sfcw = sfcw
        self.beam = beam
        self.ray_tracer = ray_tracer
        self.noise_std = noise_std

    def sweep(self, origin, yaw, mode='throw', num_rays=5, max_range=10.0):
        rays = self.ray_tracer.trace_beam(origin, yaw, mode, num_rays, max_range)
        results = []
        for ray in rays:
            dist = ray['distance']
            mat_name = ray['material_name']
            gain = ray['gain']
            if mat_name is not None and dist < max_range:
                mat = MaterialDb.get(mat_name)
                phases, amplitudes, snr = self.sfcw.simulate_reflection(
                    dist, mat['freq_response'], mat['reflection_coeff'], self.noise_std,
                )
                amplitudes = amplitudes * gain
                sfcw_range, precision = self.sfcw.range_from_phases_with_tof(phases, dist, amplitudes)
                obs_response = amplitudes / (amplitudes.max() + 1e-10)
                id_name, id_conf = MaterialDb.identify(obs_response)
                mean_snr_db = 10.0 * np.log10(np.mean(snr) + 1e-10)
                results.append({
                    'true_distance': dist,
                    'sfcw_range': sfcw_range,
                    'precision': precision,
                    'phases': phases,
                    'amplitudes': amplitudes,
                    'snr_db': mean_snr_db,
                    'material_true': mat_name,
                    'material_id': id_name,
                    'material_conf': id_conf,
                    'angle_offset': ray['angle_offset'],
                    'gain': gain,
                    'hit_point': ray['hit_point'],
                })
            else:
                results.append({
                    'true_distance': max_range,
                    'sfcw_range': max_range,
                    'precision': max_range * 0.1,
                    'phases': None,
                    'amplitudes': None,
                    'snr_db': -20.0,
                    'material_true': None,
                    'material_id': 'open',
                    'material_conf': 0.0,
                    'angle_offset': ray['angle_offset'],
                    'gain': gain,
                    'hit_point': None,
                })
        return results


# ═══════════════════════════════════════════════════════════════
# Path B Receiver
# ═══════════════════════════════════════════════════════════════

class PathBReceiver:
    def __init__(self, source_db=80.0, noise_floor_db=35.0, threshold=4.5):
        self.source_db = source_db
        self.noise_floor_db = noise_floor_db
        self.threshold = threshold

    def compute_snr(self, uav_pos, survivor_pos, velocity=np.zeros(2)):
        dist = max(0.1, float(np.linalg.norm(survivor_pos - uav_pos)))
        spl = self.source_db - 20.0 * np.log10(dist)
        motion_penalty = float(np.linalg.norm(velocity)) * 2.0
        snr = spl - self.noise_floor_db - motion_penalty + np.random.normal(0, 0.5)
        return max(0.0, snr)


# ═══════════════════════════════════════════════════════════════
# Algorithmic Commander
# ═══════════════════════════════════════════════════════════════

class AlgorithmicCommander:
    """
    All deterministic logic. RL only provides micro-kinematics.

    SCOUT → HUNT:  Path B SNR > 4.5 for 3 consecutive readings
    HUNT → RTH:    SNR > 14.0 OR distance to survivor < 0.5 m
    RTH:           Navigate to home, mission complete when within 0.8 m

    Beam mode:
      SCOUT/RTH: throw default, flood when center ray < 3 m or peripheral < 1.5 m
      HUNT:      throw (narrow beam for gradient following)
    """

    SCOUT, HUNT, RTH = 'SCOUT', 'HUNT', 'RTH'

    def __init__(self):
        self.state = self.SCOUT
        self.beam_mode = 'throw'
        self.waypoint = np.array([5.0, 4.5])
        self.home = np.array([1.0, 0.5])
        self.beacon_dropped = False
        self.survivor_confirmed_pos = None
        self._snr_history = []
        self._hunt_confirm = 0
        self._wp_idx = 0
        self._scout_waypoints = []

    def reset(self, home, scout_waypoints):
        self.state = self.SCOUT
        self.beam_mode = 'throw'
        self.home = home.copy()
        self._scout_waypoints = [wp.copy() for wp in scout_waypoints]
        self._wp_idx = 0
        self.waypoint = self._scout_waypoints[0].copy() if self._scout_waypoints else home.copy()
        self.beacon_dropped = False
        self.survivor_confirmed_pos = None
        self._snr_history = []
        self._hunt_confirm = 0

    def update(self, uav_pos, uav_yaw, snr, sfcw_results,
               survivor_pos, velocity, tdoa):
        info = {}

        self._snr_history.append(snr)
        if len(self._snr_history) > 10:
            self._snr_history.pop(0)

        # Beam mode
        if self.state in (self.SCOUT, self.RTH):
            self.beam_mode = self._auto_beam_mode(sfcw_results)
        else:
            self.beam_mode = 'throw'

        # State machine
        if self.state == self.SCOUT:
            dist_to_wp = np.linalg.norm(uav_pos - self.waypoint)
            if dist_to_wp < 1.0 and self._wp_idx < len(self._scout_waypoints) - 1:
                self._wp_idx += 1
                self.waypoint = self._scout_waypoints[self._wp_idx].copy()
            if len(self._snr_history) >= 3 and np.mean(self._snr_history[-3:]) > 4.5:
                self.state = self.HUNT
                self._hunt_confirm = 0
                info['transition'] = 'SCOUT→HUNT'

        elif self.state == self.HUNT:
            if snr > 4.5:
                self._hunt_confirm += 1
            else:
                self._hunt_confirm = max(0, self._hunt_confirm - 1)

            # TDOA azimuth for direction (not direct position)
            azimuth = tdoa.azimuth(survivor_pos, uav_pos, uav_yaw)
            look_ahead = 2.0
            world_angle = uav_yaw + azimuth
            self.waypoint = uav_pos + np.array([
                math.cos(world_angle) * look_ahead,
                math.sin(world_angle) * look_ahead,
            ])

            dist_to_surv = np.linalg.norm(uav_pos - survivor_pos)
            if snr > 14.0 or dist_to_surv < 0.5:
                self.state = self.RTH
                self.beacon_dropped = True
                self.survivor_confirmed_pos = survivor_pos.copy()
                self.waypoint = self.home.copy()
                info['transition'] = 'HUNT→RTH'
                info['beacon_dropped'] = True

        elif self.state == self.RTH:
            self.waypoint = self.home.copy()
            if np.linalg.norm(uav_pos - self.home) < 0.8:
                info['mission_complete'] = True

        return self.waypoint.copy(), self.beam_mode, self.state, info

    @staticmethod
    def _auto_beam_mode(sfcw_results):
        if not sfcw_results:
            return 'throw'
        n = len(sfcw_results)
        center = sfcw_results[n // 2]['true_distance']
        if center < 3.0:
            return 'flood'
        peripheral = [r['true_distance'] for i, r in enumerate(sfcw_results) if i != n // 2]
        if any(d < 1.5 for d in peripheral):
            return 'flood'
        return 'throw'


# ═══════════════════════════════════════════════════════════════
# Gym Environment
# ═══════════════════════════════════════════════════════════════

class EchosEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, render_mode=None):
        self.dynamics = TiltRotorDynamics2D()
        self.maze = MazeGenerator()
        self.maze.generate()
        self.ray_tracer = AcousticRayTracer(self.maze.walls, self.maze.wall_materials)
        self.sfcw = SfcwWaveform()
        self.beam = AcousticBeam()
        self.sfcw_sensor = SfcwSensor(self.sfcw, self.beam, self.ray_tracer)
        self.path_b = PathBReceiver()
        self.doppler = DopplerCompensator2D()
        self.tdoa = TdoaProcessor2D()
        self.commander = AlgorithmicCommander()

        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        self.dt = 0.1
        self.max_steps = 1500
        self.total_steps = 0
        self.uav_pos = np.array([1.0, 0.5])
        self.uav_yaw = 0.0
        self.survivor_pos = np.array([10.5, 7.5])

        self._sensor_counter = 0
        self._cached_sfcw = None
        self._cached_snr = 0.0
        self._cached_min_dist = 10.0
        self._prev_dist_wp = 0.0
        self._prev_snr = 0.0
        self._beacon_rewarded = False

        self.render_mode = render_mode
        self.fig, self.ax = None, None
        self.telemetry = []

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.maze.generate(seed=seed)
            self.ray_tracer = AcousticRayTracer(self.maze.walls, self.maze.wall_materials)
            self.sfcw_sensor.ray_tracer = self.ray_tracer

        self.dynamics.reset()
        self.uav_pos = np.array([1.0, 0.5])
        self.uav_yaw = 0.0
        self.survivor_pos = np.array([
            np.random.uniform(9.0, 11.0),
            np.random.uniform(6.0, 8.0),
        ])
        self.commander.reset(self.uav_pos.copy(), self.maze.scout_waypoints)

        self.total_steps = 0
        self._sensor_counter = 0
        self._prev_dist_wp = np.linalg.norm(self.uav_pos - self.commander.waypoint)
        self._prev_snr = 0.0
        self._beacon_rewarded = False
        self.telemetry = []

        self._update_sensors()
        return self._get_obs(), {}

    def _update_sensors(self):
        self.doppler.update(self.dynamics.velocity)
        self._cached_sfcw = self.sfcw_sensor.sweep(
            self.uav_pos, self.uav_yaw,
            mode=self.commander.beam_mode,
            num_rays=5, max_range=10.0,
        )
        self._cached_snr = self.path_b.compute_snr(
            self.uav_pos, self.survivor_pos, self.dynamics.velocity
        )
        self._cached_min_dist = min(r['true_distance'] for r in self._cached_sfcw)

    def _get_obs(self):
        if self._sensor_counter % 3 == 0:
            self._update_sensors()

        dist_wp = np.linalg.norm(self.uav_pos - self.commander.waypoint)
        rel_angle = math.atan2(
            self.commander.waypoint[1] - self.uav_pos[1],
            self.commander.waypoint[0] - self.uav_pos[0],
        ) - self.uav_yaw
        rel_angle = (rel_angle + np.pi) % (2 * np.pi) - np.pi

        rays = [r['sfcw_range'] for r in self._cached_sfcw]

        obs = np.array([
            np.clip(dist_wp / 15.0, 0.0, 1.0),
            np.clip(rel_angle / np.pi, -1.0, 1.0),
            np.clip(self._cached_snr / 20.0, 0.0, 1.0),
            np.clip(rays[0] / 10.0, 0.0, 1.0),
            np.clip(rays[1] / 10.0, 0.0, 1.0),
            np.clip(rays[2] / 10.0, 0.0, 1.0),
            np.clip(rays[3] / 10.0, 0.0, 1.0),
            np.clip(rays[4] / 10.0, 0.0, 1.0),
            1.0 if self.commander.beam_mode == 'throw' else -1.0,
            np.clip(self._cached_min_dist / 10.0, 0.0, 1.0),
        ], dtype=np.float32)
        return obs

    def _reward(self, action):
        r = 0.0
        done = False
        info = {}

        if self._cached_min_dist < 0.2:
            return -20.0, True, {'reason': 'collision'}
        if self._cached_min_dist < 0.5:
            r -= (0.5 - self._cached_min_dist) * 3.0

        dist_wp = np.linalg.norm(self.uav_pos - self.commander.waypoint)
        progress = self._prev_dist_wp - dist_wp
        self._prev_dist_wp = dist_wp

        if self.commander.state == 'SCOUT':
            r += progress * 1.0
        elif self.commander.state == 'HUNT':
            snr_prog = self._cached_snr - self._prev_snr
            self._prev_snr = self._cached_snr
            r += progress * 2.0 + snr_prog * 0.3
        elif self.commander.state == 'RTH':
            r += progress * 2.0

        if dist_wp < 0.8:
            r += 1.0
        if self.commander.beacon_dropped and not self._beacon_rewarded:
            r += 10.0
            self._beacon_rewarded = True
        if self.commander.state == 'RTH':
            if np.linalg.norm(self.uav_pos - self.commander.home) < 0.8:
                r += 20.0
                return r, True, {'reason': 'mission_complete'}

        r -= 0.01
        speed = np.linalg.norm(self.dynamics.velocity)
        if speed > 0.5 and abs(action[1]) > 0.5:
            r -= abs(action[1]) * 0.02

        return r, done, info

    def step(self, action):
        action = np.clip(np.array(action, dtype=float), -1.0, 1.0)
        self.uav_pos, self.uav_yaw = self.dynamics.step(
            self.uav_pos, self.uav_yaw, action, self.dt
        )
        self.uav_pos[0] = np.clip(self.uav_pos[0], 0.1, self.maze.width - 0.1)
        self.uav_pos[1] = np.clip(self.uav_pos[1], 0.1, self.maze.height - 0.1)

        self.total_steps += 1
        self._sensor_counter += 1

        if self._sensor_counter % 3 == 0:
            self._update_sensors()

        wp, beam, state, cmd_info = self.commander.update(
            self.uav_pos, self.uav_yaw,
            self._cached_snr, self._cached_sfcw,
            self.survivor_pos, self.dynamics.velocity, self.tdoa,
        )

        reward, terminated, term_info = self._reward(action)
        truncated = self.total_steps >= self.max_steps
        obs = self._get_obs()

        info = {
            'command': state,
            'beam_mode': beam,
            'path_b_snr': self._cached_snr,
            'min_dist': self._cached_min_dist,
            'steps': self.total_steps,
            'beacon_dropped': self.commander.beacon_dropped,
            **cmd_info, **term_info,
        }

        self.telemetry.append({
            'step': self.total_steps,
            'pos': self.uav_pos.copy(),
            'yaw': self.uav_yaw,
            'command': state,
            'beam_mode': beam,
            'snr': self._cached_snr,
            'min_dist': self._cached_min_dist,
            'sfcw': [(r['sfcw_range'], r['true_distance'], r['material_true'],
                       r['material_id']) for r in self._cached_sfcw],
        })

        return obs, reward, terminated, truncated, info


# ═══════════════════════════════════════════════════════════════
# Proportional Controller
# ═══════════════════════════════════════════════════════════════

def proportional_controller(obs):
    rel_angle = obs[1] * np.pi
    min_dist = obs[9] * 10.0
    yaw_cmd = np.clip(rel_angle / (np.pi * 0.5) * 2.0, -1.0, 1.0)
    alignment = max(0.0, math.cos(rel_angle))
    thrust = alignment * 0.7
    if min_dist < 1.0:
        thrust *= min_dist / 1.0
    return np.array([thrust, yaw_cmd], dtype=np.float32)


if __name__ == "__main__":
    env = EchosEnv(render_mode="human")
    obs, _ = env.reset()
    print("=" * 60)
    print("  Echos 2D — Navigation Test (Proportional Controller)")
    print("=" * 60)

    for step in range(1500):
        action = proportional_controller(obs)
        obs, reward, term, trunc, info = env.step(action)
        if step % 50 == 0:
            print(f"  Step {step:4d} | {info['command']:5s} | "
                  f"SNR {info['path_b_snr']:5.1f} | "
                  f"Wall {info['min_dist']:.2f} m | "
                  f"R {reward:+.2f}")
        if term or trunc:
            print(f"\n  Episode end: {info}")
            break

    print("\n  SFCW sensing log (last sweep):")
    for i, r in enumerate(env._cached_sfcw):
        print(f"    Ray {i}: true={r['true_distance']:.4f} m  "
              f"sfcw={r['sfcw_range']:.6f} m  "
              f"mat={r['material_true']}→{r['material_id']} "
              f"(conf {r['material_conf']:.2f})")
