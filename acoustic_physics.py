"""
Echos — Acoustic Physics (2D)
-----------------------------
SFCW waveform, parametric beam, material database, 2D TDOA, Doppler.
All 3D constructs (diamond array, elevation turret, z-axis) removed.
"""
import numpy as np
import math

SPEED_OF_SOUND = 343.0  # m/s


# ──────────────────────────────────────────────────────────────
# SFCW Waveform
# ──────────────────────────────────────────────────────────────

class SfcwWaveform:
    """
    Stepped-Frequency Continuous Wave.

    5 discrete frequency steps, 40→50 kHz (2.5 kHz increments).
    Sweep time ≈ 290 ms → 3.4 Hz refresh.
    Range resolution: c/(2B) = 343/20000 = 17.15 mm.

    Per-frequency phase:  φ_i = 4π·f_i·R / c  (mod 2π)
    Phase diff (adjacent): Δφ  = 4π·Δf·R / c  (mod 2π)
    Fine range:            R   = c·Δφ / (4π·Δf)
    Max unambiguous:       R   = c / (2·Δf)  = 68.6 mm  → ToF resolves ambiguity
    """

    def __init__(self, freq_start=40_000, freq_end=50_000, num_steps=5, max_range=10.0):
        self.freq_start = freq_start
        self.freq_end = freq_end
        self.num_steps = num_steps
        self.max_range = max_range

        self.frequencies = np.linspace(freq_start, freq_end, num_steps)
        self.bandwidth = freq_end - freq_start          # 10 kHz
        self.step_size = self.bandwidth / (num_steps - 1)  # 2.5 kHz

        self.dwell_time = 2.0 * max_range / SPEED_OF_SOUND   # ~58 ms
        self.sweep_time = self.dwell_time * num_steps         # ~290 ms
        self.refresh_rate = 1.0 / self.sweep_time             # ~3.4 Hz

        self.range_resolution = SPEED_OF_SOUND / (2.0 * self.bandwidth)        # 17.15 mm
        self.max_unambiguous_range = SPEED_OF_SOUND / (2.0 * self.step_size)   # 68.6 mm
        self.wavelengths = SPEED_OF_SOUND / self.frequencies

    def simulate_reflection(self, distance, freq_response=None,
                            reflection_coeff=0.5, noise_std=0.01):
        phases = np.zeros(self.num_steps)
        amplitudes = np.zeros(self.num_steps)
        atten = 1.0 / (1.0 + distance ** 2) if distance > 0.1 else 1.0

        for i, freq in enumerate(self.frequencies):
            ideal = 4.0 * np.pi * freq * distance / SPEED_OF_SOUND
            phases[i] = (ideal + np.random.normal(0, noise_std)) % (2.0 * np.pi)
            mat_gain = freq_response[i] if freq_response is not None else 1.0
            amp_noise = np.random.normal(0, noise_std * 0.1)
            amplitudes[i] = max(0.0, atten * reflection_coeff * mat_gain + amp_noise)

        snr = amplitudes / (noise_std + 1e-10)
        return phases, amplitudes, snr

    def range_from_phases_with_tof(self, phases, tof_range, amplitudes=None):
        if len(phases) < 2:
            return tof_range, tof_range * 0.1

        freq_diffs = np.diff(self.frequencies)
        phase_diffs = np.diff(phases)
        phase_diffs = np.where(phase_diffs > np.pi,  phase_diffs - 2 * np.pi, phase_diffs)
        phase_diffs = np.where(phase_diffs < -np.pi, phase_diffs + 2 * np.pi, phase_diffs)

        fine_ranges = SPEED_OF_SOUND * phase_diffs / (4.0 * np.pi * freq_diffs)
        n_cycles = np.round((tof_range - fine_ranges) / self.max_unambiguous_range)
        resolved = fine_ranges + n_cycles * self.max_unambiguous_range

        if amplitudes is not None and len(amplitudes) > 1:
            weights = np.array([
                np.sqrt(max(0.0, amplitudes[i] * amplitudes[i + 1]))
                for i in range(len(amplitudes) - 1)
            ])
            w_sum = np.sum(weights) + 1e-30
            weights = weights / w_sum
        else:
            weights = np.ones(len(resolved)) / len(resolved)

        estimated = float(np.sum(resolved * weights))
        precision = float(np.sqrt(np.sum(weights * (resolved - estimated) ** 2))) if len(resolved) > 1 else 0.001

        return estimated, precision


# ──────────────────────────────────────────────────────────────
# Acoustic Beam — Throw / Flood dual mode
# ──────────────────────────────────────────────────────────────

class AcousticBeam:
    """
    Parametric phased-array beam, firmware-switched between:

    THROW:  ±3.5° half-angle,  28 dB peak gain — long-range pencil beam
    FLOOD:  ±35°  half-angle,  12 dB peak gain — wide proximity scan

    Gain pattern: Gaussian  G(θ) = G_peak · exp(-2·(θ/θ₀)²)
    """

    def __init__(self):
        self.throw_div = np.radians(3.5)
        self.flood_div = np.radians(35.0)
        self.throw_gain_db = 28.0
        self.flood_gain_db = 12.0

    def gain(self, angle_off_axis, mode='throw'):
        theta_0 = self.throw_div if mode == 'throw' else self.flood_div
        peak_db = self.throw_gain_db if mode == 'throw' else self.flood_gain_db
        peak = 10.0 ** (peak_db / 20.0)
        return peak * np.exp(-2.0 * (angle_off_axis / theta_0) ** 2)

    def half_angle(self, mode='throw'):
        return self.throw_div if mode == 'throw' else self.flood_div


# ──────────────────────────────────────────────────────────────
# Material Database — SFCW frequency fingerprints
# ──────────────────────────────────────────────────────────────

class MaterialDb:
    MATERIALS = {
        'concrete':     {'reflection_coeff': 0.85, 'freq_response': np.array([1.00, 0.98, 0.95, 0.92, 0.88]), 'is_biological': False},
        'drywall':      {'reflection_coeff': 0.60, 'freq_response': np.array([0.90, 0.85, 0.78, 0.70, 0.60]), 'is_biological': False},
        'rubble':       {'reflection_coeff': 0.70, 'freq_response': np.array([0.80, 0.82, 0.85, 0.83, 0.78]), 'is_biological': False},
        'metal':        {'reflection_coeff': 0.95, 'freq_response': np.array([1.00, 1.00, 1.00, 1.00, 0.98]), 'is_biological': False},
        'wood':         {'reflection_coeff': 0.55, 'freq_response': np.array([0.85, 0.80, 0.72, 0.65, 0.55]), 'is_biological': False},
        'glass':        {'reflection_coeff': 0.75, 'freq_response': np.array([0.95, 0.93, 0.90, 0.85, 0.78]), 'is_biological': False},
        'human_tissue': {'reflection_coeff': 0.40, 'freq_response': np.array([0.50, 0.55, 0.70, 0.75, 0.60]), 'is_biological': True},
    }

    @classmethod
    def get(cls, name):
        return cls.MATERIALS.get(name, cls.MATERIALS['concrete'])

    @classmethod
    def identify(cls, observed_response, tolerance=0.15):
        observed = np.array(observed_response, dtype=float)
        norm = np.linalg.norm(observed)
        if norm < 1e-10:
            return 'unknown', 0.0
        observed = observed / norm

        best_name, best_score = 'unknown', -1.0
        for name, props in cls.MATERIALS.items():
            template = props['freq_response']
            template = template / (np.linalg.norm(template) + 1e-10)
            score = float(np.dot(observed, template))
            if score > best_score:
                best_score = score
                best_name = name

        confidence = max(0.0, (best_score - 0.5) / 0.5)
        return (best_name, confidence) if confidence >= tolerance else ('unknown', confidence)


# ──────────────────────────────────────────────────────────────
# 2D TDOA — Left / Right microphone pair
# ──────────────────────────────────────────────────────────────

class TdoaProcessor2D:
    def __init__(self, dx=0.10, timing_noise=1e-5):
        self.dx = dx
        self.timing_noise = timing_noise

    def azimuth(self, source_pos, uav_pos, uav_yaw):
        cos_y, sin_y = math.cos(uav_yaw), math.sin(uav_yaw)
        left_world  = uav_pos + np.array([-sin_y * self.dx,  cos_y * self.dx])
        right_world = uav_pos + np.array([ sin_y * self.dx, -cos_y * self.dx])
        d_left  = float(np.linalg.norm(source_pos - left_world))
        d_right = float(np.linalg.norm(source_pos - right_world))
        t_left  = d_left  / SPEED_OF_SOUND + np.random.normal(0, self.timing_noise)
        t_right = d_right / SPEED_OF_SOUND + np.random.normal(0, self.timing_noise)
        delta = t_right - t_left
        sin_theta = np.clip(SPEED_OF_SOUND * delta / (2.0 * self.dx), -1.0, 1.0)
        return math.asin(sin_theta)


# ──────────────────────────────────────────────────────────────
# Doppler Compensator (2D)
# ──────────────────────────────────────────────────────────────

class DopplerCompensator2D:
    def __init__(self):
        self.velocity = np.zeros(2)

    def update(self, velocity):
        self.velocity = velocity.copy()

    def correct_frequency(self, received_freq, target_direction):
        norm = np.linalg.norm(target_direction)
        if norm < 1e-6:
            return received_freq
        radial_vel = float(np.dot(self.velocity, target_direction / norm))
        return received_freq * (1.0 - 2.0 * radial_vel / SPEED_OF_SOUND)


if __name__ == "__main__":
    print("=== SFCW Physics Verification ===\n")
    sfcw = SfcwWaveform()
    print(f"Frequencies:     {sfcw.frequencies} Hz")
    print(f"Bandwidth:       {sfcw.bandwidth} Hz")
    print(f"Range resolution:{sfcw.range_resolution*1000:.2f} mm")
    print(f"Max unambig:     {sfcw.max_unambiguous_range*1000:.2f} mm")
    print(f"Sweep time:      {sfcw.sweep_time*1000:.1f} ms")
    print(f"Refresh rate:    {sfcw.refresh_rate:.2f} Hz\n")

    print("  True (m)  |  SFCW (m)  |  Error (mm)  |  Precision (mm)")
    print("  " + "-" * 60)
    for true_dist in [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 9.5]:
        phases, amps, snr = sfcw.simulate_reflection(
            true_dist,
            freq_response=MaterialDb.get('concrete')['freq_response'],
            reflection_coeff=0.85, noise_std=0.01,
        )
        est, prec = sfcw.range_from_phases_with_tof(phases, true_dist, amps)
        err_mm = abs(est - true_dist) * 1000
        print(f"  {true_dist:7.2f}  |  {est:9.6f}  |  {err_mm:8.4f}  |  {prec*1000:10.4f}")

    print("\n=== Material Identification ===\n")
    for mat_name in ['concrete', 'drywall', 'rubble', 'metal', 'wood', 'glass']:
        mat = MaterialDb.get(mat_name)
        phases, amps, snr = sfcw.simulate_reflection(
            3.0, mat['freq_response'], mat['reflection_coeff'], noise_std=0.01
        )
        obs = amps / (amps.max() + 1e-10)
        identified, conf = MaterialDb.identify(obs)
        match = "✓" if identified == mat_name else "✗"
        print(f"  {mat_name:10s} → {identified:10s}  conf={conf:.3f}  {match}")
