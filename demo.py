"""
Echos — Verification & Visualization Demo
==========================================
1. SFCW range accuracy test (true vs measured)
2. Material classification test
3. Live navigation with proportional controller
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import math

from acoustic_physics import SfcwWaveform, AcousticBeam, MaterialDb, SPEED_OF_SOUND
from sim import (EchosEnv, MazeGenerator, AcousticRayTracer, SfcwSensor,
                 PathBReceiver, proportional_controller)

plt.rcParams.update({'figure.dpi': 120, 'font.size': 10})

MAT_COLORS = {
    'concrete': '#b0b0b0', 'drywall': '#fff3e0',
    'rubble': '#d7ccc8', 'metal': '#cfd8dc', 'wood': '#deb887', 'glass': '#e0ffff',
}


def test_sfcw_accuracy():
    print("\n" + "=" * 60)
    print("  TEST 1: SFCW Range Accuracy")
    print("=" * 60)

    sfcw = SfcwWaveform()
    beam = AcousticBeam()
    maze = MazeGenerator()
    maze.generate()
    tracer = AcousticRayTracer(maze.walls, maze.wall_materials)
    sensor = SfcwSensor(sfcw, beam, tracer, noise_std=0.01)

    print(f"\n  Bandwidth: {sfcw.bandwidth} Hz  Steps: {sfcw.num_steps}")
    print(f"  Range resolution: {sfcw.range_resolution*1000:.2f} mm")
    print(f"  Max unambiguous:  {sfcw.max_unambiguous_range*1000:.2f} mm")
    print(f"  Refresh rate:     {sfcw.refresh_rate:.2f} Hz\n")

    test_cases = [
        (np.array([1.0, 4.5]), 0.0, "facing right boundary"),
        (np.array([6.0, 1.0]), math.pi / 2, "facing up"),
        (np.array([6.0, 8.0]), -math.pi / 2, "facing down"),
        (np.array([1.0, 1.0]), math.pi / 4, "facing diagonal"),
    ]

    print(f"  {'Position':>12s}  {'Yaw':>6s}  {'True (m)':>10s}  {'SFCW (m)':>10s}  "
          f"{'Error (mm)':>10s}  {'Material':>10s}")
    print("  " + "-" * 75)

    all_true, all_meas = [], []
    for pos, yaw, desc in test_cases:
        results = sensor.sweep(pos, yaw, mode='throw', num_rays=5, max_range=10.0)
        center = results[2]
        true_d = center['true_distance']
        sfcw_d = center['sfcw_range']
        err_mm = abs(sfcw_d - true_d) * 1000
        mat = center['material_true'] or 'open'
        print(f"  {str(pos):>12s}  {yaw:6.2f}  {true_d:10.4f}  {sfcw_d:10.6f}  "
              f"{err_mm:10.4f}  {mat:>10s}  ({desc})")
        if center['material_true']:
            all_true.append(true_d)
            all_meas.append(sfcw_d)

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.scatter(all_true, all_meas, c='#1f77b4', s=60, zorder=3)
    lims = [0, max(max(all_true), max(all_meas)) + 1]
    ax.plot(lims, lims, '--', color='#2ca02c', alpha=0.5, label='Ideal')
    ax.set_xlabel('True distance (m)')
    ax.set_ylabel('SFCW measured distance (m)')
    ax.set_title('SFCW Phase Ranging Accuracy', fontweight='bold')
    ax.legend()
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect('equal')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = 'pics/demo_sfcw_accuracy.png'
    fig.savefig(out, dpi=200)
    plt.close()
    print(f"\n  Plot saved: {out}")


def test_material_classification():
    print("\n" + "=" * 60)
    print("  TEST 2: Material Classification")
    print("=" * 60)

    sfcw = SfcwWaveform()
    print(f"\n  {'True':>12s}  ->  {'Identified':>12s}  {'Conf':>6s}  {'Match':>6s}")
    print("  " + "-" * 50)

    for mat_name in ['concrete', 'drywall', 'rubble', 'metal', 'wood', 'glass']:
        mat = MaterialDb.get(mat_name)
        phases, amps, snr = sfcw.simulate_reflection(
            3.0, mat['freq_response'], mat['reflection_coeff'], noise_std=0.01
        )
        obs = amps / (amps.max() + 1e-10)
        identified, conf = MaterialDb.identify(obs)
        match = "✓" if identified == mat_name else "✗"
        print(f"  {mat_name:>12s}  ->  {identified:>12s}  {conf:6.3f}  {match:>6s}")


def demo_navigation():
    print("\n" + "=" * 60)
    print("  TEST 3: Navigation Demo (Proportional Controller)")
    print("=" * 60)

    env = EchosEnv(render_mode="human")
    obs, _ = env.reset()

    trajectory = [env.uav_pos.copy()]
    commands = [env.commander.state]
    snr_log = [env._cached_snr]

    for step in range(1500):
        action = proportional_controller(obs)
        obs, reward, term, trunc, info = env.step(action)
        trajectory.append(env.uav_pos.copy())
        commands.append(info['command'])
        snr_log.append(info['path_b_snr'])

        if step % 100 == 0:
            print(f"  Step {step:4d} | {info['command']:5s} | "
                  f"Beam {info['beam_mode']:5s} | "
                  f"SNR {info['path_b_snr']:5.1f} dB | "
                  f"Wall {info['min_dist']:.2f} m")

        if term or trunc:
            print(f"\n  Episode end: {info}")
            break

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    ax = axes[0]
    for wall in env.maze.walls:
        x, y = wall.exterior.xy
        mat = env.maze.wall_materials.get(id(wall), 'concrete')
        ax.fill(x, y, alpha=0.85, color=MAT_COLORS.get(mat, 'gray'), ec='#555', lw=0.6)

    traj = np.array(trajectory)
    cmd_colors = {'SCOUT': '#1f77b4', 'HUNT': '#ff7f0e', 'RTH': '#2ca02c'}
    for i in range(len(traj) - 1):
        c = cmd_colors.get(commands[i], '#1f77b4')
        ax.plot(traj[i:i+2, 0], traj[i:i+2, 1], color=c, lw=2.5, alpha=0.8)

    ax.scatter(traj[0, 0], traj[0, 1], c='#2ca02c', s=120, zorder=5, ec='white', lw=1.5, label='Start')
    ax.scatter(traj[-1, 0], traj[-1, 1], c='#9467bd', s=120, marker='s', zorder=5, ec='white', lw=1.5, label='End')
    ax.scatter(env.survivor_pos[0], env.survivor_pos[1], c='#d62728', s=200, marker='*', zorder=6, label='Survivor')
    ax.scatter(env.commander.home[0], env.commander.home[1], c='#2ca02c', s=80, marker='H', zorder=5, label='Home')

    for wp in env.maze.scout_waypoints:
        ax.scatter(wp[0], wp[1], c='#cccccc', s=30, marker='.', zorder=3)

    ax.set_xlim(-0.5, 12.5); ax.set_ylim(-0.5, 9.5)
    ax.set_aspect('equal')
    ax.set_title('Navigation Trajectory (Proportional Controller)', fontweight='bold')
    ax.legend(fontsize=8, loc='lower right')

    ax2 = axes[1]
    ax2.plot(snr_log, color='#d62728', lw=1.5)
    ax2.axhline(y=4.5, color='#ff7f0e', ls='--', alpha=0.7, label='HUNT threshold (4.5 dB)')
    ax2.axhline(y=14.0, color='#2ca02c', ls='--', alpha=0.7, label='RTH threshold (14 dB)')
    ax2.set_xlabel('Step'); ax2.set_ylabel('Path B SNR (dB)')
    ax2.set_title('Survivor Signal Gradient', fontweight='bold')
    ax2.legend(fontsize=9); ax2.grid(alpha=0.3)

    plt.tight_layout()
    out = 'pics/demo_navigation.png'
    fig.savefig(out, dpi=200)
    plt.close()
    print(f"\n  Plot saved: {out}")

    print("\n  SFCW Sensing Summary (last sweep):")
    for i, r in enumerate(env._cached_sfcw):
        err = abs(r['sfcw_range'] - r['true_distance']) * 1000
        print(f"    Ray {i}: true={r['true_distance']:.4f} m  "
              f"sfcw={r['sfcw_range']:.6f} m  "
              f"err={err:.3f} mm  "
              f"mat={r['material_true']}->{r['material_id']}")


if __name__ == "__main__":
    test_sfcw_accuracy()
    test_material_classification()
    demo_navigation()
