# Echos

![Language](https://img.shields.io/badge/Language-Python-blue)
![Status](https://img.shields.io/badge/Status-Simulation%20Prototype-orange)

Echos is a 2D acoustic search-and-rescue drone simulation project.

The system models how a drone could search for survivors in an indoor or collapsed-building environment where GPS, cameras, and thermal sensors may fail. Instead of relying on vision, Echos uses active ultrasonic sensing and passive survivor-signal tracking.

## 🛠️ Why We Built This

This project was built for a final project.

The goal is to simulate an acoustic drone system that can:

- map obstacles using ultrasonic echoes
- estimate distance using SFCW phase-based ranging
- identify simulated material frequency signatures
- follow passive survivor-signal strength
- switch between search, hunt, and return-home mission modes

## 🧩 Features

- 📡 **SFCW Ranging** – Uses 40–50 kHz stepped-frequency continuous wave sensing for phase-based distance estimation.
- 🎯 **Dual Beam Modes** – Throw mode for narrow long-range sensing and Flood mode for wide proximity scanning.
- 🧱 **Material Signatures** – Simulates frequency responses for concrete, drywall, rubble, metal, and human tissue.
- 🧭 **Mission Commander** – Uses SCOUT, HUNT, and RTH states for search-and-rescue behavior.
- 🤖 **PPO Training Wrapper** – Includes curriculum learning stages for reinforcement learning experiments.
- 🗺️ **2D Maze Simulation** – Provides ray tracing, wall detection, and survivor search in a controlled environment.

## 📁 Project Structure

```text
Echos/
│
├── acoustic_physics.py   # SFCW, beam model, material database, 2D TDOA
├── sim.py                # Maze, ray tracing, commander, Gym environment
├── demo.py               # Verification and visualization demo
├── train.py              # Curriculum training wrapper
├── run_training.py       # PPO training script
├── echos.ipynb           # Notebook experiments
└── README.md
