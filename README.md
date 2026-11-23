# EEG / BCI Game-Control Project

A university coursework project exploring an EEG-to-game-control pipeline in Python, using a Backyard Brains Heart & Brain SpikerBox. This is an experimental learning project, not a production or clinical BCI system.

## What the project explores

- Reading and parsing EEG samples from a serial device, with recording support.
- Loading EDF recordings and preprocessing signals, including filtering, resampling, epoching, and band-power feature extraction.
- Training and evaluating classifiers on labeled signal data, then mapping predicted classes to left/right game controls.
- A Kivy-based EEG trainer/settings interface.

## Where to look

- `EEG/EEG.py` — serial reading and EEG recording code.
- `EEG/Preprocesser.py` — signal preprocessing and feature preparation.
- `EEG_game.py` — classifier experimentation and game-control integration.
- `EEG/EEG_Trainer/` — trainer interface and configuration code.

## Running and limitations

There is no pinned dependency file or one-command setup. Running the hardware path requires compatible Python packages, the EEG device, a working serial-port configuration, and local settings. Some scripts contain machine-specific paths, so the project will need configuration for a new environment. The live hardware/game path and model scores should be treated as experimental rather than as validated performance claims.

If reproducing the experiments, use recordings you have permission to process and share; do not publish identifiable or sensitive participant data without appropriate consent.

## Skills demonstrated

Python application code, serial-device integration, EEG signal-processing workflows, exploratory classifier evaluation, and integration of model output with interactive game controls.
