# HMC-Polymer: 3D Polymer Chain Simulation with Sidechains and Ions

A GPU-accelerated molecular simulation demo implementing **Hybrid Monte Carlo (HMC)** for a coarse-grained **3D polymer chain** with:

- a flexible **backbone**
- **sidechains** attached to odd-index backbone beads
- **electrostatic interactions** (Debye–Hückel)
- **multi-well angle and dihedral potentials**
- **sidechain–sidechain hydrogen-bond–like multi-well attraction**
- mobile **counter-ions** with opposite charge

This example illustrates how realistic biomolecular-like folding can emerge from a small set of physically inspired interaction terms.

All simulation components (energies, gradients, integrators, constraints) are written in **PyTorch**, enabling automatic differentiation and seamless GPU acceleration.

---

## ✨ Features

- **Full HMC integrator** with leapfrog dynamics.
- **Autograd-based forces** (no manual gradient coding).
- **3D polymer backbone** with bonded, angular, and dihedral terms.
- **Sidechains** attached to odd-index backbone atoms.
- **Screened Coulomb electrostatics** via Debye–Hückel potential.
- **Sidechain–sidechain multi-well hydrogen-bond potential**.
- **Excluded-volume (soft collision)** for all non-bonded pairs.
- **Ions** that interact with all chain atoms.
- **High acceptance rate** (typically ~0.85–0.90).
- **Real-time 3D animation** using Matplotlib.

---

## 🔧 Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourname/HMC-Polymer.git
cd HMC-Polymer
```

### 2. Install dependencies

It is recommended to use **conda**:

```bash
conda create -n hmc-polymer python=3.10 -y
conda activate hmc-polymer
```

Install required packages:
```bash
pip install torch numpy matplotlib tqdm
```

(Optional) Install a **CUDA-enabled PyTorch** build for large speedups:
```bash
# Example for CUDA 12.1, change according to your system
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

> ⚠️ Note: You must activate your conda environment before running the scripts.

The project uses the **Apache-2.0 License**.

---

## ▶️ Running the 3D Example

Run the main simulation:
```bash
python HMC-3DChain.py
```

Example terminal output:
```
Using device: cuda
N_backbone = 20, N_side = 10, N_ions = 20, N_total = 50

Start HMC (3D polymer + sidechains + ions, multi-well + opposite sidechain charge)...
HMC sampling (3D+ions): 100%|███████████████████████████████████████████████| 400/400 [00:49<00:00,  8.01it/s]
Total frames (accepted-only): 3550
Acceptance rate: 0.8875
```

After sampling completes, a real-time **3D animation** will appear, showing:

- Backbone atoms connected sequentially
- Sidechains attached in ±z directions
- Ions moving dynamically
- Folding driven by bonded and non-bonded forces
- Trail visualization of the first backbone atom

---

## 📁 File Structure
```
HMC-Polymer/
│
├── HMC-3DChain.py        # Main simulation code
├── LICENSE               # Apache-2.0
└── README.md             # This document
```

---

## 📘 Interaction Model Summary

| Term | Description |
|------|-------------|
| **Bond stretching** | Harmonic; backbone + sidechain bonds |
| **Angle potential** | Multi-well (1 − cos(θ − θ₀)) |
| **Dihedral potential** | Multi-well n-fold torsion |
| **Coulomb** | Debye–Hückel screened electrostatics |
| **Hydrogen-bond (sidechains)** | Multi-well radial attraction |
| **Excluded volume** | Soft collision penalty |
| **HMC integrator** | Leapfrog + Metropolis acceptance |

---

## 📊 Visualization
The animation uses `Matplotlib`'s 3D engine. Because 3D does not support blitting reliably, the animation uses:

```python
blit=False
interval=20  # ms, adjust for speed
```

You can tune playback speed by lowering `interval`.

---

## 📝 License
This project is licensed under the **Apache License 2.0**.

See the `LICENSE` file for details.

---

## 🤝 Contributing
Contributions are welcome! If you wish to extend the model, add new potentials, or improve performance, feel free to open an issue or pull request.

---

## 📬 Contact
If you have questions or need support, feel free to reach out via GitHub Issues.

Enjoy exploring polymer folding with HMC! 🎉
