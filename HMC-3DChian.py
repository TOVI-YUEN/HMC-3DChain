import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from tqdm import tqdm

# ================== Device & dimensionality ==================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

dim = 3

# ================== Backbone, sidechains, and ions ==================
N_backbone = 20
backbone_indices = list(range(N_backbone))

# Odd backbone beads have sidechains: 1,3,5,...,19
backbone_with_side = list(range(1, N_backbone, 2))
N_side = len(backbone_with_side)

# Number of ions
N_ions = 20

N_poly = N_backbone + N_side
N_total = N_poly + N_ions

print(f"N_backbone = {N_backbone}, N_side = {N_side}, N_ions = {N_ions}, N_total = {N_total}")

# Sidechain index mapping
parent_idx_side = []
side_indices = []
cur_side = N_backbone
for bb in backbone_with_side:
    parent_idx_side.append(bb)
    side_indices.append(cur_side)
    cur_side += 1
assert cur_side == N_poly

parent_idx_side = torch.tensor(parent_idx_side, dtype=torch.long, device=device)
side_indices = torch.tensor(side_indices, dtype=torch.long, device=device)

# Ion indices
ion_indices = torch.arange(N_poly, N_total, dtype=torch.long, device=device)

# ================== Physical parameters ==================
# Mass
m_poly = 1.0
m_ion = 0.5
m_vec = torch.full((N_total,), m_poly, device=device)
m_vec[ion_indices] = m_ion

# Weak trapping to keep everything in a finite region
k_trap_poly = 0.02    # polymer
k_trap_ion  = 0.05    # ions slightly stronger to avoid drifting too far

# Backbone covalent bonds
r0_bond = 0.5
k_bond = 400.0

# Sidechain covalent bonds
r0_side = 0.35
k_side_bond = 200.0

# Backbone bond angle ~110°
theta0_deg = 110.0
theta0 = theta0_deg * np.pi / 180.0
k_angle = 20.0       # multi-well cosine potential strength

# Backbone dihedral: multi-well cosine potential, n=3
phi0_deg = 180.0
phi0 = phi0_deg * np.pi / 180.0
k_dihedral = 1.0
n_dihedral = 3

# Charges:
# - backbone: negative
# - sidechains: opposite sign (positive)
# - ions: positive
q_backbone = -1.0
q_side = -q_backbone   # = +1.0, opposite to backbone
q_ion = +1.0

charge = torch.zeros(N_total, device=device)
charge[:N_backbone] = q_backbone
charge[N_backbone:N_poly] = q_side
charge[ion_indices] = q_ion

# Debye–Hückel type screened Coulomb
k_c = 0.1            # effective strength
kappa = 0.5          # screening parameter, larger = shorter screening length
soft_eps = 1e-2

# Sidechain–sidechain "H-bond" multi-well potential
r0_hb = 1.0
k_hb = 4.0
r_hb_cut = 1.6
w_hb = 0.4           # controls cosine period in distance

# Hard-core radii for all nonbonded pairs
r_main = 0.4
r_side = 0.3
r_ion  = 0.25
radii = torch.full((N_total,), r_main, device=device)
radii[N_backbone:N_poly] = r_side
radii[ion_indices] = r_ion

r_coll = 0.45
k_coll = 15.0

# ================== HMC parameters ==================
beta = 2.0           # inverse temperature
eps = 0.004          # leapfrog stepsize
L = 10               # number of leapfrog steps per trajectory
n_trajectories = 400

# ================== Bond matrix (to exclude bonded pairs from nonbonded terms) ==================
bond_mat = np.zeros((N_total, N_total), dtype=bool)

# Backbone bonds
for i in range(N_backbone - 1):
    bond_mat[i, i + 1] = bond_mat[i + 1, i] = True

# Sidechain bonds
for p, s in zip(parent_idx_side.cpu().numpy(), side_indices.cpu().numpy()):
    bond_mat[p, s] = bond_mat[s, p] = True

bond_matrix = torch.tensor(bond_mat, dtype=torch.bool, device=device)

# ================== Initial configuration ==================
q0 = np.zeros((N_total, dim), dtype=np.float32)

# --- Backbone: zig-zag chain in the xy-plane with given bond length and bond angle ---
q0_bb_2d = np.zeros((N_backbone, 2), dtype=np.float32)
q0_bb_2d[0] = np.array([0.0, 0.0], dtype=np.float32)

b_prev = np.array([r0_bond, 0.0], dtype=np.float32)
phi_internal = np.pi - theta0
cos_phi = np.cos(phi_internal)
sin_phi = np.sin(phi_internal)
R_plus = np.array([[cos_phi, -sin_phi],
                   [sin_phi,  cos_phi]], dtype=np.float32)
R_minus = np.array([[cos_phi,  sin_phi],
                    [-sin_phi, cos_phi]], dtype=np.float32)

for i in range(1, N_backbone):
    if i % 2 == 1:
        b = R_plus @ b_prev
    else:
        b = R_minus @ b_prev
    q0_bb_2d[i] = q0_bb_2d[i - 1] + b
    b_prev = b

center_2d = q0_bb_2d.mean(axis=0)
q0_bb_2d -= center_2d

q0[:N_backbone, :2] = q0_bb_2d

# --- Sidechains: attach along ±z on odd backbone beads ---
for k, (p, s) in enumerate(zip(parent_idx_side.cpu().numpy(),
                               side_indices.cpu().numpy())):
    sign = 1.0 if (k % 2 == 0) else -1.0
    offset = np.array([0.0, 0.0, sign * r0_side], dtype=np.float32)
    q0[s] = q0[p] + offset

# --- Ions: randomly placed around the origin, avoiding overlap with polymer ---
for k, idx in enumerate(ion_indices.cpu().numpy()):
    placed = False
    for _ in range(2000):
        xyz = np.random.normal(scale=2.0, size=3).astype(np.float32)
        d2 = np.sum((q0[:N_poly] - xyz) ** 2, axis=1)
        if np.min(d2) > (r_main + r_ion + 0.2) ** 2:
            q0[idx] = xyz
            placed = True
            break
    if not placed:
        q0[idx] = np.random.normal(scale=3.0, size=3).astype(np.float32)

# Tiny random noise to break perfect symmetry
q0 += 1e-3 * np.random.randn(N_total, dim).astype(np.float32)

q = torch.tensor(q0, dtype=torch.float32, device=device)

# ================== Energy terms ==================

def bond_energy_backbone(q: torch.Tensor):
    """Covalent bonds along the backbone."""
    b = q[1:N_backbone] - q[0:N_backbone - 1]
    b2 = torch.sum(b * b, dim=1)
    b_len = torch.sqrt(b2 + 1e-12)
    dr = b_len - r0_bond
    return 0.5 * k_bond * torch.sum(dr * dr)


def bond_energy_side(q: torch.Tensor):
    """Covalent bonds between backbone and sidechain beads."""
    if N_side == 0:
        return torch.tensor(0.0, device=q.device)
    p_pos = q[parent_idx_side]
    s_pos = q[side_indices]
    diff = s_pos - p_pos
    dist = torch.sqrt(torch.sum(diff * diff, dim=1) + 1e-12)
    dr = dist - r0_side
    return 0.5 * k_side_bond * torch.sum(dr * dr)


def angle_energy(q: torch.Tensor):
    """
    Backbone bond angle potential:
        U_angle = k_angle * Σ [ 1 - cos(θ - θ0) ]
    This is a multi-well cosine form around the target angle θ0.
    """
    if N_backbone < 3:
        return torch.tensor(0.0, device=q.device)
    eps_ = 1e-8
    U = torch.tensor(0.0, device=q.device)
    for i in range(1, N_backbone - 1):
        v1 = q[i - 1] - q[i]
        v2 = q[i + 1] - q[i]
        n1 = torch.linalg.norm(v1) + eps_
        n2 = torch.linalg.norm(v2) + eps_
        cos_t = torch.dot(v1, v2) / (n1 * n2)
        cos_t = torch.clamp(cos_t, -1.0 + 1e-6, 1.0 - 1e-6)
        theta = torch.acos(cos_t)
        dtheta = theta - theta0
        U = U + k_angle * (1.0 - torch.cos(dtheta))
    return U


def dihedral_energy(q: torch.Tensor):
    """
    Backbone dihedral potential:
        U_dih = k_dihedral * Σ [ 1 - cos(n_dihedral * (φ - φ0)) ]
    A standard multi-well torsion potential.
    """
    if N_backbone < 4:
        return torch.tensor(0.0, device=q.device)
    eps_ = 1e-8
    U = torch.tensor(0.0, device=q.device)
    for i in range(1, N_backbone - 2):
        p0 = q[i - 1]
        p1 = q[i]
        p2 = q[i + 1]
        p3 = q[i + 2]

        b1 = p1 - p0
        b2 = p2 - p1
        b3 = p3 - p2

        n1 = torch.cross(b1, b2, dim=0)
        n2 = torch.cross(b2, b3, dim=0)

        b2_norm = torch.linalg.norm(b2) + eps_
        b2u = b2 / b2_norm

        n1_norm = torch.linalg.norm(n1) + eps_
        n2_norm = torch.linalg.norm(n2) + eps_
        n1u = n1 / n1_norm
        n2u = n2 / n2_norm

        m1 = torch.cross(n1u, b2u, dim=0)

        x = torch.dot(n1u, n2u)
        y = torch.dot(m1, n2u)

        phi = torch.atan2(y, x)
        dphi = phi - phi0
        U = U + k_dihedral * (1.0 - torch.cos(n_dihedral * dphi))
    return U


def nonbond_terms(q: torch.Tensor):
    """
    Non-bonded interactions:
      - Screened Debye–Hückel Coulomb between all non-bonded charged pairs
      - Multi-well "H-bond" potential between sidechain–sidechain pairs
      - Short-range soft-core repulsion for all non-bonded pairs
    """
    Ntot = N_total
    rij = q[:, None, :] - q[None, :, :]
    r2 = torch.sum(rij * rij, dim=-1)

    eye = torch.eye(Ntot, device=q.device, dtype=torch.bool)
    r2 = r2 + eye * 1.0
    r = torch.sqrt(r2)

    idx = torch.arange(Ntot, device=q.device)
    idx_i = idx.unsqueeze(1)
    idx_j = idx.unsqueeze(0)

    not_self = ~eye
    nonbond = not_self & (~bond_matrix)

    # Debye–Hückel screened Coulomb
    r_soft2 = r2 + soft_eps**2
    r_soft = torch.sqrt(r_soft2)

    qi = charge.unsqueeze(1)
    qj = charge.unsqueeze(0)
    qq = qi * qj

    coul_pref = k_c * qq * torch.exp(-kappa * r_soft) / r_soft
    coul_mask = nonbond & (qq != 0.0)
    if coul_mask.any():
        U_c = 0.5 * torch.sum(coul_pref[coul_mask])
    else:
        U_c = torch.tensor(0.0, device=q.device)

    # Sidechain–sidechain multi-well "H-bond"–like potential
    side_mask = (idx_i >= N_backbone) & (idx_i < N_poly) & \
                (idx_j >= N_backbone) & (idx_j < N_poly)
    hb_pair_mask = nonbond & side_mask
    U_hb = torch.tensor(0.0, device=q.device)
    if hb_pair_mask.any():
        r_hb = r[hb_pair_mask]
        mask_in = r_hb < r_hb_cut
        if mask_in.any():
            x = np.pi * (r_hb[mask_in] - r0_hb) / w_hb
            U_hb = k_hb * torch.sum(1.0 - torch.cos(x))

    # Soft-core repulsion for all non-bonded pairs (simple quadratic wall)
    coll_mask = (r < r_coll) & nonbond
    dr_coll = torch.zeros_like(r)
    dr_coll[coll_mask] = r_coll - r[coll_mask]
    U_coll = 0.5 * k_coll * torch.sum(dr_coll[coll_mask] ** 2)

    return U_c, U_hb, U_coll


def U_components(q: torch.Tensor):
    """Return total energy and all components for logging."""
    U_trap_poly = 0.5 * k_trap_poly * torch.sum(q[:N_poly] * q[:N_poly])
    U_trap_ion  = 0.5 * k_trap_ion  * torch.sum(q[ion_indices] * q[ion_indices])
    U_trap = U_trap_poly + U_trap_ion

    U_bond_bb   = bond_energy_backbone(q)
    U_bond_side = bond_energy_side(q)
    U_ang = angle_energy(q)
    U_dih = dihedral_energy(q)
    U_c, U_hb, U_coll = nonbond_terms(q)

    U_tot = U_trap + U_bond_bb + U_bond_side + U_ang + U_dih + U_c + U_hb + U_coll
    return U_tot, U_trap, U_bond_bb, U_bond_side, U_ang, U_dih, U_c, U_hb, U_coll


def U_torch(q: torch.Tensor) -> torch.Tensor:
    U_tot, *_ = U_components(q)
    return U_tot


def grad_U_torch(q: torch.Tensor) -> torch.Tensor:
    """Compute ∇U(q) using autograd."""
    with torch.enable_grad():
        q_req = q.detach().clone().requires_grad_(True)
        U = U_torch(q_req)
        grad_q, = torch.autograd.grad(U, q_req, create_graph=False)
    return grad_q.to(q.device)


def K_torch(p: torch.Tensor) -> torch.Tensor:
    """Kinetic energy with per-particle mass."""
    return torch.sum(p * p / (2.0 * m_vec.unsqueeze(1)))

# ================== Leapfrog & HMC ==================

def leapfrog_torch(q, p, eps, L):
    """Standard leapfrog integrator for multi-particle HMC."""
    q = q.clone()
    p = p.clone()
    q_path = []

    # Half-step momentum
    p = p - 0.5 * eps * grad_U_torch(q)

    for step in range(L):
        # Full-step position
        q = q + eps * (p / m_vec.unsqueeze(1))
        q_path.append(q.clone())
        if step != L - 1:
            p = p - eps * grad_U_torch(q)

    # Final half-step momentum
    p = p - 0.5 * eps * grad_U_torch(q)
    # Momentum flip for reversibility
    p = -p

    q_path = torch.stack(q_path, dim=0)
    return q, p, q_path


frames_gpu = []
accept_count = 0

# Per-particle momentum std: p_i ~ N(0, m_i / beta)
sigma_p = torch.sqrt(m_vec / beta).to(device)

print("\nStart HMC (3D polymer + sidechains + ions, multi-well + opposite sidechain charge)...")
for t in tqdm(range(n_trajectories), desc="HMC sampling (3D+ions)"):
    p0 = torch.randn((N_total, dim), device=device) * sigma_p.unsqueeze(1)

    q_old = q.clone()
    p_old = p0.clone()
    H_old = U_torch(q_old) + K_torch(p_old)

    q_new, p_new, q_path = leapfrog_torch(q_old, p0, eps, L)
    H_new = U_torch(q_new) + K_torch(p_new)

    dH = H_new - H_old
    acc_prob = torch.exp(-dH).clamp(max=1.0)

    if torch.rand(1, device=device) < acc_prob:
        # Accept
        q = q_new
        accept_count += 1
        frames_gpu.append(q_path)
    else:
        # Reject: stay at q_old and do not record this trajectory (no visual "stall")
        q = q_old

    # Energy logging
    if (t + 1) % 50 == 0:
        U_vals = U_components(q)
        U_tot, U_trap, U_bond_bb, U_bond_side, U_ang, U_dih, U_c, U_hb, U_coll = U_vals
        print(
            f"[step {t+1}/{n_trajectories}] "
            f"U_tot={float(U_tot):.2f}, U_trap={float(U_trap):.2f}, "
            f"U_bond={float(U_bond_bb+U_bond_side):.2f}, "
            f"U_ang={float(U_ang):.2f}, U_dih={float(U_dih):.2f}, "
            f"U_c={float(U_c):.2f}, U_hb={float(U_hb):.2f}, "
            f"U_coll={float(U_coll):.2f}, acc={accept_count/(t+1):.3f}"
        )

# Concatenate only accepted trajectories for animation
if len(frames_gpu) > 0:
    frames_torch = torch.cat(frames_gpu, dim=0)
else:
    # Edge case: if nothing is accepted, repeat the last configuration
    frames_torch = q.unsqueeze(0).expand(10, -1, -1).clone()

frames = frames_torch.cpu().numpy()
n_frames = frames.shape[0]
print("Total frames (accepted-only):", n_frames)
print("Acceptance rate:", accept_count / n_trajectories)

# ================== 3D animation ==================
fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection='3d')

# Estimate spatial scale using backbone length
L_chain = max(10.0, (N_backbone - 1) * r0_bond * 1.5)
ax.set_xlim(-L_chain / 2, L_chain / 2)
ax.set_ylim(-L_chain / 2, L_chain / 2)
ax.set_zlim(-L_chain / 2, L_chain / 2)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('Charged Polymer with Sidechains and Ions (3D HMC)')

# Separate scatters for backbone, sidechains, and ions
backbone_scat = ax.scatter([], [], [], s=30, c='tab:blue', label='backbone')
sidechain_scat = ax.scatter([], [], [], s=20, c='tab:orange', label='sidechains')
ion_scat = ax.scatter([], [], [], s=10, c='tab:green', label='ions')

# Line connecting only backbone beads
chain_line, = ax.plot([], [], [], lw=1.0, alpha=0.8, color='tab:blue')

# Trajectory of monomer 0
trail_line, = ax.plot([], [], [], lw=0.8, alpha=0.4, color='k')

history_x = []
history_y = []
history_z = []

ax.legend(loc='upper right')


def init():
    backbone_scat._offsets3d = ([], [], [])
    sidechain_scat._offsets3d = ([], [], [])
    ion_scat._offsets3d = ([], [], [])
    chain_line.set_data([], [])
    chain_line.set_3d_properties([])
    trail_line.set_data([], [])
    trail_line.set_3d_properties([])
    return backbone_scat, sidechain_scat, ion_scat, chain_line, trail_line


def update(frame_idx):
    pos = frames[frame_idx]  # (N_total, 3)

    # Split positions
    pos_backbone = pos[:N_backbone]
    pos_side     = pos[N_backbone:N_poly]
    pos_ions     = pos[N_poly:]

    xb, yb, zb = pos_backbone[:, 0], pos_backbone[:, 1], pos_backbone[:, 2]
    xs, ys, zs = pos_side[:, 0], pos_side[:, 1], pos_side[:, 2]
    xi, yi, zi = pos_ions[:, 0], pos_ions[:, 1], pos_ions[:, 2]

    # Update scatters
    backbone_scat._offsets3d = (xb, yb, zb)
    sidechain_scat._offsets3d = (xs, ys, zs)
    ion_scat._offsets3d = (xi, yi, zi)

    # Backbone chain line (connect only backbone beads)
    chain_line.set_data(xb, yb)
    chain_line.set_3d_properties(zb)

    # Trajectory of backbone bead 0
    history_x.append(xb[0])
    history_y.append(yb[0])
    history_z.append(zb[0])
    trail_line.set_data(history_x, history_y)
    trail_line.set_3d_properties(history_z)

    return backbone_scat, sidechain_scat, ion_scat, chain_line, trail_line


ani = animation.FuncAnimation(
    fig,
    update,
    frames=n_frames,
    init_func=init,
    interval=20,      # smaller -> faster playback
    blit=False        # IMPORTANT: blit must be False for 3D
)

plt.tight_layout()
plt.show()

# To save as mp4 (requires local ffmpeg):
# ani.save("charged_polymer_sidechains_ions_3d_hmc.mp4", fps=30, dpi=150)

