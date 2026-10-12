"""Portable, inference-only 1-D reference used by Round 10 analyses.

radial1d_reference.py is an unchanged vendor copy of the project's solver.
Original: outputs/email_data_group/radial1d_model/radial1d_model.py in the
calit2 wiki worktree. SHA256:
bc800d697e1998fa8d8b57514c4ae47bcd0ffc8f55e4d78f34a8a90a212bff8d
The parametrised solve below mirrors physics_cache_vs3/build_physics_cache.py.
No function in this module opens a data file; F11 supplies preloaded frames.
"""
import numpy as np

import radial1d_reference as reference

SOURCE_SHA256 = "bc800d697e1998fa8d8b57514c4ae47bcd0ffc8f55e4d78f34a8a90a212bff8d"


def simulate(boundary, time, k_scale=1.0, r_out=0.127, n_layers=60, sub_steps=2):
    """Input-driven finite-volume solve; k_scale multiplies k(T) only."""
    boundary = np.asarray(boundary, dtype=float)
    time = np.asarray(time, dtype=float)
    if boundary.ndim != 1 or boundary.shape != time.shape or len(time) < 2:
        raise ValueError("boundary/time must be equal-length 1-D arrays")
    if not np.isfinite(boundary).all() or not np.isfinite(time).all():
        raise ValueError("non-finite boundary/time")
    if np.any(np.diff(time) <= 0) or float(k_scale) <= 0:
        raise ValueError("time must increase and k_scale must be positive")
    n = int(n_layers)
    r = np.linspace(reference.R_IN, float(r_out), n + 1)
    h = r[1] - r[0]
    face = 0.5 * (r[:-1] + r[1:])
    volumes = r * h
    volumes[n] = 0.5 * h * (r[n] - h / 4)
    w_line = np.full(n + 1, h)
    w_line[0] *= 0.5
    w_line[-1] *= 0.5
    w_line /= w_line.sum()
    w_volume = r * h
    w_volume[0] *= 0.5
    w_volume[-1] *= 0.5
    w_volume /= w_volume.sum()
    temp = np.full(n + 1, boundary[0])
    outer = np.empty(len(time))
    line = np.empty(len(time))
    volume = np.empty(len(time))
    outer[0], line[0], volume[0] = temp[-1], temp @ w_line, temp @ w_volume
    for step in range(1, len(time)):
        dt = (time[step] - time[step - 1]) / sub_steps
        for sub in range(sub_steps):
            frac = (sub + 1) / sub_steps
            wall = boundary[step - 1] * (1 - frac) + boundary[step] * frac
            conductivity = reference.k_concrete(0.5 * (temp[:-1] + temp[1:]))
            if float(k_scale) != 1.0:
                conductivity = conductivity * float(k_scale)
            conductance = conductivity * face / h
            mass = reference.RHO * reference.cp_concrete(temp[1:]) * volumes[1:] / dt
            left = conductance
            right = np.append(conductance[1:], 0.0)
            lower = -left
            diagonal = mass + left + right
            upper = -right
            rhs = mass * temp[1:]
            rhs[0] += left[0] * wall
            lower[0] = 0.0
            solved = reference.solve_tridiagonal(lower.tolist(), diagonal.tolist(),
                                                 upper.tolist(), rhs.tolist())
            temp = np.concatenate([[wall], solved])
        outer[step], line[step], volume[step] = temp[-1], temp @ w_line, temp @ w_volume
    return {"time": time, "T_inner_phys": boundary.copy(), "T_outer_phys": outer,
            "T_avg_line_phys": line, "T_avg_volume_phys": volume, "r": r,
            "k_scale": float(k_scale), "r_o": float(r_out)}


def from_frame(frame, k_scale=1.0, params=None):
    """A preloaded frame -> physics arrays. Test ownership stays with caller."""
    params = params or {}
    return simulate(frame["Input Temperature (C)"].to_numpy(float),
                    frame["Time (s)"].to_numpy(float), k_scale=k_scale,
                    r_out=params.get("r_o", 0.127),
                    n_layers=params.get("n_layers", 60),
                    sub_steps=params.get("sub_steps", 2))


def fit_prefix(frame, truth, minutes, bounds=(0.5, 1.5), params=None):
    """Fit one diffusivity from the observed prefix; never use the tail.

    Golden-section search uses T_outer and T_avg_line equally. Truth is a
    synthetic 1-D-generated property-shifted trajectory, not FE evidence.
    """
    time = frame["Time (s)"].to_numpy(float)
    end = int(np.searchsorted(time, time[0] + 60 * float(minutes), side="right"))
    end = min(len(time), max(2, end))
    prefix = frame.iloc[:end]

    def objective(scale):
        pred = from_frame(prefix, scale, params)
        errors = [pred[c][1:] - np.asarray(truth[c])[1:end]
                  for c in ("T_outer_phys", "T_avg_line_phys")]
        return float(np.mean(np.concatenate(errors) ** 2))

    lo, hi = map(float, bounds)
    ratio = (np.sqrt(5.0) - 1.0) / 2.0
    a, b = hi - ratio * (hi - lo), lo + ratio * (hi - lo)
    fa, fb = objective(a), objective(b)
    for _ in range(30):
        if fa < fb:
            hi, b, fb = b, a, fa
            a = hi - ratio * (hi - lo)
            fa = objective(a)
        else:
            lo, a, fa = a, b, fb
            b = lo + ratio * (hi - lo)
            fb = objective(b)
    scale = (lo + hi) / 2
    return {"k_scale": float(scale), "prefix_rows": end,
            "minutes": float(minutes), "prefix_mse": objective(scale)}
