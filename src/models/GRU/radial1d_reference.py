#!/usr/bin/env python3
"""
radial1d_model.py - 1-D radial heat-conduction model of the concrete ring around the pipe.

The concrete is a ring (annulus) between the pipe wall (r = 1 in) and the outer surface (r = 5 in).
The inner wall temperature is given as a function of time; the outer surface is insulated.
The script solves the radial heat equation with the temperature-dependent concrete properties of
the FE group's report and returns the outer-surface temperature, the radial (line) average and the
volume average of the concrete, and, if asked, the full radial temperature profile.

USAGE
    python radial1d_model.py FILE.csv [--drive inner|input] [--layers 60] [--out FILE_model.csv]
                             [--plot] [--profile-times 0,3600,7200,14400]

    FILE.csv         an FE temperature-input csv: Time (s), T_outer (C), T_inner (C), T_avg (C),
                     Input Temperature (C). T_outer and T_avg are optional (only used for comparison).
    --drive inner    (default) inner wall = the file's T_inner column (the FE result itself).
    --drive input    inner wall = the file's Input Temperature column (known before any simulation).
    --layers N       number of equal radial layers (default 60).
    --out FILE       output csv (default: <name of FILE.csv>_model.csv in the current folder).
    --plot           also save a PNG of file vs model next to the output csv (needs matplotlib).
    --profile-times  times in seconds; also write <out>_profile.csv with the temperature at every
                     grid point at those times (the nearest time row of the file is used).

    As a library:
        from radial1d_model import simulate
        res = simulate(T_inner_series, time_series)      # numpy arrays, same length
        res["T_outer"], res["T_avg_line"], res["T_avg_volume"]

Requires Python 3.8+, numpy and pandas (matplotlib only for --plot).
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

# ------------------------------------------------------------------------------------------------
# Geometry and concrete properties: FE Group Final Report Fall 2025, section 2.1
# (geometry sketch and "Concrete Properties" tables). The report gives no source for the values.
# ------------------------------------------------------------------------------------------------
R_IN = 0.0254        # m, inner radius of the concrete = outer radius of the steel pipe ("R 1.00 in")
R_OUT = 0.127        # m, outer radius of the concrete ("R 5.00 in")
RHO = 2400.0         # kg/m3, concrete density ("Density: 2400 kg/m3")

# Conductivity and specific heat: "linearly interpolated between temperatures and constant beyond
# defined values" (section 2.1).
K_TEMPS = [100.0, 400.0]               # C
K_VALUES = [2.5, 2.0]                  # W/(m K)  thermal conductivity at those temperatures
CP_TEMPS = [100.0, 200.0, 400.0]       # C
CP_VALUES = [900.0, 1000.0, 1100.0]    # J/(kg K) specific heat at those temperatures

# ------------------------------------------------------------------------------------------------
# Numerical settings (our choice, not from the report)
# ------------------------------------------------------------------------------------------------
N_LAYERS = 60        # equal radial layers -> N_LAYERS + 1 grid points from R_IN to R_OUT
SUB_STEPS = 2        # implicit time steps per data time step

# Column names as written in the FE group's csv files
COL_TIME = "Time (s)"
COL_OUTER = "T_outer (C)"
COL_INNER = "T_inner (C)"
COL_AVG = "T_avg (C)"
COL_INPUT = "Input Temperature (C)"
DRIVE_COLUMN = {"inner": COL_INNER, "input": COL_INPUT}


def k_concrete(T):
    """Thermal conductivity W/(m K) at temperature T (C)."""
    return np.interp(T, K_TEMPS, K_VALUES)


def cp_concrete(T):
    """Specific heat J/(kg K) at temperature T (C)."""
    return np.interp(T, CP_TEMPS, CP_VALUES)


def solve_tridiagonal(a, b, c, d):
    """Thomas algorithm for  a[i] x[i-1] + b[i] x[i] + c[i] x[i+1] = d[i]  (a[0], c[-1] not used)."""
    n = len(d)
    c2 = [0.0] * n
    d2 = [0.0] * n
    c2[0] = c[0] / b[0]
    d2[0] = d[0] / b[0]
    for i in range(1, n):                     # forward elimination
        den = b[i] - a[i] * c2[i - 1]
        c2[i] = c[i] / den
        d2[i] = (d[i] - a[i] * d2[i - 1]) / den
    x = [0.0] * n
    x[n - 1] = d2[n - 1]
    for i in range(n - 2, -1, -1):            # back substitution
        x[i] = d2[i] - c2[i] * x[i + 1]
    return x


def simulate(T_boundary, time, n_layers=N_LAYERS, sub_steps=SUB_STEPS, keep_profile=False):
    """Temperature in the concrete ring for a given inner-wall temperature history.

    Governing equation (axisymmetric conduction, no heat source), R_IN < r < R_OUT:
        rho * cp(T) * dT/dt = (1/r) * d/dr ( r * k(T) * dT/dr )
    Boundary and initial conditions:
        T(R_IN, t)     = T_boundary(t)    inner wall given, linear in time inside each data step
        dT/dr(R_OUT,t) = 0                outer surface insulated
        T(r, 0)        = T_boundary(0)    uniform start
    Discretisation (finite volume, implicit in time):
        N = n_layers equal layers of width h give grid points r_j = R_IN + j h, j = 0..N.
        Point j stands for the ring from r_j - h/2 to r_j + h/2 (half a ring at r_N); per radian and
        unit height its volume is V_j = r_j h (V_N = h/2 (r_N - h/4)). The heat flow through the face
        between points j and j+1 (at r_f = r_j + h/2) is G_f (T_j - T_(j+1)), with the conductance
        G_f = k(T_f) r_f / h and T_f the mean of T_j and T_(j+1).
        Each sub-step dt (= data time step / sub_steps) solves, for j = 1..N,
            rho cp(T_j_old) V_j (T_j - T_j_old) / dt = G_left (T_(j-1) - T_j) + G_right (T_(j+1) - T_j)
        with the temperatures on the right at the new time (backward Euler), k and cp evaluated at the
        old temperatures, T_0 = inner-wall value and G_right = 0 at r_N (insulated). This is a
        tridiagonal system, solved with the Thomas algorithm.
    Averages over the concrete (trapezoid rule over the grid points):
        line average   = integral of T dr   / (R_OUT - R_IN)      (every layer counts the same)
        volume average = integral of T r dr / integral of r dr    (weight r, i.e. by ring area)

    Parameters
        T_boundary : inner-wall temperature (C) at each time, 1-D array.
        time       : times (s), same length; or one number = constant time step.
        n_layers, sub_steps : see N_LAYERS, SUB_STEPS.
        keep_profile : also return the temperature at every grid point and time.
    Returns a dict of numpy arrays:
        "T_outer"      temperature at the outer surface (r = R_OUT)
        "T_avg_line"   line average of the concrete (see above)
        "T_avg_volume" volume average of the concrete (see above)
        "r"            radius of the grid points (m)
        "profile"      (number of times, N+1) temperatures, or None if keep_profile is False
    """
    T_b = np.asarray(T_boundary, dtype=float)
    n_t = len(T_b)
    if np.ndim(time) == 0:
        t = np.arange(n_t) * float(time)
    else:
        t = np.asarray(time, dtype=float)
    N = int(n_layers)

    r = np.linspace(R_IN, R_OUT, N + 1)          # grid points; r[0] = inner wall, r[N] = outer surface
    h = r[1] - r[0]
    r_face = 0.5 * (r[:-1] + r[1:])              # r_face[j] lies between points j and j+1
    # Volumes and conductances are both counted per radian of angle and per metre of height. The real
    # wedge angle (15 degrees) and height multiply every term of the heat balance equally and cancel,
    # so they do not appear anywhere in the model.
    V = r * h                                    # ring volume per radian and unit height: integral of r dr over r_j +- h/2
    V[N] = 0.5 * h * (r[N] - h / 4)              # half ring at the outer surface: integral of r dr from r_N - h/2 to r_N
                                                 # = (r_N^2 - (r_N - h/2)^2) / 2 = (h/2) (r_N - h/4)

    w_line = np.full(N + 1, h)                   # averaging weights (trapezoid rule over the points)
    w_line[0] *= 0.5
    w_line[-1] *= 0.5
    w_line = w_line / w_line.sum()
    w_vol = r * h
    w_vol[0] *= 0.5
    w_vol[-1] *= 0.5
    w_vol = w_vol / w_vol.sum()

    T = np.full(N + 1, T_b[0])                   # uniform start
    T_outer = np.empty(n_t)
    T_line = np.empty(n_t)
    T_vol = np.empty(n_t)
    profile = np.empty((n_t, N + 1)) if keep_profile else None

    def record(n, T):
        T_outer[n] = T[-1]
        T_line[n] = T @ w_line
        T_vol[n] = T @ w_vol
        if keep_profile:
            profile[n] = T

    record(0, T)
    for n in range(1, n_t):
        dt = (t[n] - t[n - 1]) / sub_steps
        for s in range(sub_steps):
            frac = (s + 1) / sub_steps
            T_wall = T_b[n - 1] * (1 - frac) + T_b[n] * frac       # boundary, linear inside the step

            k_face = k_concrete(0.5 * (T[:-1] + T[1:]))            # k at the N faces
            G = k_face * r_face / h                                # face conductances (per radian)
            M = RHO * cp_concrete(T[1:]) * V[1:] / dt              # heat capacity / dt, points 1..N
            G_left = G                                             # face between j-1 and j
            G_right = np.append(G[1:], 0.0)                        # face between j and j+1 (0 = insulated)

            lower = -G_left
            diag = M + G_left + G_right
            upper = -G_right
            rhs = M * T[1:]
            rhs[0] += G_left[0] * T_wall                           # known inner-wall temperature
            lower[0] = 0.0

            T_new = solve_tridiagonal(lower.tolist(), diag.tolist(), upper.tolist(), rhs.tolist())
            T = np.concatenate([[T_wall], T_new])
        record(n, T)

    return {"T_outer": T_outer, "T_avg_line": T_line, "T_avg_volume": T_vol, "r": r, "profile": profile}


# ================================================================================================
# Command line, used to create graph
# ================================================================================================
def read_csv(path, drive_col):
    """Read an FE csv; returns (time, boundary, file T_outer or None, file T_avg or None)."""
    try:
        df = pd.read_csv(path)
    except (IOError, OSError, ValueError) as e:   # missing file, no permission, empty or broken csv
        sys.exit("error: cannot read %s: %s" % (path, e))
    names = {str(c).strip().lower(): c for c in df.columns}

    def column(name, aliases=()):
        for n in (name,) + tuple(aliases):
            if n.lower() in names:
                return pd.to_numeric(df[names[n.lower()]], errors="coerce").to_numpy(float)
        return None

    t = column(COL_TIME)
    T_b = column(drive_col)
    if t is None or T_b is None:
        sys.exit("error: %s needs the columns '%s' and '%s'; found %s"
                 % (path, COL_TIME, drive_col, [str(c).strip() for c in df.columns]))
    if np.isnan(t).any() or np.isnan(T_b).any():
        sys.exit("error: empty or non-numeric values in '%s' or '%s'" % (COL_TIME, drive_col))
    if len(t) < 2 or np.any(np.diff(t) <= 0):
        sys.exit("error: '%s' must have at least 2 rows and increase strictly" % COL_TIME)
    return t, T_b, column(COL_OUTER), column(COL_AVG, ("T_ave (C)",))


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae(a, b):
    return float(np.mean(np.abs(a - b)))


def profile_rows(text, t):
    """--profile-times "0,3600,..." -> row numbers of the nearest times in the file."""
    try:
        times = [float(x) for x in text.split(",") if x.strip()]
    except ValueError:
        sys.exit("error: --profile-times must be numbers separated by commas, e.g. 0,3600,7200")
    rows = []
    for want in times:
        if want < t[0] - 1e-9 or want > t[-1] + 1e-9:
            sys.exit("error: profile time %g s is outside the file (%g to %g s)" % (want, t[0], t[-1]))
        rows.append(int(np.argmin(np.abs(t - want))))
    return rows


def write_profile(path, t, res, rows):
    out = pd.DataFrame({"Point": np.arange(len(res["r"])), "r (m)": res["r"], "r (in)": res["r"] / 0.0254})
    for i in rows:
        out["T (C) at t=%g s" % t[i]] = res["profile"][i]
    out.to_csv(path, index=False, float_format="%.6g")
    print("profile at t = %s s -> %s" % (", ".join("%g" % t[i] for i in rows), path))


def plot(png_path, t, T_b, drive_col, To_file, Ta_file, res, prof_rows, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hours = t / 3600.0
    has_file = To_file is not None or Ta_file is not None
    n_panels = 1 + int(has_file) + int(bool(prof_rows))
    fig, ax = plt.subplots(n_panels, 1, figsize=(9, 3.2 * n_panels + 1.0), squeeze=False)
    ax = list(ax[:, 0])

    a = ax.pop(0)                                   # 1) temperatures against time
    a.plot(hours, T_b, color="0.55", lw=1.2, label="inner wall used = %s" % drive_col)
    if To_file is not None:
        a.plot(hours, To_file, color="tab:red", lw=2.4, alpha=0.45, label="file T_outer")
    a.plot(hours, res["T_outer"], color="tab:red", lw=1.0, ls="--", label="model T_outer")
    if Ta_file is not None:
        a.plot(hours, Ta_file, color="tab:blue", lw=2.4, alpha=0.45, label="file T_avg")
    a.plot(hours, res["T_avg_line"], color="tab:blue", lw=1.0, ls="--", label="model T_avg (line)")
    a.plot(hours, res["T_avg_volume"], color="tab:green", lw=1.0, ls=":", label="model T_avg (volume)")
    a.set_xlabel("time (h)")
    a.set_ylabel("temperature (C)")
    a.set_title(title, fontsize=10)
    a.legend(fontsize=8, ncol=2)
    a.grid(alpha=0.3)

    if has_file:                                    # 2) model minus file
        a = ax.pop(0)
        a.axhline(0.0, color="0.6", lw=0.8)
        if To_file is not None:
            a.plot(hours, res["T_outer"] - To_file, color="tab:red", lw=1.0,
                   label="model T_outer - file T_outer")
        if Ta_file is not None:
            a.plot(hours, res["T_avg_line"] - Ta_file, color="tab:blue", lw=1.0,
                   label="model T_avg (line) - file T_avg")
        a.set_xlabel("time (h)")
        a.set_ylabel("model - file (C)")
        a.set_title("difference (the volume average is left out here; it is far off, see top panel)",
                    fontsize=9)
        a.legend(fontsize=8)
        a.grid(alpha=0.3)

    if prof_rows:                                   # 3) temperature across the concrete
        a = ax.pop(0)
        r_inch = res["r"] / 0.0254
        for i in prof_rows:
            a.plot(r_inch, res["profile"][i], lw=1.2, label="t = %g h" % (t[i] / 3600.0))
        a.set_xlabel("radius (in)   [inner wall at 1 in, outer surface at 5 in]")
        a.set_ylabel("temperature (C)")
        a.set_title("model temperature across the concrete", fontsize=10)
        a.legend(fontsize=8)
        a.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(png_path, dpi=130)
    plt.close(fig)
    print("plot -> %s" % png_path)


def main(argv=None):
    p = argparse.ArgumentParser(description="1-D radial conduction model of the concrete ring (FE report geometry "
                                            "and properties), driven by one column of an FE csv file.")
    p.add_argument("csv", help="FE temperature-input csv file")
    p.add_argument("--drive", choices=["inner", "input"], default="inner",
                   help="inner wall = file's T_inner column (default) or its Input Temperature column")
    p.add_argument("--layers", type=int, default=N_LAYERS, help="equal radial layers (default %d)" % N_LAYERS)
    p.add_argument("--out", help="output csv (default: <file name>_model.csv in the current folder)")
    p.add_argument("--plot", action="store_true", help="save a PNG of file vs model next to the output csv")
    p.add_argument("--profile-times", help="comma-separated times in s; writes <out>_profile.csv")
    args = p.parse_args(argv)
    if args.layers < 2:
        sys.exit("error: --layers must be at least 2")

    drive_col = DRIVE_COLUMN[args.drive]
    t, T_b, To_file, Ta_file = read_csv(args.csv, drive_col)
    out = args.out or os.path.splitext(os.path.basename(args.csv))[0] + "_model.csv"
    stem = os.path.splitext(out)[0]
    out_dir = os.path.dirname(os.path.abspath(out))
    if not os.path.isdir(out_dir):
        sys.exit("error: the output folder does not exist: %s" % out_dir)
    prof_rows = profile_rows(args.profile_times, t) if args.profile_times else []

    res = simulate(T_b, t, n_layers=args.layers, keep_profile=bool(prof_rows))

    table = pd.DataFrame({
        "Time (s)": t,
        "Boundary used (C)": T_b,
        "T_outer model (C)": res["T_outer"],
        "T_avg line model (C)": res["T_avg_line"],
        "T_avg volume model (C)": res["T_avg_volume"],
    })
    if To_file is not None:
        table["T_outer file (C)"] = To_file
    if Ta_file is not None:
        table["T_avg file (C)"] = Ta_file
    table.to_csv(out, index=False, float_format="%.4f")

    print("file: %s" % args.csv)
    print("inner wall driven by '%s', %d layers, %d time rows -> %s" % (drive_col, args.layers, len(t), out))
    print("errors over all %d rows (C):            RMSE    MAE" % len(t))
    if To_file is not None:
        print("  T_outer: model vs file             %6.3f  %6.3f" % (rmse(res["T_outer"], To_file),
                                                                    mae(res["T_outer"], To_file)))
    if Ta_file is not None:
        print("  T_avg:   file vs model line avg    %6.3f  %6.3f" % (rmse(res["T_avg_line"], Ta_file),
                                                                    mae(res["T_avg_line"], Ta_file)))
        print("  T_avg:   file vs model volume avg  %6.3f  %6.3f" % (rmse(res["T_avg_volume"], Ta_file),
                                                                    mae(res["T_avg_volume"], Ta_file)))
    if To_file is None and Ta_file is None:
        print("  (no T_outer / T_avg columns in the file, nothing to compare)")

    if prof_rows:
        write_profile(stem + "_profile.csv", t, res, prof_rows)
    if args.plot:
        title = "%s   (inner wall = %s, %d layers)" % (os.path.basename(args.csv), drive_col, args.layers)
        plot(stem + ".png", t, T_b, drive_col, To_file, Ta_file, res, prof_rows, title)
    return 0


if __name__ == "__main__":
    sys.exit(main())
