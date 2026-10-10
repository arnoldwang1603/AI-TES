# PhysTFv3 — ODE-Guided Attention-LSTM Thermal Predictor

**Author:** Sidd (kurras1@uci.edu)  
**Task:** Predict inner wall temperature (T_inner), outer wall temperature (T_outer), and ambient temperature (T_avg) of a thermal energy storage system given an input temperature profile.

---

## Model Architecture

The model has three sequential stages: a Transformer that produces physics coefficients, an ODE integrator that produces physics-based temperature estimates, and a BiLSTM that corrects those estimates.

```
Input: [T_in sequence, ΔT_in sequence, initial conditions]
         │
         ▼
┌─────────────────────────────┐
│     AlphaTransformer        │  4-layer Pre-LN Transformer
│  (4 heads, d_model=128)     │  produces α₁, α₂, α₃ — time-varying
│                             │  thermal diffusivity coefficients
└────────────┬────────────────┘
             │ α₁(t), α₂(t), α₃(t)
             ▼
┌─────────────────────────────┐
│      ODEIntegrator          │  Backward-Euler solver, 1440 steps
│   (backward-Euler, JIT)     │  Physics: 3-node thermal model
│   learnable position x      │  Ti, To solved via Cramer's rule
└────────────┬────────────────┘
             │ Ti_ode(t), To_ode(t)
             ▼
┌─────────────────────────────┐
│      PhysicsLSTM            │  2-layer BiLSTM (hidden=64)
│  + sliding-window attention │  learns residual Δ on top of ODE
│     (window = 120 steps)    │  2-layer attention encoder
└────────────┬────────────────┘
             │
             ▼
     Ti = Ti_ode + ΔTi
     To = To_ode + ΔTo
     Ta = Ta₀   + head_ta(h)    ← separate MLP head
```

**Total parameters:** ~1,337,000

---

## Physics Model (ODE)

The ODE represents a 3-node thermal system: input fluid (Tin), inner wall (Ti), midpoint (Tm), and outer wall (To). The backward-Euler update at each timestep is:

```
Ti_new = (Ti + dt·α₁·Tin/w²) / (1 + dt·α₁/w²)

[A] · [Tm_new, To_new]ᵀ = b     (solved via Cramer's rule)

A = [[1 + dt·α₂·(1/x² + 1/Dx²),  -dt·α₂/Dx²       ],
     [-dt·α₃/Dx²,                  1 + dt·α₃·(1/D² + 1/Dx²)]]
```

Where:
- `D = 0.1016 m` — pipe outer diameter
- `W = 0.00635 m` — wall thickness  
- `x` — learnable parameter: thermal midpoint position (constrained to [0.01, D−0.01])
- `dt = 10 s` — timestep
- `α₁, α₂, α₃` — per-timestep thermal diffusivity coefficients (predicted by Transformer)

The position parameter `x` is learned end-to-end via a sigmoid reparameterisation and represents the effective heat transfer midpoint within the pipe wall.

---

## Training

### Data
- **Training:** 311 CSV files across 10 charging/discharging scenarios (new temperature dataset)  
  - 249 train / 62 validation (80/20 random split, fixed seed)
- **Test:** 70 held-out cases

### Loss
Weighted MAE with physics regularisation:

```
L = w_Ti·|ΔTi| + w_To·|ΔTo| + w_Ta·|ΔTa|
  + α_smooth · Σ(αₖ[t+1] − αₖ[t])²
  + λ_ta · Σ(Ta[t+1] − Ta[t])²
  + λ_phys · ReLU(Ta − max(Ti,To)) + ReLU(min(Ti,To) − Ta)
```

Weights: `w_Ti=1.0, w_To=6.0, w_Ta=3.0` — outer wall weighted heavily as it is the hardest to fit.

### Ta warmup schedule
T_avg supervision is introduced gradually to stabilise training:
- Epochs 0–19: T_avg loss weight = 0 (Ti/To only)
- Epochs 20–39: T_avg weight ramps linearly to 3.0
- Epoch 40+: full supervision

### Hyperparameters
| Parameter | Value |
|---|---|
| Batch size | 16 |
| Max epochs | 300 |
| Early stopping patience | 60 |
| Learning rate | 5e-4 (AdamW) |
| LR schedule | CosineAnnealing to 1e-5 |
| Weight decay | 1e-3 |
| Gradient clip | 1.0 |
| ATTN_WINDOW | 120 steps |

---

## Experiment Results (new temperature dataset, 70 test cases)

Seed × window sweep. Lower is better.

| Seed | Attn Window | Ti MAE (°C) | To MAE (°C) | Ta MAE (°C) | Ti RMSE | To RMSE | Ta RMSE |
|------|------------|------------|------------|------------|---------|---------|---------|
| 7    | 120        | 0.669      | 1.691      | 2.844      | 0.782   | 1.911   | 3.333   |
| 21   | 120        | 0.629      | 1.089      | **2.314**  | 0.739   | 1.260   | 2.740   |
| **42**| 120       | 0.605      | 1.580      | 2.969      | 0.713   | 1.800   | 3.482   |
| 123  | 120        | 0.646      | 1.522      | 2.545      | 0.754   | 1.742   | 3.000   |
| 42   | 60         | 0.627      | 1.394      | 2.898      | 0.736   | 1.578   | 3.382   |
| 42   | 240        | 0.630      | 1.400      | 2.963      | 0.737   | 1.609   | 3.448   |
| **42**| **720**   | **0.558**  | **0.734**  | 2.535      | **0.657**| **0.893**| 2.992 |
| 42   | 1440 (full)| 0.615      | 0.802      | 2.865      | 0.721   | 0.984   | 3.342   |

**Best Ti/To:** seed=42, window=720  
**Best Ta:** seed=21, window=120

---

## Included Checkpoint

`best_model.pt` — seed=42, window=120, trained on full new temperature dataset (300 epochs with early stopping).  
`scaler.pkl` — MinMaxScaler fitted on training data, required for inference.

> Note: seed=42 window=720 produced the best sweep results but its weights were not saved during the sweep. The included checkpoint (window=120) is the full-depth trained model on new data.

---

## File Reference

| File | Purpose |
|---|---|
| `model.py` | Full architecture (AlphaTransformer, ODEIntegrator, PhysicsLSTM, ThermalPredictor) |
| `config.py` | All hyperparameters and paths |
| `dataset.py` | Data loading, scaling, ThermalDataset |
| `train.py` | Training loop with resume support (original data) |
| `train_new_temp.py` | Training on new temperature dataset (saves to `checkpoints_new_temp/`) |
| `experiment_new_temp.py` | Seeds × windows sweep on new temperature dataset |
| `evaluate.py` | Evaluation with per-case plots and metrics |
| `best_model.pt` | Trained weights (see above) |
| `scaler.pkl` | Fitted MinMaxScaler for inference |

---

## Loading for Inference

```python
import pickle, torch
from model import ThermalPredictor

# Load scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load model
device = torch.device('cpu')
model  = ThermalPredictor().to(device)
model.register_scaler(scaler)
ckpt   = torch.load('best_model.pt', map_location=device)
model.load_state_dict(ckpt['state_dict'])
model.eval()

# Forward pass — all inputs are [batch, seq_len] tensors
# preds shape: [batch, 1440, 3]  → columns: [T_inner, T_outer, T_avg]
preds, (a1, a2, a3) = model(tin_scaled, dtin_scaled, init_scaled, tin_phys, init_phys)
```

See `dataset.py` → `ThermalDataset` for how to prepare inputs from a raw CSV.
