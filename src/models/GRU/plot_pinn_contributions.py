import os
import re
import matplotlib.pyplot as plt

LOG_PATH = "sweep_logs/fix-F.log"

# Arm F Configuration Weights
W_PHYS = 0.05
W_SMOOTH_TO = 0.005
W_SMOOTH_TA = 0.01

def parse_log(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Could not find log file at {filepath}")
        
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Split logs into segments by seed
    seed_blocks = re.split(r"#{70,}\n# SEED", content)
    
    parsed_data = {}
    for block in seed_blocks[1:]:  # Skip leading metadata block
        # Extract seed number
        seed_match = re.search(r"seed=(\d+)", block)
        if not seed_match:
            continue
        seed_id = int(seed_match.group(1))
        
        epochs = []
        train_composite = []
        phys_raw = []
        smooth_to_raw = []
        smooth_ta_raw = []
        
        # Regex to parse the metrics printed every 10 epochs
        pattern = re.compile(
            r"epoch\s+(\d+)\s+\((?:warmup|main\s+)\):\s+train=([\d\.]+)\s+val=[\d\.]+\s+\|\s+PINN:\s+phys=([\d\.]+)\s+smooth_to=([\d\.]+)\s+smooth_ta=([\d\.]+)"
        )
        
        for line in block.split("\n"):
            m = pattern.search(line)
            if m:
                epochs.append(int(m.group(1)))
                train_composite.append(float(m.group(2)))
                phys_raw.append(float(m.group(3)))
                smooth_to_raw.append(float(m.group(4)))
                smooth_ta_raw.append(float(m.group(5)))
                
        if epochs:
            parsed_data[seed_id] = {
                "epochs": epochs,
                "composite": train_composite,
                "phys": phys_raw,
                "smooth_to": smooth_to_raw,
                "smooth_ta": smooth_ta_raw
            }
            
    return parsed_data

def generate_plots(data):
    fig, axes = plt.subplots(len(data), 2, figsize=(15, 5 * len(data)), sharex=True)
    if len(data) == 1:
        axes = [axes]
        
    for idx, (seed, metrics) in enumerate(sorted(data.items())):
        epochs = metrics["epochs"]
        
        # 1. Compute weighted contributions
        weighted_phys = [p * W_PHYS for p in metrics["phys"]]
        weighted_smooth_to = [s * W_SMOOTH_TO for s in metrics["smooth_to"]]
        weighted_smooth_ta = [s * W_SMOOTH_TA for s in metrics["smooth_ta"]]
        
        total_pinn_penalty = [
            wp + w_to + w_ta 
            for wp, w_to, w_ta in zip(weighted_phys, weighted_smooth_to, weighted_smooth_ta)
        ]
        
        # 2. Isolate Pure GRU Data Loss
        pure_gru_data_loss = [
            comp - pinn
            for comp, pinn in zip(metrics["composite"], total_pinn_penalty)
        ]
        
        ax_left = axes[idx][0]
        ax_right = axes[idx][1]
        
        # Left Panel: Data Loss vs Physics Regularization Loss
        ax_left.plot(epochs, pure_gru_data_loss, label="Pure GRU Data Loss ($L_{data}$)", color="blue", linewidth=2)
        ax_left.plot(epochs, total_pinn_penalty, label="Total Weighted PINN Loss ($L_{PINN}$)", color="crimson", linewidth=2)
        ax_left.set_title(f"Seed {seed}: Data Loss vs. Physics Penalty")
        ax_left.set_ylabel("Weighted Loss (Scaled Space)")
        ax_left.grid(True, alpha=0.3)
        ax_left.legend()
        ax_left.set_yscale("log")
        
        # Right Panel: Breakdown of individual physical contributions
        ax_right.plot(epochs, weighted_phys, label="Boundary Violation ($0.05 \\times phys$)", color="orange", linestyle="--")
        ax_right.plot(epochs, weighted_smooth_to, label="Outer Smoothness ($0.005 \\times smooth\_to$)", color="forestgreen", linestyle="--")
        ax_right.plot(epochs, weighted_smooth_ta, label="Average Smoothness ($0.01 \\times smooth\_ta$)", color="purple", linestyle="--")
        ax_right.set_title(f"Seed {seed}: PINN Component Contributions")
        ax_right.set_ylabel("Weighted Loss (Scaled Space)")
        ax_right.grid(True, alpha=0.3)
        ax_right.legend()
        ax_right.set_yscale("log")
        
        if idx == len(data) - 1:
            ax_left.set_xlabel("Epoch")
            ax_right.set_xlabel("Epoch")
            
    plt.suptitle("Arm F: Deep Dive Loss Contribution Analysis", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    output_img = "comparison/pinn_loss_contributions.png"
    os.makedirs(os.path.dirname(output_img), exist_ok=True)
    plt.savefig(output_img, dpi=150, bbox_inches="tight")
    print(f"Analysis plot successfully saved -> {output_img}")

if __name__ == "__main__":
    p_data = parse_log(LOG_PATH)
    generate_plots(p_data)
