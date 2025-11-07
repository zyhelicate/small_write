import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import math

def main():
    csv_path = Path("results_parix_alloc_comparison.csv")
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path.resolve()}")

    df = pd.read_csv(csv_path)

    target_k = [5, 7, 11, 13]
    modes = ["seq", "opt", "parix", "parix_opt"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    axes = axes.flatten()

    for ax, k in zip(axes, target_k):
        k_df = df[df["k"] == k].copy()
        if k_df.empty:
            ax.set_title(f"k={k}")
            ax.axis("off")
            continue

        merge_keys = ["update_bytes", "packetsize_bytes", "repeats"]
        parix = k_df[k_df["mode"] == "parix"].set_index(merge_keys)
        parix_opt = k_df[k_df["mode"] == "parix_opt"].set_index(merge_keys)

        merged = parix[["IOPS"]].join(
            parix_opt[["IOPS"]],
            how="inner",
            lsuffix="_parix",
            rsuffix="_parix_opt",
        )

        if merged.empty:
            ax.set_title(f"k={k}")
            ax.axis("off")
            continue

        merged["delta"] = merged["IOPS_parix_opt"] - merged["IOPS_parix"]
        best_key = merged["delta"].idxmax()

        values_k = []
        values_raw = []
        for mode in modes:
            rows = k_df[
                (k_df["mode"] == mode)
                & (k_df["update_bytes"] == best_key[0])
                & (k_df["packetsize_bytes"] == best_key[1])
                & (k_df["repeats"] == best_key[2])
            ]
            if rows.empty:
                values_k.append(math.nan)
                values_raw.append(math.nan)
            else:
                iops_raw = rows.iloc[0]["IOPS"]
                values_raw.append(iops_raw)
                values_k.append(iops_raw / 1000.0)

        ax.bar(modes, values_k, color=["#6c757d", "#007bff", "#20c997", "#ff7f0e"])
        ax.set_title(f"k={k}")
        ax.set_ylabel("IOPS (K)")

        valid_values = [v for v in values_k if not math.isnan(v)]
        if valid_values:
            y_max = max(50, math.ceil(max(valid_values) / 50) * 50)
        else:
            y_max = 50
        ax.set_ylim(0, y_max)
        ax.set_yticks(range(0, int(y_max) + 1, 50))

        offset = y_max * 0.02
        for idx, (val_k, val_raw) in enumerate(zip(values_k, values_raw)):
            if not math.isnan(val_k):
                ax.text(
                    idx,
                    val_k + offset,
                    f"{int(round(val_raw / 1000.0))}K",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )

    fig.suptitle("PARIX Allocation Strategies vs. IOPS", fontsize=16, fontweight="bold")
    output_path = Path("parix_alloc_comparison.png")
    fig.savefig(output_path, dpi=300)
    print(f"Figure saved to: {output_path.resolve()}")

if __name__ == "__main__":
    main()
