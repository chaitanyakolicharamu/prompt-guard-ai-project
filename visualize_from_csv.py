import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

CSV_PATH = Path("results/metrics.csv")

df = pd.read_csv(CSV_PATH)

print("\n=== Metrics loaded for visualization ===")
print(df)

# ---- Accuracy bar chart ----
plt.figure(figsize=(10, 5))
x = range(len(df))
plt.bar(x, df["accuracy"])
plt.xticks(x, df["model"], rotation=25, ha="right")
plt.ylim(0, 1.05)
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison (Prompt Injection Detection)")
plt.grid(axis="y", linestyle="--", alpha=0.3)
plt.tight_layout()
plt.savefig("accuracy_comparison.png", dpi=300)
plt.show()

# ---- F1-score bar chart ----
plt.figure(figsize=(10, 5))
x = range(len(df))
plt.bar(x, df["f1"])
plt.xticks(x, df["model"], rotation=25, ha="right")
plt.ylim(0, 1.05)
plt.ylabel("F1 Score")
plt.title("Model F1 Comparison (Prompt Injection Detection)")
plt.grid(axis="y", linestyle="--", alpha=0.3)
plt.tight_layout()
plt.savefig("f1_comparison.png", dpi=300)
plt.show()

print("\nSaved: accuracy_comparison.png and f1_comparison.png")
