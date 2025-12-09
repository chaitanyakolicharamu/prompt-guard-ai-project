import os
import matplotlib.pyplot as plt

# -------------------------------------------------------------------
# 0. Make sure results directory exists
# -------------------------------------------------------------------
os.makedirs("results", exist_ok=True)

# -------------------------------------------------------------------
# 1. Define metrics (you can adjust these numbers if needed)
#    Values here are test accuracies on the adversarial / evaluation sets.
# -------------------------------------------------------------------

# ML / Neural models (Mindgard test set)
ml_models = [
    "Keyword Rule",
    "TF-IDF + LR",
    "BiLSTM",
    "BERT Fine-Tuned"
]

ml_accuracy = [
    0.0339,   # 3.39%  (Keyword baseline on Mindgard)
    0.7547,   # 75.47% (TF-IDF + Logistic Regression)
    0.8530,   # 85.30% (BiLSTM)
    0.9001    # 90.01% (BERT fine-tuned)
]

# GenAI LLM prompting (balanced 300-sample subset)
llm_methods = [
    "Gemini Zero-Shot",
    "Gemini Few-Shot + CoT",
    "Gemini Self-Consistency"
]

llm_accuracy = [
    0.6167,   # 61.67% (Zero-shot)
    0.5667,   # 56.67% (Few-shot + CoT)
    0.5433    # 54.33% (Self-Consistency)
]

# -------------------------------------------------------------------
# 2. Bar chart: ML / Neural model accuracy only
# -------------------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.bar(ml_models, ml_accuracy)
plt.ylabel("Accuracy")
plt.ylim(0, 1.0)
plt.title("Neural Model Accuracy on Adversarial Test Set")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
ml_path = "results/ml_model_accuracy.png"
plt.savefig(ml_path, dpi=300)
plt.close()
print(f"Saved ML accuracy chart to: {ml_path}")

# -------------------------------------------------------------------
# 3. Bar chart: GenAI LLM prompting accuracy only
# -------------------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.bar(llm_methods, llm_accuracy)
plt.ylabel("Accuracy")
plt.ylim(0, 1.0)
plt.title("GenAI LLM Prompting Accuracy (Balanced Subset)")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
llm_path = "results/llm_prompting_accuracy.png"
plt.savefig(llm_path, dpi=300)
plt.close()
print(f"Saved LLM accuracy chart to: {llm_path}")

# -------------------------------------------------------------------
# 4. Combined bar chart: ML vs LLM accuracy
# -------------------------------------------------------------------
combined_labels = ml_models + llm_methods
combined_accuracy = ml_accuracy + llm_accuracy

plt.figure(figsize=(10, 5))
plt.bar(combined_labels, combined_accuracy)
plt.ylabel("Accuracy")
plt.ylim(0, 1.0)
plt.title("Accuracy Comparison: Neural Models vs LLM Prompting")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
combined_path = "results/ml_vs_llm_accuracy.png"
plt.savefig(combined_path, dpi=300)
plt.close()
print(f"Saved combined ML vs LLM chart to: {combined_path}")
