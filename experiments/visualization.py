import os
import numpy as np
import matplotlib.pyplot as plt

# Create folder
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Bar Chart
# Main comparison chart
def plot_metric_bar(results, metric, save_dir="saved_results/figures"):
    ensure_dir(save_dir)
    models = list(results.keys())
    values = [results[m][metric] for m in models]

    stds = [results[m].get(metric + "_std", 0) for m in models]

    plt.figure(figsize=(8,5))

    bars = plt.bar(models, values, yerr = stds, capsize = 5)

    plt.ylabel(metric)
    plt.title(f"{metric} Comparison")
    plt.tight_layout()
    path = os.path.join(save_dir, f"{metric}_bar.png")
    plt.savefig(path)
    plt.close()
    print(f"[Saved] {path}")

# Radar Chart
def plot_radar_chart(results, metrics, save_dir="saved_results/figures"):
    ensure_dir(save_dir)

    labels = metrics
    num_metrics = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint = False).tolist()

    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8,8), subplot_kw=dict(polar=True))

    for model, vals in results.items():
        data = [vals[m] for m in metrics]
        data += data[:1]

        ax.plot(angles, data, linewidth = 2, label = model)
        ax.fill(angles, data, alpha = 0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)

    plt.legend(loc = "upper right", bbox_to_anchor = (1.3, 1.1))
    plt.title("Model Comparison Radar Chart")
    plt.tight_layout()
    path = os.path.join(save_dir, "radar_chart.png")
    plt.savefig(path)
    plt.close()
    print(f"[Saved] {path}")

# Diversity and Repetition
def plot_diversity_scatter(results, save_dir="saved_results/figures"):
    ensure_dir(save_dir)
    plt.figure(figsize=(7,6))

    for model, vals in results.items():
        x = vals["distinct_1"]
        y = vals["repetition_rate"]
        plt.scatter(x, y, s = 120, label=model)
        plt.text(x + 0.005, y + 0.002, model)

    plt.xlabel("Distinct-1")
    plt.ylabel("Repetition Rate")
    plt.title("Diversity vs Repetition")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(save_dir, "diversity_vs_repetition.png")
    plt.savefig(path)
    plt.close()
    print(f"[Saved] {path}")

# Coherence and Syntax
def plot_quality_scatter(results, save_dir="saved_results/figures"):
    ensure_dir(save_dir)
    plt.figure(figsize=(7,6))

    for model, vals in results.items():
        x = vals["semantic_coherence"]
        y = vals["syntax_validity"]
        plt.scatter(x, y, s = 120, label = model)
        plt.text(x + 0.003, y + 0.003, model)

    plt.xlabel("Semantic Coherence")
    plt.ylabel("Syntax Validity")
    plt.title("Coherence vs Syntax")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(save_dir, "coherence_vs_syntax.png")
    plt.savefig(path)
    plt.close()
    print(f"[Saved] {path}")

# Bar: Metric
# Best report figure
def plot_multi_metric_comparison(results, metrics, save_dir="saved_results/figures"):
    ensure_dir(save_dir)

    models = list(results.keys())

    x = np.arange(len(models))

    width = 0.12

    plt.figure(figsize=(12,6))

    for i, metric in enumerate(metrics):
        vals = [results[m][metric] for m in models]
        plt.bar(x + i * width, vals, width = width, label = metric)

    plt.xticks(x + width * len(metrics) / 2, models)
    plt.legend()
    plt.title("Multi-Metric Model Comparison")
    plt.tight_layout()
    path = os.path.join(save_dir, "multi_metric_comparison.png")
    plt.savefig(path)
    plt.close()

    print(f"[Saved] {path}")