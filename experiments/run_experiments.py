from evaluation.evaluator import StoryEvaluator

from experiments.visualization import (
    plot_metric_bar,
    plot_radar_chart,
    plot_diversity_scatter,
    plot_quality_scatter,
    plot_multi_metric_comparison
)

# =========================================================
# Example story generations
# Replace with YOUR actual generated stories
# =========================================================

bigram_stories = [
    "cat story 1...",
    "cat story 2...",
    "cat story 3..."
]

trigram_stories = [
    "story...",
    "story...",
    "story..."
]

rnn_stories = [
    "story...",
    "story...",
    "story..."
]

lstm_stories = [
    "story...",
    "story...",
    "story..."
]

llm_stories = [
    "story...",
    "story...",
    "story..."
]

keywords = [
    "cat",
    "sun",
    "castle"
]

# =========================================================
# Evaluator
# =========================================================

evaluator = StoryEvaluator()

# =========================================================
# Evaluate models
# =========================================================

bigram_results = evaluator.evaluate_runs(
    bigram_stories,
    keywords
)

trigram_results = evaluator.evaluate_runs(
    trigram_stories,
    keywords
)

rnn_results = evaluator.evaluate_runs(
    rnn_stories,
    keywords
)

lstm_results = evaluator.evaluate_runs(
    lstm_stories,
    keywords
)

llm_results = evaluator.evaluate_runs(
    llm_stories,
    keywords
)

# =========================================================
# Combine all
# =========================================================

all_results = {

    "Bigram": bigram_results,

    "Trigram": trigram_results,

    "RNN": rnn_results,

    "LSTM": lstm_results,

    "LLM": llm_results
}

# =========================================================
# Print results
# =========================================================

for model, vals in all_results.items():

    print("\n===================")
    print(model)
    print("===================")

    for k, v in vals.items():
        print(f"{k}: {v}")

# =========================================================
# Generate Figures
# =========================================================

# ---- BAR CHARTS ----

plot_metric_bar(
    all_results,
    "keyword_coverage"
)

plot_metric_bar(
    all_results,
    "semantic_coherence"
)

plot_metric_bar(
    all_results,
    "syntax_validity"
)

plot_metric_bar(
    all_results,
    "distinct_1"
)

plot_metric_bar(
    all_results,
    "repetition_rate"
)

# ---- RADAR CHART ----

plot_radar_chart(

    all_results,

    [
        "keyword_coverage",
        "distinct_1",
        "semantic_coherence",
        "syntax_validity",
        "repetition_rate"
    ]
)

# ---- SCATTER PLOTS ----

plot_diversity_scatter(
    all_results
)

plot_quality_scatter(
    all_results
)

# ---- MULTI METRIC FIGURE ----

plot_multi_metric_comparison(

    all_results,

    [
        "keyword_coverage",
        "semantic_coherence",
        "syntax_validity",
        "distinct_1",
        "repetition_rate"
    ]
)

print("\nAll figures generated.")