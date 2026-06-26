import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from jiwer import wer, cer
from sentence_transformers import SentenceTransformer, util
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# ── Tuneable constants ──────────────────────────────────────────────────────
EVAL_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
# ────────────────────────────────────────────────────────────────────────────

_PROJECT_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
)
_METADATA_CSV = os.path.join(_PROJECT_ROOT, "data", "metadata.csv")
_OUTPUT_PLOT  = os.path.join(_PROJECT_ROOT, "data", "evaluation_plot.png")


def calculate_metrics(
    reference: str,
    hypothesis: str,
    model: SentenceTransformer,
) -> dict[str, float]:
    """
    Calculate syntactic and semantic metrics comparing a hypothesis transcript
    against a reference transcript.

    Parameters
    ----------
    reference : str
        The reference (ground truth or corrected) text.
    hypothesis : str
        The hypothesis (raw or generated) text.
    model : SentenceTransformer
        The SentenceTransformer model used to compute semantic embeddings.

    Returns
    -------
    dict[str, float]
        A dictionary containing WER, CER, BLEU, and SEAR scores.
    """
    # Normalize inputs
    ref_norm = " ".join(reference.lower().split())
    hyp_norm = " ".join(hypothesis.lower().split())

    if not ref_norm and not hyp_norm:
        return {"WER": 0.0, "CER": 0.0, "BLEU": 1.0, "SEAR": 0.0, "similarity": 1.0}
    elif not ref_norm or not hyp_norm:
        return {"WER": 1.0, "CER": 1.0, "BLEU": 0.0, "SEAR": 1.0, "similarity": 0.0}

    # 1. Syntactic Metrics (WER/CER)
    score_wer = wer(ref_norm, hyp_norm)
    score_cer = cer(ref_norm, hyp_norm)

    # 2. BLEU Score
    smoother = SmoothingFunction().method1
    score_bleu = sentence_bleu(
        [ref_norm.split()],
        hyp_norm.split(),
        smoothing_function=smoother,
    )

    # 3. Semantic Metrics (SEAR)
    emb_ref = model.encode(reference, convert_to_tensor=True)
    emb_hyp = model.encode(hypothesis, convert_to_tensor=True)

    cos_sim = util.cos_sim(emb_ref, emb_hyp).item()
    sim_score = max(0.0, min(1.0, cos_sim))
    score_sear = 1.0 - sim_score

    return {
        "WER": round(score_wer, 4),
        "CER": round(score_cer, 4),
        "BLEU": round(score_bleu, 4),
        "SEAR": round(score_sear, 4),
        "similarity": round(sim_score, 4),
    }


def _generate_and_save_plot(
    avg_wer: float,
    avg_cer: float,
    avg_sear: float,
    plot_path: str,
) -> None:
    """
    Generate a bar chart comparing Whisper Baseline and POSO Pipeline error rates,
    and save the output image.
    """
    metrics = ["WER", "CER", "SEAR"]
    whisper_scores = [avg_wer, avg_cer, avg_sear]
    poso_scores = [0.0, 0.0, 0.0]  # POSO acts as reference, error rates are 0.0

    x = np.arange(len(metrics))
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(9, 6), dpi=100)

    rects1 = ax.bar(
        x - bar_width / 2,
        whisper_scores,
        bar_width,
        label="Whisper Baseline",
        color="#e06666",
        edgecolor="black",
    )
    rects2 = ax.bar(
        x + bar_width / 2,
        poso_scores,
        bar_width,
        label="POSO Pipeline (Reference)",
        color="#6aa84f",
        edgecolor="black",
    )

    ax.set_ylabel(
        "Error Rates (Lower is Better)",
        fontsize=12,
        fontweight="bold",
        labelpad=10,
    )
    ax.set_title(
        "POSO Performance Evaluation:\nWhisper vs. Context-Enhanced Correction",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11, fontweight="bold")
    
    # Set y-axis limit dynamically with some padding
    max_val = max(whisper_scores)
    ax.set_ylim(0, max(0.5, max_val * 1.3))
    
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(fontsize=11, loc="upper right")

    def annotate_bars(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.4f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

    annotate_bars(rects1)
    annotate_bars(rects2)

    plt.tight_layout()
    
    # Save the plot
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path)
    print(f"[Evaluation] Plot saved to: {plot_path}")
    
    # Show the plot if interactive
    try:
        if plt.get_backend() != "agg":
            plt.show()
    except Exception:
        pass
    finally:
        plt.close(fig)


def run_evaluation(
    csv_path: str = _METADATA_CSV,
    plot_path: str = _OUTPUT_PLOT,
) -> None:
    """
    Read metadata from the given CSV path, extract the initial whisper transcripts
    and RAG corrected transcripts (the last two columns), evaluate them against each
    other, print summary statistics, and generate a comparison plot.

    Parameters
    ----------
    csv_path : str
        Path to the metadata CSV file.
    plot_path : str
        Path where the evaluation bar chart should be saved.

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist.
    ValueError
        If the CSV file is empty or has fewer than two columns.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"[Evaluation] CSV file not found: {csv_path}")

    # Read data
    rows = []
    with open(csv_path, "r", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        for r in reader:
            if r:
                rows.append(r)

    if not rows:
        raise ValueError(f"[Evaluation] No data rows found in {csv_path}")

    print(f"[Evaluation] Loaded {len(rows)} samples from {csv_path}")
    print("[Evaluation] Loading SentenceTransformer model...")
    model = SentenceTransformer(EVAL_MODEL_NAME)
    print("[Evaluation] Model loaded.")

    all_metrics = []
    
    print("\n" + "=" * 100)
    print(f"{'Dialect':<10} | {'Whisper Raw':<30} | {'RAG Corrected (Ref)':<30} | {'WER':<8} | {'CER':<8} | {'SEAR':<8}")
    print("-" * 100)

    for r in rows:
        if len(r) < 2:
            continue
        
        dialect = r[1] if len(r) > 1 else "unknown"
        whisper_raw = r[-2]
        corrected_ref = r[-1]

        # Calculate metrics
        m = calculate_metrics(reference=corrected_ref, hypothesis=whisper_raw, model=model)
        all_metrics.append(m)

        # Truncate strings for display
        w_disp = (whisper_raw[:27] + "...") if len(whisper_raw) > 30 else whisper_raw
        c_disp = (corrected_ref[:27] + "...") if len(corrected_ref) > 30 else corrected_ref

        print(f"{dialect:<10} | {w_disp:<30} | {c_disp:<30} | {m['WER']:.4f} | {m['CER']:.4f} | {m['SEAR']:.4f}")

    print("=" * 100)

    if not all_metrics:
        print("[Evaluation] No valid rows to evaluate.")
        return

    # Compute averages
    avg_wer = np.mean([m["WER"] for m in all_metrics])
    avg_cer = np.mean([m["CER"] for m in all_metrics])
    avg_bleu = np.mean([m["BLEU"] for m in all_metrics])
    avg_sear = np.mean([m["SEAR"] for m in all_metrics])

    print(f"\n[Evaluation] Overall Dataset Averages:")
    print(f"  • Word Error Rate (WER)      : {avg_wer:.4f}")
    print(f"  • Character Error Rate (CER) : {avg_cer:.4f}")
    print(f"  • BLEU Score                 : {avg_bleu:.4f}")
    print(f"  • Semantic Error Rate (SEAR) : {avg_sear:.4f}")

    # Generate and save the plot
    _generate_and_save_plot(avg_wer, avg_cer, avg_sear, plot_path)


if __name__ == "__main__":
    import sys

    # Allow custom metadata csv path as CLI argument
    target_csv = sys.argv[1] if len(sys.argv) > 1 else _METADATA_CSV
    run_evaluation(target_csv)
