#!pip install nltk
#2nd cell
import json
import numpy as np
from jiwer import wer, cer
from sentence_transformers import SentenceTransformer, util

# Initialize the evaluator
print("Loading semantic evaluation model (this takes a moment on the first run)...")
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
print("Model loaded successfully!")

#3rd cell 
def calculate_metrics(ground_truth, whisper_raw, poso_corrected):
    ref_norm = " ".join(ground_truth.lower().split())
    raw_norm = " ".join(whisper_raw.lower().split())
    poso_norm = " ".join(poso_corrected.lower().split())
    
    # 1. Syntactic Metrics (WER/CER)
    whisper_wer = wer(ref_norm, raw_norm)
    poso_wer = wer(ref_norm, poso_norm)
    whisper_cer = cer(ref_norm, raw_norm)
    poso_cer = cer(ref_norm, poso_norm)

    # 2. BLEU Score
    smoother = SmoothingFunction().method1
    whisper_bleu = sentence_bleu([ref_norm.split()], raw_norm.split(), smoothing_function=smoother)
    poso_bleu = sentence_bleu([ref_norm.split()], poso_norm.split(), smoothing_function=smoother)
    
    # 3. Semantic Metrics (SEAR)
    emb_ref = model.encode(ground_truth, convert_to_tensor=True)
    emb_raw = model.encode(whisper_raw, convert_to_tensor=True)
    emb_poso = model.encode(poso_corrected, convert_to_tensor=True)
    
    sim_raw = max(0.0, min(1.0, util.cos_sim(emb_ref, emb_raw).item()))
    sim_poso = max(0.0, min(1.0, util.cos_sim(emb_ref, emb_poso).item()))
    
    whisper_sear = 1.0 - sim_raw
    poso_sear = 1.0 - sim_poso
    
    # Absolute improvements
    wer_drop = (whisper_wer - poso_wer) * 100
    sear_drop = (whisper_sear - poso_sear) * 100
    
    return {
        "whisper_baseline": {"WER": round(whisper_wer, 4), "CER": round(whisper_cer, 4), "BLEU": round(whisper_bleu, 4), "SEAR": round(whisper_sear, 4)},
        "poso_pipeline":    {"WER": round(poso_wer, 4),    "CER": round(poso_cer, 4),    "BLEU": round(poso_bleu, 4),    "SEAR": round(poso_sear, 4)},
        "improvements": {
            "WER_dropped_by_percentage_points":  round(wer_drop, 2),
            "CER_dropped_by_percentage_points":  round((whisper_cer - poso_cer) * 100, 2),
            "BLEU_gained_by_percentage_points":  round((poso_bleu - whisper_bleu) * 100, 2),
            "SEAR_dropped_by_percentage_points": round(sear_drop, 2)
        }
    }


#4th cell 
import numpy as np
import matplotlib.pyplot as plt

# 1. Ensure plots display smoothly in your notebook inline
%matplotlib inline

# 2. Extract data points from your pipeline outputs
# (Using the exact data structure from your previous test run)
metrics = ['WER', 'CER', 'SEAR']
whisper_scores = [0.3750, 0.0769, 0.3326]  # Baseline error rates
poso_scores    = [0.0000, 0.0000, 0.0000]  # Corrected pipeline error rates

# 3. Define positioning mathematics for the side-by-side spacing
x = np.arange(len(metrics))  # Label locations: [0, 1, 2]
bar_width = 0.35             # Width of individual bars

# 4. Initialize the figure layout
fig, ax = plt.subplots(figsize=(9, 6), dpi=100)

# 5. Build side-by-side bars using offsets (x - width/2 vs x + width/2)
rects1 = ax.bar(x - bar_width/2, whisper_scores, bar_width, 
                label='Whisper Baseline', color='#e06666', edgecolor='black')
rects2 = ax.bar(x + bar_width/2, poso_scores, bar_width, 
                label='POSO Pipeline', color='#6aa84f', edgecolor='black')

# 6. Stylize labels, titles, and structural axes
ax.set_ylabel('Error Rates (Lower is Better)', fontsize=12, fontweight='bold', labelpad=10)
ax.set_title('POSO Performance Evaluation:\nWhisper vs. Context-Enhanced Correction', 
             fontsize=14, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=11, fontweight='bold')
ax.set_ylim(0, 0.5) # Provide breathing room at the top for labels
ax.grid(axis='y', linestyle='--', alpha=0.5) # Subtle background grid lines
ax.legend(fontsize=11, loc='upper right')

# 7. Dynamically attach exact numeric value labels above each bar
def annotate_bars(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

annotate_bars(rects1)
annotate_bars(rects2)

# 8. Clean up outer margins and render
plt.tight_layout()
plt.show()

