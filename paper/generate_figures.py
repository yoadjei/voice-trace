"""generate paper figures for voicetrace paper.

figures are tuned for ieee single-column width (~3.4 in) at 300 dpi
so that text remains legible after the latex compiler shrinks them.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from pathlib import Path

OUT = Path("paper/figures")
OUT.mkdir(parents=True, exist_ok=True)

COLORS = {
    "khaya":  "#2E86AB",
    "llm":    "#A23B72",
    "geo":    "#F18F01",
    "output": "#C73E1D",
    "caller": "#1F4E79",
    "bg":     "#FAFAFA",
    "arrow":  "#333333",
    "muted":  "#555555",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.edgecolor": "#444",
    "axes.linewidth": 0.8,
})


# ---------------------------------------------------------------------------
# Figure 1 : Architecture diagram
# ---------------------------------------------------------------------------

def fig_architecture():
    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.4)
    ax.axis("off")
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    def box(x, y, w, h, color, title, sub):
        r = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.10,rounding_size=0.18",
            facecolor=color, edgecolor="white", linewidth=1.6,
            alpha=0.95, zorder=3,
        )
        ax.add_patch(r)
        ax.text(x + w / 2, y + h / 2 + 0.22, title,
                ha="center", va="center",
                fontsize=13, fontweight="bold", color="white", zorder=4)
        ax.text(x + w / 2, y + h / 2 - 0.30, sub,
                ha="center", va="center",
                fontsize=11, color="white", alpha=0.95, zorder=4)

    def arrow(x0, y0, x1, y1, color=None, lw=1.8):
        ax.add_patch(FancyArrowPatch(
            (x0, y0), (x1, y1),
            arrowstyle="-|>", mutation_scale=18,
            color=color or COLORS["arrow"], linewidth=lw, zorder=2,
        ))

    # caller
    caller = FancyBboxPatch(
        (0.15, 3.55), 1.45, 1.10,
        boxstyle="round,pad=0.08,rounding_size=0.14",
        facecolor=COLORS["caller"], edgecolor="white", linewidth=1.4, zorder=3,
    )
    ax.add_patch(caller)
    ax.text(0.875, 4.32, "Caller", ha="center", va="center",
            fontsize=12, fontweight="bold", color="white", zorder=4)
    ax.text(0.875, 3.85, "(any of 5 languages)", ha="center", va="center",
            fontsize=9, color="white", alpha=0.95, zorder=4)

    # pipeline stages along a horizontal axis
    stages = [
        (1.95, 3.55, 1.85, 1.10, COLORS["khaya"], "Stage 1", "Khaya ASR"),
        (4.10, 3.55, 1.85, 1.10, COLORS["khaya"], "Stage 2", "Khaya MT"),
        (6.25, 3.55, 1.85, 1.10, COLORS["llm"],   "Stage 3", "Claude LLM"),
        (8.40, 3.55, 1.85, 1.10, COLORS["khaya"], "Stage 4", "Khaya TTS"),
    ]
    for (x, y, w, h, c, title, sub) in stages:
        box(x, y, w, h, c, title, sub)

    # arrows along the pipeline
    arrow(1.60, 4.10, 1.95, 4.10)
    arrow(3.80, 4.10, 4.10, 4.10)
    arrow(5.95, 4.10, 6.25, 4.10)
    arrow(8.10, 4.10, 8.40, 4.10)
    arrow(10.25, 4.10, 10.85, 4.10, color=COLORS["output"])

    ax.text(11.45, 4.10, "Spoken\nresponse",
            ha="center", va="center",
            fontsize=11, color=COLORS["output"], fontweight="bold")

    # surveillance branch
    arrow(7.175, 3.55, 7.175, 2.10, color=COLORS["geo"])
    box(6.25, 0.95, 1.85, 1.10, COLORS["geo"], "Stage 5", "Geocode + Log")
    ax.text(7.175, 0.45, "Surveillance record",
            ha="center", va="center",
            fontsize=11, color=COLORS["geo"], fontweight="bold")

    # legend
    handles = [
        mpatches.Patch(color=COLORS["khaya"], label="Khaya API (ASR / MT / TTS)"),
        mpatches.Patch(color=COLORS["llm"],   label="Claude LLM"),
        mpatches.Patch(color=COLORS["geo"],   label="Geocoding & Logging"),
    ]
    leg = ax.legend(
        handles=handles, loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3, fontsize=10, framealpha=0.0, handlelength=1.4,
    )
    for txt in leg.get_texts():
        txt.set_color(COLORS["muted"])

    fig.tight_layout(pad=0.6)
    out = OUT / "architecture_diagram.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {out}")


# ---------------------------------------------------------------------------
# Figure 2 : F1 / BERTScore / kappa by language
# ---------------------------------------------------------------------------

def fig_f1_languages():
    langs      = ["Twi", "Fante", "Ewe", "Ga", "Dagbani"]
    macro_f1   = [0.6595, 0.5528, 0.2330, 0.1863, 0.1726]
    bertscore  = [0.3688, 0.6954, 0.5713, 0.5133, 0.5212]
    kappa_gold = [0.803,  0.519,  0.132,  0.093,  0.076]

    x = np.arange(len(langs))
    width = 0.27

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.set_facecolor("#FCFCFC")
    fig.patch.set_facecolor("white")

    b1 = ax.bar(x - width, macro_f1,   width,
                label="Macro F1 (extraction)",
                color="#2E86AB", edgecolor="white", linewidth=0.8)
    b2 = ax.bar(x,         bertscore,  width,
                label="BERTScore F1 (round trip)",
                color="#F18F01", edgecolor="white", linewidth=0.8)
    b3 = ax.bar(x + width, kappa_gold, width,
                label=r"Mean $\kappa$ vs gold",
                color="#A23B72", edgecolor="white", linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(langs, fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_ylim(0, 1.20)
    ax.tick_params(axis="y", labelsize=10)
    ax.set_yticks(np.arange(0, 1.01, 0.2))

    ax.set_title(
        "Track 2: extraction F1, translation fidelity, and "
        r"$\kappa$ against gold annotation",
        fontsize=12, pad=12,
    )

    leg = ax.legend(
        fontsize=10, loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3, frameon=False,
    )
    for txt in leg.get_texts():
        txt.set_color(COLORS["muted"])

    ax.axhline(0.6, color="#888", linestyle="--", lw=0.9, alpha=0.6)
    ax.text(len(langs) - 1 + 0.42, 0.61,
            r"$\kappa = 0.6$ threshold",
            fontsize=9, color="#666", va="bottom", ha="right")

    ax.yaxis.grid(True, linestyle="--", alpha=0.45)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # value labels on bars
    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.018,
                    f"{h:.2f}", ha="center", va="bottom",
                    fontsize=9, color="#222")

    # tier shading (cosmetic, behind bars)
    ax.axvspan(-0.5, 1.5, color="#2E86AB", alpha=0.05, zorder=0)
    ax.axvspan(1.5, 4.5, color="#A23B72", alpha=0.04, zorder=0)
    ax.text(0.5, 0.93, "Tier 1", ha="center", va="bottom",
            fontsize=10, color="#2E86AB", fontweight="bold", alpha=0.9)
    ax.text(3.0, 0.93, "Tier 2", ha="center", va="bottom",
            fontsize=10, color="#A23B72", fontweight="bold", alpha=0.9)

    fig.tight_layout(pad=0.8)
    out = OUT / "f1_by_language.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {out}")


# ---------------------------------------------------------------------------
# Figure 3 : WER comparison across languages (new, helps tell the story)
# ---------------------------------------------------------------------------

def fig_wer():
    langs = ["Assault\n(Twi)", "Drowning\n(Twi)", "Burn\n(Twi)",
             "Occup.\n(Twi)", "RTA\n(Twi)", "Fall\n(Twi)"]
    wer = [47.5, 49.4, 50.1, 50.2, 53.1, 53.2]

    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#FCFCFC")

    bars = ax.barh(langs, wer, color="#2E86AB",
                   edgecolor="white", linewidth=0.8, alpha=0.92)

    overall = 51.4
    ax.axvline(overall, color="#C73E1D", linestyle="--", lw=1.2)
    ax.text(overall + 0.4, len(langs) - 0.3,
            f"overall WER = {overall:.1f}%",
            color="#C73E1D", fontsize=10, va="center")

    for bar, val in zip(bars, wer):
        ax.text(val + 0.4, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=10, color="#222")

    ax.set_xlim(0, max(wer) + 8)
    ax.set_xlabel("Word Error Rate (%)", fontsize=12)
    ax.set_title("Track 1: ASR word error rate by injury type (Twi)",
                 fontsize=12, pad=10)
    ax.tick_params(axis="both", labelsize=10)
    ax.xaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout(pad=0.6)
    out = OUT / "wer_by_injury.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {out}")


# ---------------------------------------------------------------------------
# Figure 4 : Confusion matrix for Twi injury_type (post-ASR)
# ---------------------------------------------------------------------------

def fig_confusion_twi():
    import pandas as pd

    GOLD = Path("data/synthetic/gold_annotations.csv")
    PRED = Path("data/synthetic/extraction_twi.csv")
    if not (GOLD.exists() and PRED.exists()):
        print("skip confusion matrix: data files missing")
        return

    gold = pd.read_csv(GOLD)
    pred = pd.read_csv(PRED)

    # standardise key
    keys = [k for k in ("id", "narrative_id") if k in gold.columns and k in pred.columns]
    if not keys:
        print("skip confusion matrix: no shared key column")
        return
    key = keys[0]

    df = gold[[key, "injury_type"]].merge(
        pred[[key, "injury_type"]], on=key, suffixes=("_gold", "_pred")
    )
    labels = ["rta", "fall", "assault", "burn",
              "drowning", "occupational", "unknown"]

    cm = np.zeros((len(labels), len(labels)), dtype=int)
    label_idx = {l: i for i, l in enumerate(labels)}
    for _, row in df.iterrows():
        g = str(row["injury_type_gold"]).strip().lower()
        p = str(row["injury_type_pred"]).strip().lower()
        if g in label_idx and p in label_idx:
            cm[label_idx[g], label_idx[p]] += 1

    fig, ax = plt.subplots(figsize=(7.0, 5.6))
    fig.patch.set_facecolor("white")
    im = ax.imshow(cm, cmap="Blues", aspect="auto")

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Predicted (extraction on round-tripped Twi)",
                  fontsize=11)
    ax.set_ylabel("Gold annotation", fontsize=11)
    ax.set_title("injury_type confusion matrix (Twi, n = 80)",
                 fontsize=12, pad=10)

    vmax = cm.max() if cm.max() > 0 else 1
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = cm[i, j]
            if v == 0:
                continue
            ax.text(j, i, str(v),
                    ha="center", va="center",
                    fontsize=10,
                    color="white" if v > vmax * 0.55 else "#222")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=9)

    fig.tight_layout(pad=0.6)
    out = OUT / "confusion_twi_injury.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    fig_architecture()
    fig_f1_languages()
    fig_wer()
    fig_confusion_twi()
    print("done.")
