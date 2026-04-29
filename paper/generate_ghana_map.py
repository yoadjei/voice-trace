"""
Generate a Ghana language-region map for the VoiceTrace presentation.
Uses matplotlib patches from hardcoded region polygons — no external download.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np
import os

OUT = os.path.join(os.path.dirname(__file__), "figures", "ghana_language_map.png")

# ── Language → colour map ───────────────────────────────────────────────────
LANG_COLORS = {
    "Twi":     "#2E7D32",   # green
    "Fante":   "#1A736B",   # teal
    "Ga":      "#0D2B55",   # navy
    "Ewe":     "#6C3582",   # purple
    "Dagbani": "#D35400",   # orange
    "Other":   "#C8C8C8",   # grey
}

# ── Ghana's 16 regions mapped to primary language ───────────────────────────
# (approximate — primary/dominant language for surveillance coverage purposes)
REGION_LANG = {
    "Greater Accra":    "Ga",
    "Ashanti":          "Twi",
    "Eastern":          "Twi",
    "Central":          "Fante",
    "Western":          "Fante",
    "Western North":    "Twi",
    "Ahafo":            "Twi",
    "Bono":             "Twi",
    "Bono East":        "Twi",
    "Volta":            "Ewe",
    "Oti":              "Ewe",
    "Northern":         "Dagbani",
    "North East":       "Dagbani",
    "Savannah":         "Dagbani",
    "Upper East":       "Other",
    "Upper West":       "Other",
}

# ── Approximate region centroids (lon, lat) ──────────────────────────────────
CENTROIDS = {
    "Greater Accra":  (-0.20,  5.60),
    "Ashanti":        (-1.62,  6.75),
    "Eastern":        (-0.47,  6.50),
    "Central":        (-1.20,  5.55),
    "Western":        (-2.30,  5.20),
    "Western North":  (-2.70,  6.50),
    "Ahafo":          (-2.55,  7.25),
    "Bono":           (-2.40,  7.80),
    "Bono East":      (-1.70,  7.90),
    "Volta":          ( 0.32,  6.75),
    "Oti":            ( 0.20,  8.10),
    "Northern":       (-0.95,  9.50),
    "North East":     (-0.25,  10.5),
    "Savannah":       (-1.80,  9.10),
    "Upper East":     (-0.70,  10.9),
    "Upper West":     (-2.60,  10.7),
}

# ── Download Ghana shapefile via requests if available, else use fallback ───
def try_gadm():
    """Try to load GADM level-1 Ghana shapefile. Returns GeoDataFrame or None."""
    try:
        import geopandas as gpd, io, zipfile, requests
        url = ("https://geodata.ucdavis.edu/gadm/gadm4.1/shp/gadm41_GHA_shp.zip")
        r = requests.get(url, timeout=15)
        if r.status_code != 200:
            return None
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall("/tmp/gadm_gha")
        gdf = gpd.read_file("/tmp/gadm_gha/gadm41_GHA_1.shp")
        return gdf
    except Exception:
        return None


def make_map_matplotlib():
    """Fallback: draw approximate region shapes as scatter + annotation."""
    fig, ax = plt.subplots(figsize=(7, 9), facecolor="#F0F4F8")
    ax.set_facecolor("#B8D4E8")   # sea colour

    # Ghana bounding box approx: lon -3.25 to 1.2, lat 4.7 to 11.2
    ax.set_xlim(-3.4, 1.5)
    ax.set_ylim(4.5, 11.4)
    ax.set_aspect("equal")
    ax.axis("off")

    # draw approximate Ghana outline using a polygon
    # coarse clockwise vertices of Ghana's border
    ghana_outline = np.array([
        [-3.25, 5.0], [-3.10, 4.75], [-2.70, 4.72], [-2.40, 4.78],
        [-1.90, 4.73], [-1.60, 4.73], [-1.20, 4.85], [-0.75, 4.78],
        [-0.30, 4.80], [ 0.00, 4.95], [ 0.35, 5.20], [ 0.55, 5.55],
        [ 0.60, 6.25], [ 1.20, 6.50], [ 1.15, 7.00], [ 0.90, 7.40],
        [ 0.55, 8.20], [ 0.60, 9.00], [ 0.40, 9.50], [ 0.30,10.10],
        [-0.05,10.70], [-0.35,11.00], [-0.75,11.10], [-1.20,11.05],
        [-2.00,11.00], [-2.80,10.65], [-3.05,10.30], [-2.95, 9.60],
        [-2.65, 9.00], [-2.85, 8.30], [-3.05, 7.60], [-2.80, 6.85],
        [-3.15, 6.30], [-3.25, 5.70], [-3.25, 5.0],
    ])
    outline_patch = mpatches.Polygon(ghana_outline, closed=True,
                                facecolor="#E8E8E8", edgecolor="#555", lw=1.5, zorder=1)
    ax.add_patch(outline_patch)

    # plot each region centroid as a coloured circle
    for region, (lon, lat) in CENTROIDS.items():
        lang = REGION_LANG[region]
        col  = LANG_COLORS[lang]
        ax.scatter(lon, lat, s=420, color=col, zorder=3,
                   edgecolors="white", linewidths=1.2)

    # region labels (small, white, centred on dot)
    label_offsets = {
        "Greater Accra":  (0.0,  -0.28),
        "Ashanti":        (0.0,  -0.28),
        "Eastern":        (0.22,  0.0),
        "Central":        (0.0,  -0.28),
        "Western":        (-0.1, -0.30),
        "Western North":  (-0.25, 0.18),
        "Ahafo":          (0.22,  0.0),
        "Bono":           (-0.25, 0.18),
        "Bono East":      (0.22,  0.0),
        "Volta":          (0.30,  0.0),
        "Oti":            (0.30,  0.0),
        "Northern":       (0.0,  -0.30),
        "North East":     (0.28,  0.0),
        "Savannah":       (-0.28, 0.0),
        "Upper East":     (0.28,  0.0),
        "Upper West":     (-0.30, 0.0),
    }
    for region, (lon, lat) in CENTROIDS.items():
        dx, dy = label_offsets.get(region, (0, -0.28))
        short = region.replace(" Region", "")
        ax.annotate(short, (lon, lat),
                    xytext=(lon + dx, lat + dy),
                    fontsize=6.5, ha="center", va="center",
                    color="#222",
                    path_effects=[pe.withStroke(linewidth=2, foreground="white")],
                    zorder=4)

    # ── legend ───────────────────────────────────────────────────────────────
    legend_entries = []
    pop = {"Twi": "~9M", "Fante": "~1.5M", "Ga": "~0.8M",
           "Ewe": "~2M", "Dagbani": "~1.2M", "Other": "—"}
    for lang, col in LANG_COLORS.items():
        patch = mpatches.Patch(facecolor=col, edgecolor="white", linewidth=0.8,
                               label=f"{lang}  {pop[lang]}")
        legend_entries.append(patch)

    leg = ax.legend(handles=legend_entries, loc="lower left",
                    title="Primary Language", title_fontsize=9,
                    fontsize=8.5, framealpha=0.92,
                    edgecolor="#999",
                    bbox_to_anchor=(0.01, 0.01))
    leg.get_frame().set_linewidth(0.8)

    # ── title ─────────────────────────────────────────────────────────────────
    ax.set_title("Ghana — VoiceTrace Language Coverage\nby Region",
                 fontsize=12, fontweight="bold", color="#0D2B55", pad=8)

    # ── compass rose (simple N arrow) ─────────────────────────────────────────
    ax.annotate("N", xy=(1.1, 10.9), fontsize=11, fontweight="bold",
                ha="center", color="#333")
    ax.annotate("", xy=(1.1, 10.7), xytext=(1.1, 10.3),
                arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.5))

    plt.tight_layout(pad=0.5)
    plt.savefig(OUT, dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved (matplotlib fallback): {OUT}")


def make_map_geopandas(gdf):
    """Preferred: render GADM shapefile with proper geometry."""
    import geopandas as gpd

    name_col = "NAME_1" if "NAME_1" in gdf.columns else gdf.columns[2]
    gdf = gdf.copy()
    gdf["lang"] = gdf[name_col].map(
        lambda n: next((REGION_LANG[r] for r in REGION_LANG if r.lower() in n.lower()), "Other")
    )
    gdf["color"] = gdf["lang"].map(LANG_COLORS)

    fig, ax = plt.subplots(figsize=(7, 9), facecolor="#F0F4F8")
    ax.set_facecolor("#B8D4E8")
    ax.axis("off")

    gdf.plot(ax=ax, color=gdf["color"], edgecolor="white", linewidth=0.7)

    # region name labels
    for _, row in gdf.iterrows():
        cx = row.geometry.centroid.x
        cy = row.geometry.centroid.y
        ax.text(cx, cy, row[name_col], fontsize=6, ha="center", va="center",
                color="white", fontweight="bold",
                path_effects=[pe.withStroke(linewidth=1.5, foreground="#333")])

    # legend
    entries = [mpatches.Patch(facecolor=LANG_COLORS[l], edgecolor="white",
                               label=l) for l in LANG_COLORS]
    ax.legend(handles=entries, loc="lower left", fontsize=9,
              title="Primary Language", title_fontsize=9)
    ax.set_title("Ghana — VoiceTrace Language Coverage", fontsize=12,
                 fontweight="bold", color="#0D2B55")

    plt.tight_layout()
    plt.savefig(OUT, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved (geopandas): {OUT}")


if __name__ == "__main__":
    gdf = try_gadm()
    if gdf is not None:
        make_map_geopandas(gdf)
    else:
        make_map_matplotlib()
