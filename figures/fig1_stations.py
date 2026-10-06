"""Stage E — Figure 1: multi-scale CNEMC station map on a China province base map (R3.5).

Draws China's provincial boundaries, shades the five study provinces in colour, overlays the
1,733 monitoring stations, and adds an enlarged Beijing-Tianjin-Hebei (BTH) inset.
Province polygons: Natural Earth 1:10m admin-1 (cached locally on first run).
Out: outputs/figures/fig1_stations.png
"""
from __future__ import annotations
import os, sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COLORS = {"BTH": "#d7191c", "Jiangsu": "#2c7bb6", "Jilin": "#7b3294",
          "Guangdong": "#1a9641", "Xinjiang": "#e66101"}
REGION_PROV = {"Jilin": ["Jilin"], "Jiangsu": ["Jiangsu"], "Guangdong": ["Guangdong"],
               "Xinjiang": ["Xinjiang"], "BTH": ["Beijing", "Tianjin", "Hebei"]}
BTH_BOX = (113.0, 36.0, 120.5, 42.5)
NE_URL = "https://naturalearth.s3.amazonaws.com/10m_cultural/ne_10m_admin_1_states_provinces.zip"


def china_provinces():
    import geopandas as gpd
    cache = os.path.join(ROOT, "outputs", "grids", "china_provinces.gpkg")
    if os.path.exists(cache):
        return gpd.read_file(cache)
    g = gpd.read_file(NE_URL)
    cn = g[g["admin"] == "China"].copy()
    namecol = "name_en" if "name_en" in cn.columns else "name"
    cn = cn[[namecol, "geometry"]].rename(columns={namecol: "prov"})
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    cn.to_file(cache, driver="GPKG")
    return cn


def _region_of(prov_name):
    for region, toks in REGION_PROV.items():
        if any(t.lower() in str(prov_name).lower() for t in toks):
            return region
    return None


def basemap(ax, prov, lw_border=0.4):
    """Grey provincial boundaries + shaded study provinces."""
    study = prov.copy(); study["region"] = study["prov"].map(_region_of)
    for r, c in COLORS.items():
        sub = study[study.region == r]
        if len(sub):
            sub.plot(ax=ax, color=c, alpha=0.30, zorder=0, linewidth=0)
    prov.boundary.plot(ax=ax, color="0.6", lw=lw_border, zorder=1)


def main():
    prov = china_provinces()
    st = pd.read_csv(os.path.join(ROOT, "dataset/national_AQ_stations.csv"))
    prov2reg = {"吉林省": "Jilin", "江苏省": "Jiangsu", "广东省": "Guangdong",
                "新疆维吾尔自治区": "Xinjiang", "北京市": "BTH", "天津市": "BTH", "河北省": "BTH"}
    st["region"] = st["province"].map(prov2reg)

    # figsize + box chosen so the main axes box ratio (~0.70) matches China's geographic
    # aspect (lat-range*1.2 / lon-range), so set_aspect(1.2) fills the box without distortion or gaps.
    fig = plt.figure(figsize=(7.3, 3.7))
    ax = fig.add_axes([0.06, 0.10, 0.62, 0.85])
    basemap(ax, prov)
    other = st[st.region.isna()]
    ax.scatter(other.lon, other.lat, s=5, c="0.35", alpha=0.6, edgecolors="none", zorder=3)
    for r, c in COLORS.items():
        d = st[st.region == r]
        ax.scatter(d.lon, d.lat, s=14, c=c, edgecolors="k", linewidths=0.2, zorder=4)
    ax.add_patch(Rectangle((BTH_BOX[0], BTH_BOX[1]), BTH_BOX[2]-BTH_BOX[0], BTH_BOX[3]-BTH_BOX[1],
                           fill=False, ec="k", lw=1.2, ls="--", zorder=5))
    ax.set_xlabel("Longitude (°E)"); ax.set_ylabel("Latitude (°N)")
    ax.set_xlim(73, 135.5); ax.set_ylim(17.5, 54)
    ax.set_aspect(1.2)   # correct geographic proportions (no distortion)
    handles = [Patch(facecolor=c, alpha=0.5, edgecolor="k",
                     label=f"{r} (n={int((st.region==r).sum())})") for r, c in COLORS.items()]
    handles.append(plt.Line2D([], [], marker="o", ls="", color="0.35", ms=4, label="Other stations"))
    ax.legend(handles=handles, loc="lower left", fontsize=8, framealpha=0.9)

    # BTH inset — top edge aligned with the main map's top edge (0.10 + 0.85 = 0.95)
    axi = fig.add_axes([0.71, 0.55, 0.27, 0.40])
    basemap(axi, prov, lw_border=0.6)
    b = st[(st.lon.between(BTH_BOX[0], BTH_BOX[2])) & (st.lat.between(BTH_BOX[1], BTH_BOX[3]))]
    axi.scatter(b.lon, b.lat, s=18, c=[COLORS.get(r, "0.35") for r in b.region],
                edgecolors="k", lw=0.3, zorder=4)
    axi.set_xlim(BTH_BOX[0], BTH_BOX[2]); axi.set_ylim(BTH_BOX[1], BTH_BOX[3])
    axi.set_aspect(1.2)
    axi.tick_params(labelsize=7)
    axi.text(0.035, 0.965, "BTH inset", transform=axi.transAxes, fontsize=9, fontweight="bold",
             va="top", ha="left", bbox=dict(fc="white", ec="0.7", alpha=0.85, pad=1.6))

    out = os.path.join(ROOT, "outputs", "figures", "fig1_stations.png")
    fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out}  (stations={len(st)}, study-region stations={int(st.region.notna().sum())})")


if __name__ == "__main__":
    main()
