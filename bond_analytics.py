"""
Bond Analytics Tool
====================
Computes YTM, modified duration, convexity and price sensitivity
for a fixed income portfolio. Fetches live yield curve from FRED.

Author: Federico Bonessi | The Meridian Playbook
themeridianplaybook.com
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import requests

# ── PORTFOLIO ────────────────────────────────────────────────────────────────
BONDS = [
    {"name": "US Treasury 2Y",   "face": 1000, "coupon": 4.50, "maturity": 2,  "price": 982.0,  "type": "Govt"},
    {"name": "US Treasury 10Y",  "face": 1000, "coupon": 4.25, "maturity": 10, "price": 958.0,  "type": "Govt"},
    {"name": "US Treasury 30Y",  "face": 1000, "coupon": 4.00, "maturity": 30, "price": 891.0,  "type": "Govt"},
    {"name": "Apple IG Corp",    "face": 1000, "coupon": 3.85, "maturity": 7,  "price": 942.0,  "type": "IG Corp"},
    {"name": "Goldman Sachs HY", "face": 1000, "coupon": 5.75, "maturity": 5,  "price": 978.0,  "type": "HY Corp"},
    {"name": "EUR Bund 10Y",     "face": 1000, "coupon": 2.60, "maturity": 10, "price": 915.0,  "type": "Govt"},
    {"name": "EM Sovereign",     "face": 1000, "coupon": 6.50, "maturity": 8,  "price": 945.0,  "type": "EM"},
]

FRED_SERIES = {
    "2Y":  "DGS2",
    "5Y":  "DGS5",
    "10Y": "DGS10",
    "30Y": "DGS30",
}

COLORS = {
    "Govt":    "#2e6da4",
    "IG Corp": "#27ae60",
    "HY Corp": "#d4824a",
    "EM":      "#8e44ad",
}

# ── FRED FETCH ───────────────────────────────────────────────────────────────
def fetch_yield_curve() -> dict:
    curve = {}
    for tenor, series in FRED_SERIES.items():
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
        try:
            r = requests.get(url, timeout=8)
            for line in reversed(r.text.strip().split("\n")):
                parts = line.split(",")
                if len(parts) == 2 and parts[1].strip() not in (".", ""):
                    curve[tenor] = float(parts[1].strip())
                    break
        except Exception:
            pass
    fallback = {"2Y": 4.85, "5Y": 4.60, "10Y": 4.35, "30Y": 4.55}
    for k in fallback:
        if k not in curve:
            curve[k] = fallback[k]
    return curve

# ── BOND MATH ─────────────────────────────────────────────────────────────────
def compute_ytm(bond: dict, tol: float = 1e-8) -> float:
    face    = bond["face"]
    coupon  = bond["coupon"] / 100 * face / 2   # semi-annual
    n       = bond["maturity"] * 2
    price   = bond["price"]

    ytm = 0.05
    for _ in range(2000):
        cf  = [coupon] * n
        cf[-1] += face
        t   = np.arange(1, n + 1)
        pv  = sum(c / (1 + ytm/2)**ti for c, ti in zip(cf, t))
        dpv = sum(-ti/2 * c / (1 + ytm/2)**(ti+1) for c, ti in zip(cf, t))
        if abs(dpv) < 1e-12: break
        ytm -= (pv - price) / dpv
    return ytm * 100

def compute_duration(bond: dict, ytm_pct: float) -> tuple:
    face   = bond["face"]
    coupon = bond["coupon"] / 100 * face / 2
    n      = bond["maturity"] * 2
    ytm    = ytm_pct / 100 / 2

    cf = [coupon] * n
    cf[-1] += face
    t  = np.arange(1, n + 1)

    pv_total  = sum(c / (1 + ytm)**ti for c, ti in zip(cf, t))
    mac_dur   = sum((ti / 2) * c / (1 + ytm)**ti for c, ti in zip(cf, t)) / pv_total
    mod_dur   = mac_dur / (1 + ytm)
    convexity = sum((ti/2)**2 * c / (1 + ytm)**(ti+2)
                    for c, ti in zip(cf, t)) / pv_total

    return mac_dur, mod_dur, convexity

def price_change(mod_dur: float, convexity: float,
                 price: float, dy: float) -> float:
    return price * (-mod_dur * dy + 0.5 * convexity * dy**2)

# ── DASHBOARD ────────────────────────────────────────────────────────────────
def make_dashboard(bonds: list, curve: dict):
    # Compute all metrics
    results = []
    for b in bonds:
        ytm   = compute_ytm(b)
        mac, mod, conv = compute_duration(b, ytm)
        results.append({**b, "ytm": ytm, "mac_dur": mac,
                        "mod_dur": mod, "convexity": conv})

    fig = plt.figure(figsize=(20, 16), facecolor="white")
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.44, wspace=0.34)

    NAVY  = "#1a3a5c"
    GREY  = "#888899"

    def sax(ax, title=""):
        ax.set_facecolor("white")
        ax.tick_params(colors="#333344", labelsize=8)
        ax.spines[:].set_color("#e8e8e8")
        for l in ax.get_xticklabels() + ax.get_yticklabels():
            l.set_color("#333344")
        if title:
            ax.set_title(title, color=NAVY, fontsize=10,
                         fontweight="bold", pad=9)
        ax.grid(color="#f0f0f0", linewidth=0.5)

    # ── TITLE ────────────────────────────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0, :])
    ax0.set_facecolor("white"); ax0.axis("off")
    ax0.text(0.5, 0.85, "BOND ANALYTICS TOOL",
             ha="center", color=NAVY, fontsize=20, fontweight="bold",
             transform=ax0.transAxes)
    ax0.text(0.5, 0.58,
             f"Fixed Income Portfolio  |  {len(bonds)} instruments  |  "
             f"Live Yield Curve: 2Y {curve['2Y']:.2f}%  "
             f"10Y {curve['10Y']:.2f}%  30Y {curve['30Y']:.2f}%",
             ha="center", color="#444455", fontsize=10,
             transform=ax0.transAxes)
    avg_dur = np.mean([r["mod_dur"] for r in results])
    avg_ytm = np.mean([r["ytm"] for r in results])
    ax0.text(0.5, 0.30,
             f"Portfolio Avg YTM: {avg_ytm:.2f}%   "
             f"Avg Modified Duration: {avg_dur:.1f}y   "
             f"Source: FRED (live)",
             ha="center", color=NAVY, fontsize=9,
             transform=ax0.transAxes)
    ax0.axhline(0.08, color=NAVY, linewidth=0.7, xmin=0.08, xmax=0.92)

    names = [r["name"] for r in results]
    cols  = [COLORS[r["type"]] for r in results]

    # ── 1. YTM BAR ───────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[1, 0])
    ytms = [r["ytm"] for r in results]
    bars = ax1.barh(names, ytms, color=cols, alpha=0.85)
    for bar, val in zip(bars, ytms):
        ax1.text(val + 0.05, bar.get_y() + bar.get_height()/2,
                 f"{val:.2f}%", va="center", fontsize=8, color=NAVY)
    ax1.set_xlabel("YTM (%)", color="#555566", fontsize=8)
    sax(ax1, "Yield to Maturity")

    # ── 2. MODIFIED DURATION ─────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 1])
    mods = [r["mod_dur"] for r in results]
    bars = ax2.barh(names, mods, color=cols, alpha=0.85)
    for bar, val in zip(bars, mods):
        ax2.text(val + 0.05, bar.get_y() + bar.get_height()/2,
                 f"{val:.1f}y", va="center", fontsize=8, color=NAVY)
    ax2.set_xlabel("Modified Duration (years)", color="#555566", fontsize=8)
    sax(ax2, "Modified Duration")

    # ── 3. YIELD CURVE (LIVE) ─────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 2])
    tenors = [2, 5, 10, 30]
    yields = [curve["2Y"], curve["5Y"], curve["10Y"], curve["30Y"]]
    ax3.plot(tenors, yields, color=NAVY, linewidth=2.2, marker="o",
             markersize=6, markerfacecolor="white", markeredgecolor=NAVY,
             markeredgewidth=1.5, zorder=3)
    ax3.fill_between(tenors, yields, min(yields) - 0.1,
                     alpha=0.08, color=NAVY)
    for t, y in zip(tenors, yields):
        ax3.text(t, y + 0.04, f"{y:.2f}%", ha="center",
                 fontsize=8, color=NAVY, fontweight="bold")
    ax3.set_xlabel("Maturity (years)", color="#555566", fontsize=8)
    ax3.set_ylabel("Yield (%)", color="#555566", fontsize=8)
    sax(ax3, "Live US Yield Curve (FRED)")

    # ── 4. PRICE SENSITIVITY (+/- 100bps) ────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, :2])
    x   = np.linspace(-0.02, 0.02, 100)
    for r in results:
        pct_chg = [-r["mod_dur"]*dy + 0.5*r["convexity"]*dy**2
                   for dy in x]
        ax4.plot(x * 100, [p * 100 for p in pct_chg],
                 color=COLORS[r["type"]], linewidth=1.6,
                 label=r["name"], alpha=0.85)
    ax4.axhline(0, color=GREY, linewidth=0.7, linestyle="--")
    ax4.axvline(0, color=GREY, linewidth=0.7, linestyle="--")
    ax4.axvline(1,  color="#e74c3c", linewidth=0.8, linestyle=":", alpha=0.6)
    ax4.axvline(-1, color="#27ae60", linewidth=0.8, linestyle=":", alpha=0.6)
    ax4.set_xlabel("Yield Change (bps)", color="#555566", fontsize=8)
    ax4.set_ylabel("Price Change (%)", color="#555566", fontsize=8)
    ax4.legend(fontsize=7.5, loc="upper right", ncol=2)
    sax(ax4, "Price Sensitivity to Yield Changes (Duration + Convexity)")

    # ── 5. METRICS TABLE ────────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 2])
    ax5.set_facecolor("white"); ax5.axis("off")
    ax5.set_title("Portfolio Summary", color=NAVY,
                  fontsize=10, fontweight="bold", pad=9)

    rows = [["Bond", "YTM", "Dur", "Conv"]]
    for r in results:
        rows.append([
            r["name"].replace("US Treasury ", "UST ").replace(" Corp", ""),
            f"{r['ytm']:.2f}%",
            f"{r['mod_dur']:.1f}y",
            f"{r['convexity']:.1f}",
        ])

    tbl = ax5.table(cellText=rows[1:], colLabels=rows[0],
                    cellLoc="center", loc="center",
                    bbox=[0, 0.02, 1, 0.96])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)

    for (row, col), cell in tbl.get_celld().items():
        cell.set_facecolor("white" if row > 0 else "#f0f4f8")
        cell.set_edgecolor("#e8e8e8")
        if row == 0:
            cell.set_text_props(color=NAVY, fontweight="bold")
        else:
            cell.set_text_props(color="#333344")

    # Legend
    legend_items = list(COLORS.items())
    for i, (ltype, col) in enumerate(legend_items):
        ax5.add_patch(plt.Rectangle((0.0, -0.06 - i*0.032), 0.015, 0.02,
                                     transform=ax5.transAxes,
                                     color=col, clip_on=False))
        ax5.text(0.025, -0.05 - i*0.032, ltype,
                 transform=ax5.transAxes, fontsize=7.5, color="#555566")

    fig.text(0.5, 0.005,
             "The Meridian Playbook  |  themeridianplaybook.com  |  "
             "Not investment advice.",
             ha="center", color=GREY, fontsize=7.5)

    import os; os.makedirs("outputs", exist_ok=True)
    out = "outputs/bond_analytics.png"
    plt.savefig(out, dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print(f"  ✓ Saved → {out}")

# ── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    print("\n╔══════════════════════════════════════════════════╗")
    print("║   BOND ANALYTICS TOOL                            ║")
    print("║   The Meridian Playbook                          ║")
    print("╚══════════════════════════════════════════════════╝\n")

    print("📡 Fetching live yield curve from FRED...")
    curve = fetch_yield_curve()
    print(f"   2Y: {curve['2Y']:.2f}%  5Y: {curve['5Y']:.2f}%  "
          f"10Y: {curve['10Y']:.2f}%  30Y: {curve['30Y']:.2f}%")

    print("\n📐 Computing bond metrics...")
    for b in BONDS:
        ytm = compute_ytm(b)
        mac, mod, conv = compute_duration(b, ytm)
        print(f"   {b['name']:<22} YTM: {ytm:.2f}%  "
              f"Dur: {mod:.1f}y  Conv: {conv:.1f}")

    print("\n🖼️  Generating dashboard...")
    make_dashboard(BONDS, curve)
    print("\n✅  Done.\n")

if __name__ == "__main__":
    main()
