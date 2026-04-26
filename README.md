# Bond Analytics Tool

**YTM, modified duration, convexity and price sensitivity for a fixed income portfolio — with live yield curve from FRED.**

Built as part of [The Meridian Playbook](https://themeridianplaybook.com).

## What It Does

- Computes **YTM** via Newton-Raphson iteration
- Computes **Macaulay duration**, **modified duration** and **convexity**
- Models **price sensitivity** to ±200bps yield changes
- Fetches the **live US yield curve** (2Y, 5Y, 10Y, 30Y) from FRED
- Produces a clean white-background dashboard PNG

## Installation

```bash
pip install -r requirements.txt
python bond_analytics.py
```

## Customise

Edit the `BONDS` list to add your own instruments — face value, coupon, maturity, current price and type.

---

*Federico Bonessi — MSc Finance, IÉSEG School of Management*
*[themeridianplaybook.com](https://themeridianplaybook.com)*
