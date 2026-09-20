#!/usr/bin/env python3
"""Build the standalone dashboard: results.json + plotly.min.js -> one self-contained HTML file.

    python build_dashboard.py                      # -> uplift_dashboard.html
    python build_dashboard.py --results other.json --out my.html
"""
import argparse
import datetime as dt
import json
import os
from pathlib import Path

import plotly


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results.json")
    ap.add_argument("--template", default=str(Path(__file__).with_name("dashboard_template.html")))
    ap.add_argument("--out", default="uplift_dashboard.html")
    a = ap.parse_args()

    results = json.loads(Path(a.results).read_text())
    ds = results["datasets"]
    ds.setdefault("criteo", {
        "key": "criteo", "label": "Criteo", "status": "not_run",
        "reason": "The Criteo download host was unreachable from the environment that built this file.",
    })
    for key, label in (("synthetic", "Synthetic"), ("hillstrom", "Hillstrom")):
        ds.setdefault(key, {"key": key, "label": label, "status": "not_run", "reason": "This dataset has not been run."})
    results["built"] = dt.date.today().strftime("%b %d, %Y").replace(" 0", " ")

    plotly_js = Path(plotly.__file__).parent.joinpath("package_data", "plotly.min.js").read_text(encoding="utf-8")
    plotly_js = plotly_js.replace("</script", "<\\/script")
    payload = json.dumps(results, separators=(",", ":")).replace("</", "<\\/")

    html = Path(a.template).read_text(encoding="utf-8")
    assert "/*__RESULTS__*/" in html and "/*__PLOTLY__*/" in html
    html = html.replace("/*__RESULTS__*/", payload).replace("/*__PLOTLY__*/", plotly_js)
    Path(a.out).write_text(html, encoding="utf-8")
    print(f"wrote {a.out}  ({os.path.getsize(a.out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
