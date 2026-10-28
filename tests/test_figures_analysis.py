import numpy as np

from ymlab import efficiency_analysis as EA
from ymlab import figures as F


def test_published_decomposition_matches_ess_ratio():
    for d in EA.campaign_dirs():
        rows = EA.paired_decomposition(EA.load_records(d))
        assert rows
        for r in rows:
            assert abs(r["ratio"] - r["ratio_check"]) < 1e-9 * max(1, r["ratio"])


def test_mean_of_ratios_differs_from_ratio_of_means_generally():
    d = EA.campaign_dirs()[-1]
    rec = EA.load_records(d)
    agg = EA.aggregate(EA.paired_decomposition(rec))
    rom = EA.ratio_of_means(rec)
    assert any(abs(a["mean_of_ratios"] - rom[(a["beta"], a["sweeps"])]) > 1e-3 for a in agg)


def test_figures_generate(tmp_path):
    out = F.generate_static(tmp_path)
    assert "gain_vs_cost" in out
    for paths in out.values():
        for p in paths:
            assert p.exists() and p.stat().st_size > 1000
