from __future__ import annotations

import pandas as pd

from groundedgen import finish, utils


def _numbers(extractor="g/siglip"):
    ci = {"estimate": 0.1, "low": 0.0, "high": 0.2}
    rates = {f"{c}|{a}": {"HR": ci, "EP": ci, "GD": ci, "GF": ci, "extraction_share": ci,
                          "leakage": {"estimate": None, "low": None, "high": None}, "inflation": ci}
             for c in ("A|llava-1.5-7b-hf|", "B|Qwen2.5-7B-Instruct|siglip")
             for a in ("category", "color", "material")}
    return {"split": "test", "primary_extractor": extractor, "rates": rates,
            "rq1_gd_diff_B_minus_C": {"Qwen|color": ci},
            "validation": {"apg_fallback": {"q|s": {"n": 200, "rate": 0.05}}},
            "ucr": {"A|llava-1.5-7b-hf|": {"n": 10, "ucr": 1.2, "subjective_per_description": 0.1,
                                           "parse_error_rate": 0.0}}}


def test_summary_states_headline_numbers_and_pending_parts():
    cfg = utils.load_config()
    tests = pd.DataFrame([{"llm": "Qwen/Qwen2.5-7B-Instruct", "attribute": "color", "left": "B",
                           "right": "B-Abs", "rate_diff": 0.15, "p_holm": 0.01, "reject": True,
                           "n10": 5, "n01": 0}])
    text = finish.summary_markdown(cfg, _numbers(), None, tests)
    assert "| A | 0.10 [0.00, 0.20]" in text and "| C | n/a" in text
    assert "leakage n/a" in text  # undefined rates do not crash
    assert "significant; 5 vs 0" in text and "UCR 1.20" in text
    assert "Judge validation: pending" in text and "Pending (ratings" in text
    assert "## Deviations" in text


def test_primary_keys_match_numbers_json_format():
    cfg = utils.load_config()
    keys = finish.primary_keys(cfg, "google/siglip-so400m-patch14-384")
    assert keys["A"] == "A|llava-1.5-7b-hf|"
    assert keys["APG"] == "APG|Qwen2.5-7B-Instruct|siglip-so400m-patch14-384"
