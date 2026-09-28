"""Behavioral probes also runnable against 86c7aef without any new API.

These assert the NEW explicit scope specification, not a discovered old policy.
Set TOOLKIT_CASES_SOURCE to the baseline examples/model_review_demo.py to reproduce.
"""
import importlib.util
import os
from pathlib import Path
import sys
import pytest


@pytest.mark.parametrize("domain,case", [("lending", "SYN-003"), ("insurance", "INS-012")])
def test_common_favorable_restriction_cannot_bypass_review(domain, case):
    path=Path(os.environ.get("TOOLKIT_CASES_SOURCE",Path(__file__).resolve().parents[1]/"examples/model_review_demo.py"))
    spec=importlib.util.spec_from_file_location("scope_probe",path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    scenario=next(s for s in module.SCENARIOS if s.key==domain)
    row=next(r for r in module.run_scenario(scenario)["rows"] if r["record_ref"]==case)
    assert row["route"] != "automated", "New common-scope specification requires review regardless of confidence"
