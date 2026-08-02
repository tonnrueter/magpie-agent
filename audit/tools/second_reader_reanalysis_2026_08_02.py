#!/usr/bin/env python3
"""Second-reader reanalysis of arena 1A and the R55/R60 exposure gradient (2026-08-02).

Produced by the independent review of the 2026-08-02 handoff briefing. Two analyses,
each self-controlled by first reproducing the published numbers from the raw artifacts:

1. TRAP-LEVEL CLASS EFFECT (arena 1A). The published class effect
   (attribution_role 28.1% vs 4.2%, Fisher p = 0.0012) is computed at ANSWER level,
   but class varies only BETWEEN traps: the attribution class is exactly the 2 seeded
   traps T5+T6 (both from commit ec119a9 -- see arena_1a_prereg.md trap table), and
   each trap's answers share the trap. The honest unit of analysis is the trap.
   The 2-of-8 permutation test here gives p ~= 0.036 -- evidence, but ~30x weaker
   than the answer-level p, and class-vs-item is unidentifiable at 2 traps/class.

2. COMPOSITION STANDARDIZATION (exposure gradient). R55's audited hubs and R60's
   carbon_balance make very different claim-type mixes (hubs: 17.9% citation claims,
   the lowest-yield class; carbon_balance: 1.0%). Reweighting R55's per-class rates
   to carbon_balance's claim mix predicts ~7.2%, not 5.6% -- composition alone
   explains ~a quarter of the exposure gap. CAVEAT: uses R55's per-class rates,
   which are SUSPECT (same two-agent claim/defect classification mechanism whose
   disagreement voided R60's per-class rates). A bound, not a measurement.

Inputs (all repo artifacts, no LLM output re-trusted):
  audit/data/arena_1a_regrade.json                      -- per-answer verdicts + trap key
  audit/integrated/depth_residual_density.json          -- R55 per-class numerators/denominators
  audit/archive/rounds/round60_depth/_checkpoint_audits.json -- R60 carbon_balance claim ledger
"""

import itertools
import json
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# The attribution-class traps, per the arena 1A prereg trap table (T5: vm_prod
# consumer/producer inversion; T6: vm_land_forestry declaration attribution).
# Both were seeded from commit ec119a9 by the hypothesis framer.
ATTRIBUTION_TRAPS = {"T5", "T6"}


def fisher_two_sided(a, b, c, d):
    """Fisher exact test, two-sided by summing hypergeometric probs <= p(observed)."""
    n = a + b + c + d
    row1, col1 = a + b, a + c
    def p(x):
        return comb(col1, x) * comb(n - col1, row1 - x) / comb(n, row1)
    p_obs = p(a)
    lo, hi = max(0, row1 + col1 - n), min(row1, col1)
    return sum(p(x) for x in range(lo, hi + 1) if p(x) <= p_obs + 1e-12)


def trap_level_class_effect():
    d = json.loads((ROOT / "audit/data/arena_1a_regrade.json").read_text())
    key, cons = d["key"], d["consensus"]

    traps = {}
    for aid, meta in key.items():
        v = cons.get(aid)
        if v is None:
            continue
        t = traps.setdefault(meta["trap"], {"n": 0, "narrow": 0})
        t["n"] += 1
        t["narrow"] += (v == "ASSERTS_FALSEHOOD")

    att_k = sum(traps[t]["narrow"] for t in ATTRIBUTION_TRAPS)
    att_n = sum(traps[t]["n"] for t in ATTRIBUTION_TRAPS)
    oth_k = sum(v["narrow"] for t, v in traps.items() if t not in ATTRIBUTION_TRAPS)
    oth_n = sum(v["n"] for t, v in traps.items() if t not in ATTRIBUTION_TRAPS)

    # Positive control: reproduce the published answer-level numbers before
    # computing anything new. (Published: 28.1% vs 4.2%, Fisher p = 0.0012 on
    # n=103 scored; the consensus dict holds 104 entries -- the 1-answer
    # denominator difference does not move any figure below at 3 decimals.)
    assert att_k == 9 and att_n == 32, (att_k, att_n)
    assert sum(v["narrow"] for v in traps.values()) == 12
    p_answer = fisher_two_sided(att_k, att_n - att_k, oth_k, oth_n - oth_k)
    assert abs(p_answer - 0.0012) < 0.0002, p_answer

    # Trap-level permutation: under the null (no class effect), any 2 of the 8
    # traps could carry the 'attribution' label. Statistic: pooled narrow-rate
    # difference. One-sided (the pre-stated direction: attribution higher).
    obs = att_k / att_n - oth_k / oth_n
    hits, total = 0, 0
    for pair in itertools.combinations(sorted(traps), 2):
        k1 = sum(traps[t]["narrow"] for t in pair)
        n1 = sum(traps[t]["n"] for t in pair)
        k0 = sum(v["narrow"] for t, v in traps.items() if t not in pair)
        n0 = sum(v["n"] for t, v in traps.items() if t not in pair)
        total += 1
        if k1 / n1 - k0 / n0 >= obs - 1e-12:
            hits += 1

    print("== Arena 1A class effect, re-analysed at the trap level ==")
    print(f"answer-level (as published): {att_k}/{att_n} vs {oth_k}/{oth_n}, "
          f"Fisher p = {p_answer:.4f}  [reproduced]")
    print(f"trap-level permutation (2-of-8): p = {hits}/{total} = {hits/total:.3f}")
    per_trap = {t: f"{v['narrow']}/{v['n']}" for t, v in sorted(traps.items())}
    print(f"per-trap narrow counts: {per_trap}")
    return hits / total


def composition_standardization():
    r55 = json.loads((ROOT / "audit/integrated/depth_residual_density.json").read_text())
    r60 = json.loads(
        (ROOT / "audit/archive/rounds/round60_depth/_checkpoint_audits.json").read_text())

    denom = r55["matrix"]["pooled"]["denominator"]["by_class"]
    by_class = r55["matrix"]["pooled"]["by_class"]
    numer = {c: sum(sev.values()) for c, sev in by_class.items()}
    cb = r60["ledgers"]["carbon_balance_conservation"]["by_class"]

    # Positive controls: totals must match the published R55 record (28/498)
    # and the carbon_balance ledger total (210).
    assert sum(numer.values()) == 28 and sum(denom.values()) == 498
    assert sum(cb.values()) == 210

    r55_rate = sum(numer.values()) / sum(denom.values())
    # Directly standardized rate: R55 per-class rates weighted by the
    # carbon_balance claim mix. Classes absent from R55's taxonomy keep their
    # R55 rate where defined; 'other' exists in both.
    expected = sum(cb[c] * (numer[c] / denom[c]) for c in cb) / sum(cb.values())

    print("\n== Exposure gradient: composition standardization ==")
    print(f"R55 pooled rate (own mix):                  {r55_rate:.1%}")
    print(f"R55 rates reweighted to carbon_balance mix: {expected:.1%}")
    print(f"carbon_balance observed (code-fact):        43/210 = {43/210:.1%}")
    for c in sorted(cb, key=lambda c: -cb[c]):
        print(f"  {c:22s} R55 mix {denom[c]/498:5.1%}  cb mix {cb[c]/210:5.1%}  "
              f"R55 rate {numer[c]/denom[c]:5.1%}")
    print("CAVEAT: R55 per-class rates are SUSPECT (two-agent classification, "
          "see the 2026-08-02 retraction); this is a bound, not a measurement.")
    return expected


if __name__ == "__main__":
    trap_level_class_effect()
    composition_standardization()
