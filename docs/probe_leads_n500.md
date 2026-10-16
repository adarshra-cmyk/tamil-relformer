# Plan-named secondary analyses, n=500 (exploratory, 2026-10-12)

Source: `python scripts/probe_leads.py runs/clean_v1 --n 500` (5 seeds, paired seed+item bootstrap).
Plan section 10: if morphology does not improve F1, examine low-data settings, case-signalled
relations and analyzer quality. Plan section 8: macro-F1 and per-relation F1. All rows are in
`runs/probe_leads_n500.json`; nothing below is a confirmed result. Done after H2 was registered
(docs/preregistration_h2.md, 14:35).

| analysis | E1 | control | E2a | E2b | E2c |
|---|---|---|---|---|---|
| micro, all | +0.009 | +0.007 | +0.004 | +0.008 | +0.007 |
| macro, all | +0.006 | +0.004 | +0.004 | +0.004 | +0.006 |
| micro, case-signalled relations (train-defined: COUNTRY, DEATH_PLACE, EDUCATED_AT; 102 items) | +0.002 | -0.006 | -0.019 | -0.029 | -0.020 |
| micro, Stanza obj case right (813) | +0.009 | +0.006 | +0.006 | +0.009 | +0.010 |
| micro, Stanza obj case wrong (74) | +0.014 | +0.012 | -0.015 | -0.003 | -0.020 |
| right minus wrong (interaction) | -0.005 | -0.006 | +0.021 | +0.012 | +0.030 |

No interval in this table excludes zero (interaction CIs: E2a [-0.013,+0.058], E2b [-0.030,+0.053],
E2c [-0.015,+0.080]).

**Lead (preliminary):** case features help only when the analyzer's case is right. The interaction
is positive for all three case configs and about zero for both non-case configs (E1, surface-copy
control). Caveats: 74 wrong-case items; 35 of them are Stanza "no case" vs gold OTHER.
Prospective test = H2 (matcher and gold case, 5 seeds), moved to the front of the queue.

**Not a lead:** PER:RESIDENCE F1, E2c +0.100 [+0.014,+0.190], also appears in the surface-copy
control (+0.079 [+0.010,+0.160]), so it is not a morphology effect. 2 of 70 per-relation tests
clear zero, about what chance gives at 5%.

**Against the plan hypothesis:** on train-defined case-signalled relations every case config is
below E0.
