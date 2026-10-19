# Pre-specified decision rules

Each file was written before the result it governs; the date and the data that existed at the time are stated at
the top of every section. Later changes are numbered amendments with their own dates, each written before the data it
concerns.

| File | Covers |
|---|---|
| `hw_prereg.md` | Dhaka–Chattogram highway: sampling and split, labels, training candidates, the transfer test and the rule that decides the map (≥ 70 % gaps; 40–70 % trees per km; < 40 % no map), usability check, satellite canopy, heat ranking (amendments 1–8) |
| `mly_prereg.md` | Mapillary photos of Pasadena: settings searched on validation, pass mark against Google panoramas, result |
| `vlm_prereg.md` | vision–language second look: model, prompts, threshold rule, pass rule (+5 points, 80 % of free spots kept, interval above zero), results with both detectors |
| `e7_prereg.md` | monocular depth comparison (Pasadena and Bangladesh field photos) |

Rules for the Bangladesh split, the Pasadena merge and gap settings and the trunk-point review were fixed in the
project plan before the corresponding runs; the settings searched and chosen are listed in the Methods and in
`results/` (for example `pas_e2_dedup_comparison.csv`, `pas_student_decision.json`).
