## Verification of prior issues

**1. Pixel vs. range ratio inconsistency — resolved.** The abstract's "23 vs 76 px" now matches the blind-click comparison in §5.1: "the predicted point of run 1... lies a median of 23.0 px away... and the box bottom 75.6 px... on the 94 trees" — and the separate proposal-accepted comparison (20.6–22.1 vs 52.6–59.8 px, different sample/purpose) is clearly distinguished rather than conflated. Ratio (23/76≈0.30) matches "a third."

**2. 'Near the two cities' overclaim — resolved.** Table 5's caveat ("Two of the top fifteen (km 42 and 87) lie in the rural middle of the route, where they rank high because no tree was detected rather than because they are hot") now explicitly flags the 2/15 exceptions, leaving 13/15 genuinely near-city, which supports "most of them near the two cities" in the Conclusion.

**3. Undisclosed second VLM threshold — resolved.** §3.5 and §4.5 now disclose it as a named, dated, secondary analysis: "A second operating point... was added during revision; its rule was written down before it was computed, but after the results of the first threshold were known, so it is reported as a secondary, exploratory setting."

## New issues

**Major — contradictory sample size for the field-photo validation set.** §3.3 and Table 2 state the Bangladesh field test uses **26 trees**: "Of 26 trees, 22 have a trunk base visible in the photo; the other 4 are reported but not scored" (§3.3); Table 2: "26 trees photographed with phones." But Fig. 2's caption describes **19 field photos**: "GPS-tagged field photos (Dhaka) ... (star; 10 of the 19 field photos carry a GPS position)." Trees ≠ photos is possible, but no text reconciles 26 trees with only 19 photos, and §5.3's "10 sites" matches the Fig. 2 number, not the §3.3 number — readers cannot tell whether the geolocation validation set is 19 or 26.

**Minor numeric contradiction — arithmetic slip in the keypoint ablation.** §5.2: "under strict scoring 0.562 against 0.584 (difference −0.021, −0.029 to −0.014)." 0.562 − 0.584 = −0.022, not −0.021; the reported point estimate doesn't match the two reported scores it's derived from.

## Verdict

**Major revision.** The three previously raised issues are genuinely fixed. The new 26-vs-19 discrepancy in the field-validation sample size is a basic data-integrity error that must be corrected before acceptance; the −0.021/−0.022 slip is trivial but should be fixed in the same pass.