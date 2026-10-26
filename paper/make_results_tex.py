"""Results section (LaTeX) with every number read from the result files (audited values from audit/audit_values.json). Highway subsection added when its results exist.
Usage: python make_results_tex.py  ->  manuscript_v2/sec_results.tex"""
import json, os
BL = json.load(open('blind_trunk_analysis.json'))
KA = json.load(open('pas_trunk_ablation.json'))
KV = {r['variant']: r['AP_4m'] for r in KA['val_best']}
import numpy as np, pandas as pd

def j(f): return json.load(open(f))
U = pd.read_csv('detector_unified_coco_scores.csv', index_col=0)
S = pd.read_csv('detector_unified_summary.csv').set_index('model')
NAMES = {'frcnn (native COCO scorer)': ('Faster R-CNN (ResNet-50 FPN v2)$^{a}$', 832), 'posel': ('YOLO11l-pose', 1280), 'yolo832': ('YOLO11m', 832),
         'yolo11m': ('YOLO11m', 1024), 'pose': ('YOLO11m-pose (main model)', 1024), 'rtdetr_l': ('RT-DETR-L', 832),
         'yolov4': ('YOLOv4 (original configuration)$^{b}$', 832), 'gdino_zs': ('Grounding DINO-B, zero-shot', None)}
rows = []
for k, (nm, px) in NAMES.items():
    r = S.loc[k]; sd = '' if pd.isna(r.AP50_sd) else f' $\\pm$ {r.AP50_sd:.3f}'
    rows.append(f"{nm} & {px if px else '--'} & {int(r.n)} & {r.AP50:.3f}{sd} & {r.AP75:.3f} & {r.AP50_95:.3f} \\\\")
hwe = pd.read_csv('detector_hw_effect.csv').set_index('run')
T = pd.read_csv('trunk_point_main_model_3runs.csv')
s = j('pas_student_decision.json'); te = s['test_student']
e7 = j('pas_e7_result.json'); e7s = j('pas_e7_scaled_result.json'); mly = j('mly_test_results.json')['results']
E3 = pd.read_csv('e3_summary_v2.csv'); g = pd.read_csv('field_ground_truth_feb.csv'); B = pd.read_csv('e7_bd_summary.csv')
GV = pd.read_csv('pas_student_gap_validation_test.csv'); CS = pd.read_csv('pas_student_gap_city_sites_test.csv')
DN = pd.read_csv('pas_density_validation_student.csv').set_index('metric'); V1 = j('vlm_result.json'); V2 = j('vlm2_result.json')
DB = pd.read_csv('pas_detect_by_dbh.csv').set_index('dbh')
def gv(S_, base=False):
    lab = f'{S_} (baseline: no trees detected)' if base else str(S_)
    return GV[GV.S.astype(str) == lab].iloc[0]
g12, b12 = gv(12), gv(12, True)
g10, b10, g15, b15 = gv(10), gv(10, True), gv(15), gv(15, True)
# audited values (audit/audit_values.json, written by the 19 Oct 2026 audit; see audit_report.md)
A = j('audit/audit_values.json')
OW, LK, YV, TK = A['old_weights'], A['old_split_leakage'], A['yolov4_retrain'], A['trunk_key']
TA = pd.DataFrame(A['trunk_anchoring']); CO = TA[TA.subset.str.startswith('corrected')]
REC, FIN, FIRST = A['e2_point']['records'], A['e2_point']['retrained (final)'], A['e2_point']['first detector, validation-chosen merge']
U12 = json.load(open('u1_u2_keypoint.json')); U3 = json.load(open('u3_density_estimators.json')); U4 = json.load(open('u4_vlm_high_purity.json'))
U5 = json.load(open('u5_trip_accumulation.json')); U6 = json.load(open('u6_sentinel2.json')); U7 = json.load(open('u7_heat_priority.json'))
_r = U12['s1_reviewed']
UX = dict(rprms=U12['repeatability']['rmse_px'], rpp95=U12['repeatability']['p95_px'], b0n=U12['s0_blind']['n'], b0k=U12['s0_blind']['keypoint_median_px'],
          b0b=U12['s0_blind']['boxbottom_median_px'], b0c=100 * U12['s0_blind']['keypoint_closer'],
          rb15=_r['range_err_h1.5_D15']['boxbottom_median_m'], rb30=_r['range_err_h3.0_D15']['boxbottom_median_m'],
          rk15=_r['range_err_h1.5_D15']['keypoint_median_m'], rk30=_r['range_err_h3.0_D15']['keypoint_median_m'],
          xg15=100 * _r['range_err_h1.5_D15']['share_gt4m_boxbottom'], xg30=100 * _r['range_err_h3.0_D15']['share_gt4m_boxbottom'],
          gk15=100 * _r['range_err_h1.5_D15']['share_gt4m_keypoint'], gk30=100 * _r['range_err_h3.0_D15']['share_gt4m_keypoint'],
          u4p=100 * U4['test']['precision'], u4g=U4['test']['gain_pp'], u4lo=U4['test']['ci_pp'][0], u4hi=U4['test']['ci_pp'][1], u4k=100 * U4['test']['kept'],
          u3w=U3['factors']['window'], u3h0=U3['as_detected']['hottest10_median'], u3h1=U3['protocol_div_recall']['hottest10_median'], u3h2=U3['window_adjusted']['hottest10_median'],
          u3c0=U3['as_detected']['coolest10_median'], u3c1=U3['protocol_div_recall']['coolest10_median'], u3c2=U3['window_adjusted']['coolest10_median'],
          u5n4=U5['n_stretches'], u5k1=U5['k1']['median_rho'], u5k4=U5['k4']['median_rho'], u5n2=U5['exploratory_min2']['n_stretches'],
          u5a=U5['exploratory_min2']['k1']['median_rho'], u5b=U5['exploratory_min2']['k2']['median_rho'], u5all=U5['exploratory_min2']['all_trips_rho'],
          u6r=U6['share']['rho'], u6p=U6['share']['p'], u6n=U6['n'], u6rm=U6['median']['rho'], u6pm=U6['median']['p'], u6rc=U6['canopy_vs_share_rho'],
          u7d=U7['lst_vs_dcity'], u7t=U7['partial_lst_trees_given_dcity'][0], u7tp=U7['partial_lst_trees_given_dcity'][1],
          u7c=U7['partial_lst_canopy_given_dcity'][0], u7cp=U7['partial_lst_canopy_given_dcity'][1], u7o=U7['top10_overlap_with_10_hottest'],
          u7s=sum(r['share_in_top15'] >= 0.75 for r in U7['top15']))
PT = json.load(open('pas_table.json'))   # one paired 400 m block bootstrap for every Pasadena map (src/pas_table_bootstrap.py)
_pb = lambda k: {r: {'ci95': PT['ci95'][k][r]} for r in ('official', 'strict')} | {'n_blocks': PT['n_blocks']}
BF, B1 = _pb('retrained_box'), _pb('first_detector')
KD = PT['paired_differences']['pose_keypoint_K_minus_pose_box_bottom_B']
ABOVE = PT['share_above_branson_0581']['retrained_box']; PT_point = lambda k: PT['point'][k]['AP4']; E3A, SC = A['e3'], A['second_check']
cs = CS[CS.S == 12].set_index('site_type')
vac, nor = cs.loc['city vacant planting sites'], cs.loc['city "no room for replant" sites']
e3a = E3.iloc[0]; e3l = E3.iloc[3]; bda, bdl = B.iloc[2], B.iloc[3]

assert abs(te['AP_4m'] - FIN['AP4_official']) < 1e-3, (te['AP_4m'], FIN['AP4_official'])
assert int(e3a.n) == E3A['n'] and abs(e3a.median_abs_m - E3A['ours_fixed_h1.5']['median_abs_m']) < 6e-3 and abs(e3l.median_abs_m - E3A['ours_loo']['median_abs_m']) < 6e-3
assert abs(g.diff_m.abs().median() - SC['gps_vs_aruler']['median_abs_m']) < 0.06
X1 = dict(v700=OW['old_validation_random700']['AP50'], twin=LK['valid_twin_in_train_pct'], t270=OW['released_test_set_270']['AP50'],
          v4it=f"{YV['last_iteration']:,}", v4max=f"{YV['cfg_max_batches']:,}", v4min=YV['train_minutes'],
          kK=KA['test']['K']['AP_4m'], kB=KA['test']['B']['AP_4m'], vK=KV['K'], vB=KV['B'], kd=KA['paired_bootstrap_K_minus_B']['official']['point'], klo=KD['official']['ci95'][0], khi=KD['official']['ci95'][1], kn=PT['n_rep'], kslo=KD['strict']['ci95'][0], bbd=KA['test']['B']['AP_4m']-PT_point('retrained_box'), bblo=PT['paired_differences']['pose_box_bottom_B_minus_retrained_box']['official']['ci95'][0], bbhi=PT['paired_differences']['pose_box_bottom_B_minus_retrained_box']['official']['ci95'][1], kshi=KD['strict']['ci95'][1], ksd=KA['test']['K']['AP4_strict']-KA['test']['B']['AP4_strict'], sK=KA['test']['K']['AP4_strict'], sB=KA['test']['B']['AP4_strict'], mK=KA['test']['K']['gate_median_matched10'], mB=KA['test']['B']['gate_median_matched10'], nK=f"{KA['test']['K']['n_trees']:,}", nB=f"{KA['test']['B']['n_trees']:,}", rK=KA['test']['K']['max_recall'], rB=KA['test']['B']['max_recall'], tkacc=TK['accepted_pct'], bln=200, blm=BL['tasfik']['all']['n'], blk=BL['tasfik']['all']['keypoint_median_px'], blkr=BL['tasfik']['all']['keypoint_rel_pct'], blb=BL['tasfik']['all']['boxbottom_median_px'], blbr=BL['tasfik']['all']['boxbottom_rel_pct'], blc=100*BL['tasfik']['all']['keypoint_closer_share'], blvk=BL['tasfik']['visible']['keypoint_median_px'], blvb=BL['tasfik']['visible']['boxbottom_median_px'], blvn=BL['tasfik']['visible']['n'], blhk=BL['tasfik']['hidden']['keypoint_median_px'], blhb=BL['tasfik']['hidden']['boxbottom_median_px'], blhn=BL['tasfik']['hidden']['n'], blrep=BL['same_annotator_first_review_vs_blind']['median_px'], conlo=int(CO.n.min()), conhi=int(CO.n.max()), cpl=CO.pose_median_px.min(), cph=CO.pose_median_px.max(),
          cbl=CO.box_bottom_median_px.min(), cbh=CO.box_bottom_median_px.max(), cbt=(f'{CO.pose_better_than_box_pct.min():.0f}' if round(CO.pose_better_than_box_pct.min()) == round(CO.pose_better_than_box_pct.max())
               else f'{CO.pose_better_than_box_pct.min():.0f}--{CO.pose_better_than_box_pct.max():.0f}'),
          nrec=f"{REC['test_records']:,}", ntree=f"{REC['real_trees']:,}", nnon=f"{REC['non_tree_records']:,}", ndiff=f"{REC['difficult_records']:,}",
          nscored=f"{REC['test_records'] - REC['difficult_records']:,}", aplo=BF['official']['ci95'][0], aphi=BF['official']['ci95'][1], nblk=BF['n_blocks'],
          aps=FIN['AP4_strict_all_real_trees'], apslo=BF['strict']['ci95'][0], apshi=BF['strict']['ci95'][1], recs=FIN['max_recall_strict'],
          n11=f"{FIN['one_to_one_matches_official']:,}", med11=FIN['median_error_matched_official'], above=ABOVE * 100,
          ap0=FIRST['AP4_official'], ap0lo=B1['official']['ci95'][0], ap0hi=B1['official']['ci95'][1],
          nfield=E3A['field_clicks'], arlo=E3A['aruler_range_m'][0], arhi=E3A['aruler_range_m'][1],
          e3alo=E3A['ours_fixed_h1.5']['ci95'][0], e3ahi=E3A['ours_fixed_h1.5']['ci95'][1], e3llo=E3A['ours_loo']['ci95'][0], e3lhi=E3A['ours_loo']['ci95'][1],
          ratio=E3A['ratio_ours_median'], wl=E3A['wilcoxon_p_loo'], wu=E3A['wilcoxon_p_uncalibrated'], ngps=SC['gps_vs_aruler']['n'],
          g10=g10.precision * 100, b10=b10.precision * 100, g15=g15.precision * 100, b15=b15.precision * 100,
          qb=SC['vlm_question_b']['auc'], qblo=SC['vlm_question_b']['ci'][0], qbhi=SC['vlm_question_b']['ci'][1])

tex = r"""%% ---------------------------------------------------------------
\section{Results}\label{sec:results}
%% ---------------------------------------------------------------
\subsection{Tree detection}\label{sec:e1}
Table~\ref{tab:detectors} compares detectors on the route-grouped Bangladesh test split, with every model scored by the
COCO evaluator \citep{lin2014coco}, through a shared script on exported predictions for all models except Faster R-CNN, which was scored by the same evaluator inside its training job (two or three runs per trained modern detector). The modern detectors reach AP$_{50}$ between %(lo).3f and %(hi).3f. Repeated runs of one architecture
with different random seeds differ by a standard deviation of %(sdlo).3f--%(sdhi).3f, so the differences between
architectures are of the same size as the run-to-run variation. The original configuration of the earlier version of this
work (YOLOv4), retrained on this split, reaches %(v4).3f. The previously reported AP$_{50}$ of 90.7\%% is reproduced
(%(v700).1f\%% on a random sample of 700 of its images) on the validation list used during training, which also selected the reported weights; %(twin).1f\%% of its
images show a scene that is also in the training list, as the original or as a flipped or blurred copy. On the earlier
work's held-out test set the same weights reach %(t270).1f\%%. A zero-shot open-vocabulary detector (Grounding DINO) reaches
%(gd).3f. Adding the highway training photos (Section~\ref{sec:e6}) did not reduce performance on this split
(AP$_{50}$ %(hwa).3f for configuration A and %(hwb).3f $\pm$ %(hwbsd).3f over three runs for configuration B, against
%(pose).3f $\pm$ %(posesd).3f for the same architecture trained without them).

\begin{table}[!htb]\centering\small
\caption{Detection on the Bangladesh test split (276 images, 789 trees; route-grouped, no near-duplicate of a test image in
training). Mean over runs with different seeds $\pm$ standard deviation. $^{a}$Scored with the same COCO evaluator code inside its training job, on the same test images and with the same IoU thresholds, rather than by the shared script on exported predictions.
$^{b}$Retraining stopped at iteration %(v4it)s of the configured %(v4max)s (training budget of %(v4min).0f\,min in one
Kaggle session); the original, fully trained weights reach %(t270).1f\%% AP$_{50}$ on the earlier work's held-out test set.}\label{tab:detectors}
\begin{tabular}{lccccc}\toprule
Detector & Input (px) & Runs & AP$_{50}$ & AP$_{75}$ & AP$_{50:95}$\\\midrule
%(rows)s
\bottomrule\end{tabular}\end{table}

The main model is YOLO11m-pose because it also predicts the trunk--ground contact point.
Against the reviewed trunk-base points of the test trees, the predicted point lies a median of %(tp)s\,px away
in the three runs, against %(bb)s\,px for the bottom-centre of the box on the same trees. Because %(tkacc).1f\%% of these
reference points are automatic proposals that the annotator accepted unchanged, which could favour a model trained on
proposals of the same kind, the annotator also clicked the trunk base of %(bln)d randomly drawn test trees from scratch,
with no proposal shown. Against these points, the predicted point of run~1, compared on the same trees, lies a median of %(blk).1f\,px away
(%(blkr).1f\%% of the box height) and the box bottom %(blb).1f\,px (%(blbr).1f\%%) on the %(blm)d trees that the run detects (the other clicked trees have no matching detection and cannot be scored); the predicted
point is the closer of the two for %(blc).0f\%% of them (Wilcoxon $p<10^{-10}$), for visible bases (%(blvk).1f against %(blvb).1f\,px, $n=$%(blvn)d) and
for hidden ones (%(blhk).1f against %(blhb).1f\,px, $n=$%(blhn)d). The same annotator's blind clicks lie a median of %(blrep).1f\,px from
his earlier, proposal-assisted points (19.8\,px for hidden bases; root-mean-square %(rprms).1f\,px, 95th percentile %(rpp95).1f\,px), an estimate of the annotation
noise; a second annotator was not available. The second training run with saved predictions gives the same picture on its %(b0n)d matched
trees (%(b0k).1f against %(b0b).1f\,px; closer for %(b0c).0f\%%).
Translated into range for a level camera (Eq.~\eqref{eq:err}, assuming a 90$^\circ$ horizontal field of view because the
screenshots carry no focal length), only the vertical component of the error matters; its median for the box bottom on the reviewed test trees corresponds to
%(rb15).2f\,m at 15\,m from a camera 1.5\,m high and %(rb30).2f\,m from a camera 3\,m high; for the keypoint the values are %(rk15).2f\,m
and %(rk30).2f\,m. At 15\,m the box-bottom error exceeds the 4\,m matching tolerance for %(xg15).0f\%% of trees at 1.5\,m and %(xg30).0f\%% at 3\,m
(keypoint: %(gk15).0f\%% and %(gk30).0f\%%).

Figure~\ref{fig:trunk} shows the full distributions: the predicted point is closer than the box bottom over the whole
range, and the advantage remains when the distances are scaled by the size of the tree in the image and for hidden bases
whose reference position the annotator placed. Figure~\ref{fig:examples} shows typical successes and the three main
failure modes: small, distant trees; continuous rows of canopy that the detector and the annotator divide differently;
and trunk bases hidden by vegetation.

\input{sections/fig_examples}
\input{sections/fig_trunk}
\input{sections/fig_validation}
\subsection{Map positions in Pasadena}\label{sec:e2}
The Pasadena Urban Trees test area holds %(nrec)s inventory records: %(ntree)s trees and %(nnon)s planting sites, stumps
and other non-tree records. Scored with the dataset's own evaluator (a detection is correct if it lies within 4\,m of a
record; the %(ndiff)s records flagged as difficult are ignored, leaving %(nscored)s), the pipeline reaches AP@4\,m %(ap).3f
(95\%% interval %(aplo).3f--%(aphi).3f from a spatial block bootstrap with 400\,m blocks, %(nblk)d blocks) and a maximum
recall of %(rec).3f (Fig.~\ref{fig:validation}a,b). Under strict scoring, in which every inventory tree counts and the
non-tree records are removed, AP@4\,m is %(aps).3f (%(apslo).3f--%(apshi).3f) and the maximum recall %(recs).3f. For mapped trees
with a record within 10\,m, the distance to the nearest record has a median of %(med).2f\,m (90th percentile %(p90).2f\,m);
for the evaluator's one-to-one matches within 4\,m (%(n11)s pairs) the median error is %(med11).2f\,m. The result is comparable
to or higher than the street-view-only result reported for this benchmark, 0.581 \citep{wegner2016cataloging,branson2018google}:
%(above).0f\%% of the bootstrap resamples exceed it. The full model of those studies, which adds aerial images and map data,
reaches 0.706. These positions use the bottom-centre of each box as the ground point.

In the pre-specified keypoint ablation (Section~\ref{sec:det}), the keypoint variant K reaches AP@4\,m %(kK).3f on the
test area and the box-bottom variant B of the same detections %(kB).3f (validation: %(vK).3f and %(vB).3f); the paired
difference is $%(kd).3f$ (95\%% interval $%(klo).3f$ to $%(khi)+.3f$, %(kn)d paired block resamples), and under strict scoring
%(sK).4f against %(sB).4f (difference $%(ksd).3f$, $%(kslo).3f$ to $%(kshi).3f$). By the rule fixed in advance, the keypoint does not improve the map score on these panoramas.
For mapped trees with a record within 10\,m, the median distance to the nearest record is %(mK).2f\,m for K and %(mB).2f\,m
for B; K keeps fewer trees (%(nK)s against %(nB)s) at a lower maximum recall (%(rK).3f against %(rB).3f). The box bottoms of this keypoint model score higher than the retrained box detector used for the main results
(paired difference %(bbd)+.3f, 95\%% interval %(bblo)+.3f to %(bbhi)+.3f). The main results keep the detector that was fixed
before this test, so they are, if anything, conservative.

Table~\ref{tab:pasadena} lists all Pasadena configurations.

\input{sections/tab_pasadena}

The detector used for these results was retrained on inventory-verified pseudo-labels from the training area
(Section~\ref{sec:det}). It was adopted because it scored higher on the validation area than the first detector
(AP@4\,m %(vstu).3f vs.\ %(vcur).3f, each with its merge setting chosen on validation); on the test area the first detector
reaches %(ap0).3f (%(ap0lo).3f--%(ap0hi).3f). Each pipeline version was scored on the test area once, but the test area was
scored for several versions (the first detector with its initial and its validation-chosen merge setting, the retrained
detector, the depth model and the Mapillary photos), and the decision to retrain
followed diagnostics that included the test area. The validation gain is therefore the independent evidence for retraining.

Monocular depth is the usual way to obtain range in street-image tree mapping \citep{lumnitz2021mapping,dong2025stvmnet,lake2026continental,carnot2026mapanything}.
A metric monocular depth network (Depth Anything V2, outdoor, large; \citealp{yang2024depth}) used in place of the flat-ground geometry, with the same
detections and bearings, reaches AP@4\,m %(da).3f: its ranges are about 2.25 times those of the flat-ground geometry at all distances from 4 to 40\,m. With a single
scale factor fitted on surveyed trees of the validation area ($\times$%(k).2f; a secondary analysis added after the
primary result) it reaches %(das).3f, inside the 95\%% interval of the
flat-ground geometry (%(aplo).3f--%(aphi).3f), which needs no surveyed trees to set its scale. Mapillary photos of the same test area reach only AP@4\,m %(mly).3f
(random bearings give %(mch).3f), mainly because the detector finds few of the surveyed trees in those photos.
With vehicle camera heights of 1.3 and 1.9\,m instead of 1.6\,m, AP@4\,m is %(mlo).3f and %(mhi).3f.

\subsection{Distances in Bangladesh}\label{sec:e3}
Of the %(nfield)d field trees with an ARuler reference, %(n3)d have a trunk base visible in the photo (ARuler distances
%(arlo).1f--%(arhi).1f\,m). For these, the distance computed from a single phone photo has a median absolute error of
%(e3a).2f\,m (95\%% CI %(e3alo).2f--%(e3ahi).2f; %(w3a).0f\%% within 2\,m) with fixed settings, and %(e3l).2f\,m
(%(e3llo).2f--%(e3lhi).2f; %(w3l).0f\%%) after a leave-one-out scale correction (Fig.~\ref{fig:validation}c). With fixed settings
the computed distances are short (median ratio to ARuler %(ratio).2f). Phone GPS positions at the %(ngps)d sites where both are
available differ from ARuler by a median of %(gps).1f\,m, so GPS is not a usable reference at this scale. The monocular depth
network has a median error of %(bda).2f\,m as is and %(bdl).2f\,m after the same correction: worse than the ground-plane
distances in both comparisons, significantly so after the correction (Wilcoxon signed-rank test, $p=$%(wl).3f; without the
correction $p=$%(wu).3f).

\input{sections/fig_pasadena_map}
\subsection{Planting gaps}\label{sec:e5}
On the Pasadena test area, with a planting spacing of 12\,m, %(gp).1f\%% of the road length flagged as free is free according
to the inventory, against %(gb).1f\%% for an empty map (no detector), and the method finds %(gr).1f\%% of the free length
(Fig.~\ref{fig:validation}d). The result holds at 10\,m (%(g10).1f\%% vs.\ %(b10).1f\%%) and 15\,m (%(g15).1f\%% vs.\
%(b15).1f\%%). Per street side, the free length predicted and the free length
in the inventory correlate with $r=$%(gpr).2f. Of the city's recorded vacant planting sites, %(vac).1f\%% fall inside
predicted gaps (chance: %(vch).1f\%%); of the sites recorded as having no room for a replacement tree, %(nor).1f\%%.
The gaps therefore mark empty roadside length; whether a gap can be planted (utilities, driveways, sight lines) is not
decided from the images. Figure~\ref{fig:pasmap} shows an excerpt: most empty kerb in the inventory is also empty in the map,
and short additional gaps appear where the map misses a tree, typically a small one. Trees per kilometre per street side agree with the inventory with $r=$%(dr).2f, but are
under-counted (median %(dd).1f vs.\ %(ds).1f per km). Missed trees are mostly young: %(d3).1f\%% of inventory trees with a
trunk diameter of at most 3 inches have a mapped tree within 4\,m (a lenient criterion that does not require a one-to-one
match), against %(d24lo).1f--%(d24hi).1f\%% of trees above 12 inches.

A zero-shot vision--language model asked whether a tree already stands at each candidate spot (Qwen3-VL-4B; \citealp{qwen2025qwen3vl}), with its
threshold chosen on the validation area, raised the share of candidate spots that are truly free from %(v0).1f\%% to
%(v1).1f\%% (+%(vg).1f percentage points, 95\%% CI %(vlo).1f to %(vhi).1f) while keeping %(vk).1f\%% of the truly free spots.
The pre-specified rule required a gain of at least 5 points, a 95\%% interval above zero and at least 80\%% of the truly free
spots kept; the step passed with both the first
detector (+%(vg0).1f points) and the retrained detector. A second question (whether there is room to plant) did not separate
the city's vacant sites from its no-room sites (AUC %(qb).3f, 95\%% CI %(qblo).3f--%(qbhi).3f) and is not used.
A second, stricter threshold (Section~\ref{sec:vlm}), set as the most precise one on the validation area that keeps at least half of the
truly free spots, raises the share of truly free candidates on the test area to %(u4p).1f\%% (+%(u4g).1f points, 95\%% CI
%(u4lo).1f to %(u4hi).1f) while keeping %(u4k).1f\%% of them. The two thresholds give planners a choice between a broad screening list
and a shorter, purer list for site visits.
""" % dict(**UX, lo=S.loc[['rtdetr_l', 'pose', 'yolo11m', 'yolo832', 'posel', 'frcnn (native COCO scorer)']].AP50.min(),
           hi=S.loc[['rtdetr_l', 'pose', 'yolo11m', 'yolo832', 'posel', 'frcnn (native COCO scorer)']].AP50.max(),
           sdlo=S.loc[['rtdetr_l', 'pose', 'yolo11m', 'yolo832', 'posel', 'frcnn (native COCO scorer)']].AP50_sd.min(),
           sdhi=S.loc[['rtdetr_l', 'pose', 'yolo11m', 'yolo832', 'posel', 'frcnn (native COCO scorer)']].AP50_sd.max(),
           v4=S.loc['yolov4'].AP50, gd=S.loc['gdino_zs'].AP50, hwa=hwe.loc['hwa_s0'].AP50,
           hwb=hwe.loc[['hwb_s0', 'hwb_s1', 'hwb_s2']].AP50.mean(), hwbsd=hwe.loc[['hwb_s0', 'hwb_s1', 'hwb_s2']].AP50.std(ddof=1),
           pose=S.loc['pose'].AP50, posesd=S.loc['pose'].AP50_sd, rows='\n'.join(rows),
           tp='--'.join(f'{x:.1f}' for x in sorted(T.median_px)[::2]), bb='--'.join(f'{x:.1f}' for x in sorted(T.box_bottom_median_same_trees)[::2]),
           ap=te['AP_4m'], rec=te['max_recall'], med=te['gate_median_matched10'], p90=te['p90_matched10'],
           vcur=[r for r in s['val_best'] if r['detector'] == 'current'][0]['AP_4m'], vstu=[r for r in s['val_best'] if r['detector'] == 'student'][0]['AP_4m'],
           da=e7['test_depth']['AP_4m'], k=e7s['k'], das=e7s['test']['AP_4m'], mly=mly['Mapillary (chosen setting)']['AP_4m'], mch=mly['Chance (random bearing)']['AP_4m'], mlo=mly['Mapillary, camera height 1.3 m']['AP_4m'], mhi=mly['Mapillary, camera height 1.9 m']['AP_4m'],
           n3=int(e3a.n), e3a=e3a.median_abs_m, w3a=e3a.within_2m_pct, e3l=e3l.median_abs_m, w3l=e3l.within_2m_pct, gps=g.diff_m.abs().median(),
           bda=bda.median_abs_m, bdl=bdl.median_abs_m, gp=g12.precision * 100, gb=b12.precision * 100, gr=g12.recall * 100, gpr=g12.pearson_r_per_side,
           vac=vac.in_pred_gap * 100, vch=vac.chance * 100, nor=nor.in_pred_gap * 100, dr=DN.loc['trees per km'].r,
           dd=DN.loc['trees per km'].detected_median, ds=DN.loc['trees per km'].survey_median,
           v0=V2['test']['unfiltered'] * 100, v1=V2['test']['precision'] * 100, vg=V2['test']['gain_pp'], vlo=V2['test']['ci_pp'][0], vhi=V2['test']['ci_pp'][1],
           vk=V2['test']['kept'] * 100, vg0=V1['test']['gain_pp'],
           d3=DB.loc['<=3 in', 'retrained'] * 100, d24lo=DB.loc[['13-24 in', '>24 in'], 'retrained'].min() * 100,
           d24hi=DB.loc[['13-24 in', '>24 in'], 'retrained'].max() * 100, **X1)
if os.path.exists('hw_test_results.json'):
    HT = j('hw_test_results.json'); HO = j('hw_outputs.json'); HS = j('map_final/hw_summary.json'); UQ = j('usability_result.json')['test']
    HV = j('hw_val_box_ap50.json')
    PI = j('hw_pseudo_info.json'); LB = json.load(open('hw_train_labels.json'))
    HR = pd.read_csv('hw_heat_ranking.csv'); LS = pd.read_csv('lst_corridor_1km.csv'); e8 = HS['e8']
    hot = HR[HR.heat_rank <= 10].km; ends = int(((hot < 30) | (hot > 209)).sum())
    HAU, HRR, HE, HX, RK, USV = A['hw_audit'], A['hw_test_rerun'], A['heat'], A['heat_extremes'], A['route_km'], A['usability']
    X2 = dict(ntrip=HRR['trips'], tlo=HRR['trip_cluster']['ci95'][0] * 100, thi=HRR['trip_cluster']['ci95'][1] * 100,
              recu=HO['recall_used'], clickmed=HAU['click_range_quantiles_m']['0.5'], winrec=HAU['mean_recall_in_window'] * 100,
              nwin=int(pd.DataFrame(HAU['runs']).n_clicks_in_window.iloc[0]), winfp=(1 - HAU['mean_precision_proxy_in_window']) * 100,
              mapf=1 / HO['recall_used'], winf=HAU['mean_precision_proxy_in_window'] / HAU['mean_recall_in_window'],
              over=1 / HAU['implied_density_factor_vs_map'], uvauc=USV['val_auc'], ptc=HE['shift_p_trees_canopy'],
              rex=HE['rho_excl_30km_ends'], pex=HE['p_excl'], nex=HE['n_excl'], pst=HE['shift_p_one_sided'], nsh=HE['n_shifts'],
              psc=HE['shift_p_canopy'], h10=HX['hottest10_raw'], c10=HX['coolest10_raw'], h10c=HX['hottest10_corrected'], c10c=HX['coolest10_corrected'])
    assert abs(HRR['mean'] - HT['new']['mean']) < 1e-3
    tex += r"""
\input{sections/fig_highway}
\subsection{Transfer to the Dhaka--Chattogram highway}\label{sec:e6}
The detector was adapted to crowd-sourced Mapillary photos of the highway (phones and action cameras; the map uses photos
taken 2023--2026) with %(ntr)d labelled training photos and %(nva)d validation photos (%(nbox)d boxes). Of two configurations, the one
with windscreen-style augmentation scored higher on the highway validation photos (box AP$_{50}$ %(vb).3f vs.\ %(va).3f).
Self-training on %(nps)s pseudo-labelled photos (confidence threshold set for a precision of 0.80 on validation) reached
%(vc).3f, below the pre-specified margin (an improvement of at least 0.02), and was not used.

On the %(nte)d trees marked in %(nph)d usable test photos from %(ntrip)d trips, scored once, the boxes of the adapted detector
contain %(new).1f\%% of the trees (mean of three runs; 95\%% CI %(nlo).1f--%(nhi).1f by photo-level bootstrap, %(tlo).1f--%(thi).1f when
whole trips are resampled), against %(old).1f\%% (%(olo).1f--%(ohi).1f) for the detector trained without highway photos
(Fig.~\ref{fig:highway}b). The share is computed over all marked trees: the planned restriction to trees within 20\,m of the
camera was dropped before any model was run on the test photos, because it relied on Mapillary's stored camera orientations,
which proved unusable for ground-plane distances (amendment~4 of the pre-specified protocol). Under the pre-specified rule
(at least 70\%%: planting-gap map; 40--70\%%: trees per kilometre; below 40\%%: no map), the highway output is trees per
kilometre, and no individual planting gaps are reported for the highway.

The protocol also specified dividing the counts by the measured detection rate (%(recu).3f). That rate was measured on marked
trees at all distances (median %(clickmed).0f\,m from the camera), whereas the map counts only trees 2--20\,m from the camera,
where the detector finds more of the marked trees (%(winrec).0f\%%, measured on only %(nwin)d trees) and %(winfp).0f\%% of its boxes
contain no marked tree (some of them may be trees the annotator did not mark). We therefore report the densities as detected
alongside the protocol's estimate, and as a third estimate the as-detected density multiplied by the precision and divided by the
recall measured inside the 2--20\,m window (factor %(u3w).2f). The ten hottest stretches then have %(u3h0).1f, %(u3h1).1f and %(u3h2).1f trees
per kilometre (as detected, protocol, window) and the ten coolest %(u3c0).1f, %(u3c1).1f and %(u3c2).1f. Because each estimate scales every
stretch by the same factor, all rank correlations below are identical for the three. Choosing the as-detected values for the
figures was decided after the highway test.
Dividing by the detection rate would multiply them by %(mapf).1f, about %(over).1f times the factor implied inside the 2--20\,m
window (%(winf).1f), so corrected densities are an upper bound. Comparisons between stretches do not depend on this choice.

Photos accepted by the usability check (AUC %(uauc).3f on the test photos and %(uvauc).3f on the validation photos; %(uacc).1f\%% of usable photos kept, %(urej).1f\%% of unusable photos
rejected) in trips with a reliable horizon cover %(ckm).1f\,km of the %(rkm).1f\,km route (%(trips)d trips, %(phot)s photos;
%(trees)s trees after merging within trips; Figs.~\ref{fig:highway}a and \ref{fig:hwmap}a). Tree density agrees with satellite canopy cover (1\,m
canopy height map, \citealp{tolan2024canopy}; 2015--2020 imagery): Spearman $\rho=$%(rk).2f per kilometre ($n=$%(nk)d; circular-shift test along the
route, $p=$%(ptc).3f) and %(rh).2f per 100\,m
($n=$%(nh)s; Fig.~\ref{fig:highway}c). Of the 100\,m stretches without a tree in any trip, %(e10).1f\%% have canopy cover below
10\%%, against %(t10).1f\%% of stretches with trees. Counts from single trips agree poorly between trips covering the same piece
of road (Spearman %(r100).2f per 100\,m, %(p100)s trip pairs; %(r1k).2f per kilometre, %(p1k)d pairs): a single trip sees few
trees (median %(mtk).0f per kilometre), so the map combines all trips covering a stretch; doing so lowers the share of
50\,m road sides without any tree from %(es).1f\%% to %(ec).1f\%%. Only %(u5n4)d stretches are covered by four or more trips, too
few to trace agreement as trips accumulate (Spearman with canopy %(u5k1).2f for one trip and %(u5k4).2f for four, wide intervals). On the %(u5n2)d
stretches with at least two trips, an exploratory comparison gives $\rho=$%(u5a).2f with canopy for one randomly drawn trip,
%(u5b).2f for two and %(u5all).2f for all trips.

The canopy map predates the photos by 3--11 years. Against vegetation from Sentinel-2 images of the 2023/24 and 2024/25 dry
seasons (share of 10\,m pixels within 20\,m of the route with NDVI above 0.4), tree density gives Spearman $\rho=$%(u6r).2f
($n=$%(u6n)d stretches, circular-shift $p=$%(u6p).3f); the median NDVI gives %(u6rm).2f ($p=$%(u6pm).3f), and the two satellite
measures agree with each other at %(u6rc).2f.

\input{sections/fig_highway_map}
Hot-season surface temperature ranges from %(tmin).1f to %(tmax).1f\,$^\circ$C between 1\,km stretches (Fig.~\ref{fig:hwmap}b). The ten hottest mapped
stretches (%(endtxt)s within 30\,km of Dhaka or Chattogram) have a median of %(h10).1f trees per kilometre as detected
(upper bound %(h10c).1f), against %(c10).1f (upper bound %(c10c).1f) in the ten coolest. The hottest stretches lie at the urban ends, where
tree density is lowest; along the rural middle of the route, temperature and tree density are unrelated (Spearman
$%(rex).2f$, $p=$%(pex).2f, $n=$%(nex)d stretches more than 30\,km from either city). Across all mapped stretches, the negative
correlation between temperature and tree density (Spearman $%(rlt).2f$) does not survive a circular-shift test that keeps the
spatial autocorrelation along the route ($p=$%(pst).2f, %(nsh)d shifts), whereas the correlation between temperature and
canopy cover does ($%(rlc).2f$, $p=$%(psc).3f). Distance to the nearer city explains much of the temperature (Spearman
$%(u7d).2f$); after removing it from both variables, temperature and tree density remain weakly negatively related (partial
$\rho=%(u7t).2f$, circular-shift $p=%(u7tp).3f$; canopy: $%(u7c).2f$, $p=%(u7cp).3f$).
For planning we rank stretches by a priority score, the robust $z$-score of temperature minus that of $\log(1+$trees per km$)$.
%(u7o)d of its top ten are among the ten hottest stretches, and %(u7s)d of its top fifteen stay in the top fifteen in at least 75\%%
of 1,000 resamples of the trips covering each stretch (Table~\ref{tab:priority}). Two of the top fifteen (km 42 and 87) lie in the rural middle of the route, where they rank high because no tree was detected rather than because they are hot.

\input{sections/tab_priority}
""" % dict(**UX, ntr=sum(v['split'] == 'train' for v in LB.values()), nva=sum(v['split'] == 'val' for v in LB.values()),
           nbox=sum(len(v['lines']) for v in LB.values()), vb=HV['hwb_s0'], va=HV['hwa_s0'], nps=f"{PI['pseudo_photos']:,}", vc=HV['hwc_s0'],
           nte=HT['n_trees'], nph=HT['n_photos'], new=HT['new']['mean'] * 100, nlo=HT['new']['ci95'][0] * 100, nhi=HT['new']['ci95'][1] * 100,
           old=HT['old']['mean'] * 100, olo=HT['old']['ci95'][0] * 100, ohi=HT['old']['ci95'][1] * 100,
           uauc=UQ['auc'], uacc=UQ['usable_accepted'] * 100, urej=UQ['unusable_rejected'] * 100,
           ckm=HS['covered_km'], rkm=RK['map'], trips=HS['trips_in_map'], phot=f"{HS['photos_in_map']:,}", trees=f"{HS['trees']:,}",
           rk=HO['heat']['spearman_trees_canopy_1km'], nk=HO['heat']['n_stretches'], rh=HO['canopy_100m']['spearman_trees_canopy'],
           nh=f"{HO['canopy_100m']['n_stretches']:,}", e10=HO['canopy_100m']['empty_with_canopy_lt10'] * 100,
           t10=HO['canopy_100m']['withtrees_with_canopy_lt10'] * 100, r100=e8['spearman'], p100=f"{e8['n_pairs']:,}",
           r1k=e8['km_secondary']['spearman'], p1k=e8['km_secondary']['n_pairs'], mtk=e8['km_secondary']['median_trees_per_km'],
           es=HS['empty_share_single_trip'] * 100, ec=HS['empty_share_combined'] * 100, tmin=LS.lst_median_c.min(), tmax=LS.lst_median_c.max(),
           endtxt='all' if ends == 10 else f'{ends} of them', rlt=HO['heat']['spearman_lst_trees'], rlc=HO['heat']['spearman_lst_canopy'], **X2)
os.makedirs('manuscript_v2', exist_ok=True); open('manuscript_v2/sec_results.tex', 'w').write(tex)
print(len(tex), 'chars'); print(tex[:600])
