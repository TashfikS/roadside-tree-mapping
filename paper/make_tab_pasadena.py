"""Pasadena comparison table (tab:pasadena) from saved results only. Run from the repository root.
Inputs: results/q1_upgrades/pas_table.json, results/pas_student_decision.json, results/pas_trunk_ablation.json,
results/pas_e7_result.json, results/pas_e7_scaled_result.json, results/pas_e7_scaled_val.csv. Output: paper/tab_pasadena.tex"""
import json, pandas as pd
J = lambda f: json.load(open(f'results/{f}'))
PT, SD, KA, E7, E7S = J('q1_upgrades/pas_table.json'), J('pas_student_decision.json'), J('pas_trunk_ablation.json'), J('pas_e7_result.json'), J('pas_e7_scaled_result.json')
V7 = pd.read_csv('results/pas_e7_scaled_val.csv'); v7 = V7.loc[V7.AP_4m.idxmax()]
val = {r['detector']: r['AP_4m'] for r in SD['val_best']}; kv = {r['variant']: r['AP_4m'] for r in KA['val_best']}
ci = lambda k, r='official': '%.3f--%.3f' % tuple(PT['ci95'][k][r])
pt = lambda k, r='AP4': PT['point'][k][r]
rows = [
    ('First detector (human labels, 458 panoramas)', 'box bottom', 'camera height', val['current'], pt('first_detector'), ci('first_detector'), pt('first_detector', 'AP4_strict')),
    ('Retrained detector (inventory-verified labels)$^{a}$', 'box bottom', 'camera height', val['student'], pt('retrained_box'), ci('retrained_box'), pt('retrained_box', 'AP4_strict')),
    ('Keypoint model, same training set', 'box bottom (B)', 'camera height', kv['B'], pt('pose_box_bottom_B'), ci('pose_box_bottom_B'), pt('pose_box_bottom_B', 'AP4_strict')),
    ('Keypoint model, same training set', 'keypoint (K)', 'camera height', kv['K'], pt('pose_keypoint_K'), ci('pose_keypoint_K'), pt('pose_keypoint_K', 'AP4_strict')),
    ('Retrained detector', 'box', 'depth network, as is', E7['val_best']['AP_4m'], pt('depth_as_is'), ci('depth_as_is'), pt('depth_as_is', 'AP4_strict')),
    ('Retrained detector', 'box', r'depth network $\times$%.2f$^{b}$' % E7S['k'], float(v7.AP_4m), E7S['test']['AP_4m'], '--', None),
]
L = [r'\begin{table}[t]\centering\footnotesize',
     r'\caption{Pasadena test area: map score of every configuration, each with its settings chosen on the validation area. Intervals: 400\,m spatial block bootstrap with the same %d resamples for all maps (paired). Strict: every inventory tree counts and non-tree records are removed.}\label{tab:pasadena}' % PT['n_rep'],
     r'\setlength{\tabcolsep}{3.5pt}', r'\begin{tabular}{p{4.0cm}p{1.7cm}p{2.4cm}cccc}\toprule',
     r'Detector & Ground point & Range & Val.\ AP@4\,m & Test AP@4\,m & 95\% interval & Strict\\\midrule']
for d, g, rg, v, t, c, s in rows:
    L.append(f'{d} & {g} & {rg} & {v:.3f} & {t:.3f} & {c} & ' + ('--' if s is None else f'{s:.3f}') + r'\\')
L += [r'\midrule', r'\citet{branson2018google}, street view only & \multicolumn{2}{l}{multi-view fusion} & -- & 0.581 & -- & --\\',
      r'\citet{branson2018google}, street view, aerial and map & \multicolumn{2}{l}{multi-view fusion} & -- & 0.706 & -- & --\\', r'\bottomrule',
      r'\multicolumn{7}{p{14.0cm}}{\scriptsize $^{a}$Used for all other results in the paper; chosen on validation before the keypoint model existed. $^{b}$Scale fitted on surveyed trees of the validation area; analysis added after the primary depth-network result, no tree file saved for the interval. Branson et al.: reported values, same benchmark evaluator.}\\',
      r'\end{tabular}', r'\end{table}']
open('paper/tab_pasadena.tex', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))
