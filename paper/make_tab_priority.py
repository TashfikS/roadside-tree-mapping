"""Priority table (tab:priority) from results/q1_upgrades/u7_heat_priority.json. Run from the repository root."""
import json
U = json.load(open('results/q1_upgrades/u7_heat_priority.json'))
L = [r'\begin{table}[t]\centering\footnotesize',
     r'\caption{Dhaka--Chattogram highway: the 15 mapped kilometres with the highest planting-priority score (robust $z$-score of hot-season surface temperature minus that of $\log(1+$trees per km$)$). Stability: share of 1,000 resamples of the covering trips in which the stretch stays in the top 15. A ranking for survey planning, not a causal estimate.}\label{tab:priority}',
     r'\begin{tabular}{rrrrr}\toprule', r'Rank & km from Dhaka & Surface temp.\ ($^\circ$C) & Trees per km (as detected) & Stability\\\midrule']
for i, r in enumerate(U['top15'], 1):
    L.append(f"{i} & {r['km']}--{r['km'] + 1} & {r['lst']:.1f} & {r['trees_per_km']:.1f} & {100 * r['share_in_top15']:.0f}\\%" + r'\\')
L += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
open('paper/tab_priority.tex', 'w').write('\n'.join(L) + '\n')
