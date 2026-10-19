"""Photo usability check for the highway map (paper Section 3.4; hw_prereg.md, amendment 5).

L2-regularised logistic regression on the usab_features.py features (8 image statistics + YOLO11s-cls embedding),
standardised on the training photos; C chosen by 5-fold cross-validated AUC on the training photos; threshold by
Youden's J on the validation photos; the 150 test photos are scored once.
Inputs : usab_features_450.parquet (usab_features.py), ../data/highway_usability_labels.csv (made by make_hw_labels.py),
         ../data/highway_test_trunk_points.csv (first-pass test statuses; unusable photos are marked 'unusable').
Outputs: usability_result.json, usability_model.pkl, usability_scores_450.csv
"""
import os, json, pickle
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_STATUS = os.environ.get('HW_TEST_STATUS', os.path.join(ROOT, 'data', 'highway_test_status_pass1.csv'))
FT = pd.read_parquet('usab_features_450.parquet').set_index('id')
UL = pd.read_csv(os.environ.get('HW_USABILITY_LABELS', os.path.join(ROOT, 'data', 'highway_usability_labels.csv')), dtype={'id': str})
TT = pd.read_csv(TEST_STATUS, dtype={'id': str}).groupby('id').status.first().reset_index()
TT['usable'] = (TT.status != 'unusable').astype(int); TT['split'] = 'test'
LAB = pd.concat([UL, TT[['id', 'split', 'usable']]], ignore_index=True); assert LAB.id.isin(FT.index).all()
Xall = FT.loc[LAB.id].values.astype(np.float64)
y = LAB.usable.values; sp = LAB.split.values
tr, va, tst = sp == 'train', sp == 'val', sp == 'test'
sc = StandardScaler().fit(Xall[tr]); X = sc.transform(Xall)
cv = StratifiedKFold(5, shuffle=True, random_state=0); cvauc = {}
for Cc in (0.01, 0.1, 1, 10):
    p = cross_val_predict(LogisticRegression(C=Cc, max_iter=5000), X[tr], y[tr], cv=cv, method='predict_proba')[:, 1]
    cvauc[Cc] = roc_auc_score(y[tr], p)
Cb = max(cvauc, key=cvauc.get); clf = LogisticRegression(C=Cb, max_iter=5000).fit(X[tr], y[tr])
pv = clf.predict_proba(X[va])[:, 1]; fpr, tpr, th = roc_curve(y[va], pv); j = np.argmax(tpr - fpr); thr = float(th[j])
pt = clf.predict_proba(X[tst])[:, 1]
res = dict(n_train=int(tr.sum()), n_train_unusable=int((y[tr] == 0).sum()), n_val=int(va.sum()), n_val_unusable=int((y[va] == 0).sum()),
           cv_auc={str(k): round(v, 3) for k, v in cvauc.items()}, C=Cb, threshold_p_usable=round(thr, 4), val_auc=round(roc_auc_score(y[va], pv), 3),
           val_usable_kept=round(float(tpr[j]), 3), val_unusable_rejected=round(float(1 - fpr[j]), 3),
           test=dict(n=int(tst.sum()), n_unusable=int((y[tst] == 0).sum()), auc=round(roc_auc_score(y[tst], pt), 3),
                     usable_accepted=round(float((pt[y[tst] == 1] >= thr).mean()), 3), unusable_rejected=round(float((pt[y[tst] == 0] < thr).mean()), 3),
                     accepted_share=round(float((pt >= thr).mean()), 3)))
print(json.dumps(res, indent=1)); json.dump(res, open('usability_result.json', 'w'), indent=1)
pickle.dump(dict(scaler=sc, clf=clf, threshold=thr, features=list(FT.columns)), open('usability_model.pkl', 'wb'))
LAB.assign(p_usable=clf.predict_proba(X)[:, 1]).to_csv('usability_scores_450.csv', index=False)
