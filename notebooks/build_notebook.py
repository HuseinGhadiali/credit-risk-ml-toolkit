import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md("# Credit Default Risk: the Modern Toolkit for Classical ML\n"
       "**LazyPredict** (screen 25+ models) → **imbalanced-learn** (does resampling help?) → **Optuna** (Bayesian tuning with pruning) → "
       "**calibration** → credit-risk metrics (Gini, KS, lift, cut-off strategy) → **300–900 score scaling** → **SHAP** reason codes → "
       "**fairlearn** audit → **PSI** stability.\n\n"
       "> Data: *Default of Credit Card Clients* (UCI; Yeh & Lien 2009), 30,000 real card holders, CC0 via the Hugging Face Hub "
       "(`scikit-learn/credit-card-clients`). Target: default on next month's payment."),
    code("import sys, warnings, time\nsys.path.insert(0, '../src')\nwarnings.filterwarnings('ignore')\nimport numpy as np, pandas as pd\n"
         "from IPython.display import Image, display\nfrom sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score\n"
         "from credit import *\nimport charts\n"
         "raw = load_raw()\ndf = engineer(clean(raw))\nX, y = xy(df)\n"
         "X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)\n"
         "print(df.shape, f'default rate {y.mean():.1%}')\ndf.head(3)"),
    md("## 1. Data quality and feature engineering\nThe UCI data dictionary has undocumented codes (EDUCATION 0/5/6, MARRIAGE 0); they are grouped as *other*. "
       "Eleven behavioural features are added: delinquency counts, worst delay, utilisation level and trend, share of bill paid, months with no payment, "
       "and months over the limit."),
    code("pd.crosstab(df.max_delay_months.clip(upper=4), df[TARGET], normalize='index').rename(columns={0: 'paid', 1: 'defaulted'}).round(3)"),
    md("## 2. LazyPredict: screen every sklearn-compatible classifier with one call"),
    code("from lazypredict.Supervised import LazyClassifier\nt = time.time()\n"
         "lb, _ = LazyClassifier(verbose=0, ignore_warnings=True, random_state=SEED).fit(X_tr, X_te, y_tr, y_te)\n"
         "lb = lb.sort_values('ROC AUC', ascending=False)\nprint(f'{len(lb)} models in {time.time() - t:.0f}s')\n"
         "charts.lazypredict_chart(lb)\nlb.to_csv('../reports/lazypredict_leaderboard.csv')\ndisplay(lb.head(12).round(4))\n"
         "display(Image('../images/lazypredict_leaderboard.png'))"),
    md("## 3. Class imbalance: does SMOTE help?\n22% defaulters. Compare LightGBM as-is, with class weights, and with SMOTE inside the CV pipeline "
       "(resampling must happen *inside* each fold, never before the split). 3-fold CV keeps the notebook CPU-friendly."),
    code("import lightgbm as lgb\nfrom imblearn.over_sampling import SMOTE\nfrom imblearn.pipeline import Pipeline as ImbPipeline\n"
         "from sklearn.metrics import average_precision_score\n"
         "base = dict(n_estimators=400, learning_rate=0.03, num_leaves=31, verbose=-1, random_state=SEED)\n"
         "variants = {'As-is': lgb.LGBMClassifier(**base),\n            'Class weights': lgb.LGBMClassifier(**base, class_weight='balanced'),\n"
         "            'SMOTE': ImbPipeline([('smote', SMOTE(random_state=SEED)), ('m', lgb.LGBMClassifier(**base))])}\n"
         "cv = StratifiedKFold(3, shuffle=True, random_state=SEED)\nrows, probs = [], {}\n"
         "for name, m in variants.items():\n    auc = cross_val_score(m, X_tr, y_tr, cv=cv, scoring='roc_auc').mean()\n"
         "    p = m.fit(X_tr, y_tr).predict_proba(X_te)[:, 1]; probs[name] = p\n"
         "    rows.append({'variant': name, 'cv_roc_auc': auc, 'holdout_pr_auc': average_precision_score(y_te, p),\n"
         "                 'mean_predicted_pd': p.mean(), 'actual_default_rate': y_te.mean(), 'brier': brier_score_loss(y_te, p)})\n"
         "imb = pd.DataFrame(rows).set_index('variant').round(4)\nimb"),
    md("**Lesson:** ranking power (AUC) barely moves, but SMOTE and class weights **inflate predicted PDs**. "
       "In credit risk the probabilities themselves are used for pricing and provisioning, so the as-is model plus a decision threshold is preferred."),
    md("## 4. Optuna: Bayesian hyper-parameter search with pruning"),
    code("import optuna\noptuna.logging.set_verbosity(optuna.logging.WARNING)\n"
         "def objective(trial):\n    params = dict(n_estimators=trial.suggest_int('n_estimators', 200, 700, step=100),\n"
         "        learning_rate=trial.suggest_float('learning_rate', 0.01, 0.1, log=True), num_leaves=trial.suggest_int('num_leaves', 8, 96),\n"
         "        min_child_samples=trial.suggest_int('min_child_samples', 10, 200), subsample=trial.suggest_float('subsample', 0.5, 1.0),\n"
         "        subsample_freq=1, colsample_bytree=trial.suggest_float('colsample_bytree', 0.4, 1.0),\n"
         "        reg_lambda=trial.suggest_float('reg_lambda', 1e-3, 30, log=True), verbose=-1, random_state=SEED)\n"
         "    scores = []\n    for k, (a, b) in enumerate(cv.split(X_tr, y_tr)):\n"
         "        m = lgb.LGBMClassifier(**params).fit(X_tr.iloc[a], y_tr.iloc[a])\n"
         "        scores.append(roc_auc_score(y_tr.iloc[b], m.predict_proba(X_tr.iloc[b])[:, 1]))\n"
         "        trial.report(np.mean(scores), k)\n        if trial.should_prune():\n            raise optuna.TrialPruned()\n    return np.mean(scores)\n"
         "study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED), pruner=optuna.pruners.MedianPruner(n_warmup_steps=2))\n"
         "t = time.time(); study.optimize(objective, n_trials=20)\n"
         "pruned = sum(t_.state == optuna.trial.TrialState.PRUNED for t_ in study.trials)\n"
         "print(f'best CV AUC {study.best_value:.4f} after {len(study.trials)} trials ({pruned} pruned) in {time.time() - t:.0f}s')\nstudy.best_params"),
    code("import matplotlib.pyplot as plt\nfrom optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances\n"
         "charts.apply_style()\nax = plot_param_importances(study); ax.figure.set_size_inches(7, 4); ax.figure.tight_layout()\n"
         "ax.figure.savefig('../images/optuna_param_importance.png', dpi=130); plt.close(ax.figure)\n"
         "ax = plot_optimization_history(study); ax.figure.set_size_inches(7, 4); ax.figure.tight_layout()\n"
         "ax.figure.savefig('../images/optuna_history.png', dpi=130); plt.close(ax.figure)\n"
         "display(Image('../images/optuna_param_importance.png'))"),
    md("## 5. Holdout comparison: interpretable baseline vs tuned boosting vs CatBoost/XGBoost"),
    code("from sklearn.linear_model import LogisticRegression\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.pipeline import make_pipeline\n"
         "from catboost import CatBoostClassifier\nfrom xgboost import XGBClassifier\nfrom sklearn.calibration import CalibratedClassifierCV, calibration_curve\n"
         "tuned = lgb.LGBMClassifier(**study.best_params, subsample_freq=1, verbose=-1, random_state=SEED).fit(X_tr, y_tr)\n"
         "models = {'Logistic regression (scorecard-style)': make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X_tr, y_tr),\n"
         "          'XGBoost (defaults)': XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=4, random_state=SEED).fit(X_tr, y_tr),\n"
         "          'CatBoost (defaults)': CatBoostClassifier(verbose=0, random_seed=SEED).fit(X_tr, y_tr),\n"
         "          'LightGBM (Optuna-tuned)': tuned}\n"
         "calibrated = CalibratedClassifierCV(lgb.LGBMClassifier(**study.best_params, subsample_freq=1, verbose=-1, random_state=SEED), method='isotonic', cv=3).fit(X_tr, y_tr)\n"
         "models['LightGBM tuned + isotonic calibration'] = calibrated\n"
         "P = {k: m.predict_proba(X_te)[:, 1] for k, m in models.items()}\n"
         "results = pd.DataFrame([metric_row(k, y_te, p) for k, p in P.items()]).set_index('model').round(4)\nresults.to_csv('../reports/holdout_results.csv')\nresults"),
    code("curves = {k: calibration_curve(y_te, p, n_bins=10, strategy='quantile') for k, p in\n"
         "          {'As-is LightGBM': probs['As-is'], 'SMOTE': probs['SMOTE'], 'Class weights': probs['Class weights'],\n"
         "           'Tuned + isotonic': P['LightGBM tuned + isotonic calibration']}.items()}\n"
         "charts.calibration_chart({k: (v[1], v[0]) for k, v in curves.items()})\ndisplay(Image('../images/calibration.png'))"),
    md("## 6. Credit-risk view: gains, cut-off and a 300–900 score"),
    code("pd_final = P['LightGBM tuned + isotonic calibration']\ngains = gains_table(y_te, pd_final)\ngains.to_csv('../reports/gains_table.csv')\ngains"),
    code("cut = cutoff_strategy(y_te, pd_final)\nbest = int(cut.loc[cut.profit_index.idxmax(), 'approval_rate_pct'])\nthreshold = np.quantile(pd_final, best / 100)\n"
         "charts.cutoff_chart(cut, best)\ndisplay(cut.round(2))\ndisplay(Image('../images/cutoff_strategy.png'))"),
    code("scores = pd_to_score(pd_final)\nbands = score_bands(y_te, scores)\ncharts.score_band_chart(bands)\ndisplay(bands)\ndisplay(Image('../images/score_bands.png'))"),
    md("## 7. SHAP: global drivers and per-applicant reason codes"),
    code("import shap\nexplainer = shap.TreeExplainer(tuned)\nsv = explainer.shap_values(X_te)\nsv = sv[1] if isinstance(sv, list) else sv\n"
         "plt.figure()\nshap.summary_plot(sv, X_te, max_display=12, show=False, plot_size=(8, 5.5))\n"
         "plt.title('What drives default risk (SHAP, holdout)', loc='left', fontsize=12, fontweight='bold')\nplt.tight_layout()\n"
         "plt.savefig('../images/shap_summary.png', dpi=130, bbox_inches='tight'); plt.close()\ndisplay(Image('../images/shap_summary.png'))"),
    code("# reason codes for six declined applicants spread from borderline to highest risk\ndeclined_idx = np.flatnonzero(pd_final > threshold)\ndeclined = declined_idx[np.argsort(pd_final[declined_idx])][np.linspace(0, len(declined_idx) - 1, 6).astype(int)]\nreasons = reason_codes(sv[declined], list(X_te.columns))\n"
         "pd.DataFrame({'score': scores[declined], 'pd': pd_final[declined].round(3), 'principal_reasons': ['; '.join(r) for r in reasons]})"),
    md("## 8. Fairness audit (fairlearn)\nSex and age are **not** model inputs for the decision in many jurisdictions. Here they are used only to audit outcomes at the chosen cut-off."),
    code("from fairlearn.metrics import MetricFrame, selection_rate, true_positive_rate, false_positive_rate, demographic_parity_ratio\n"
         "threshold = np.quantile(pd_final, best / 100)\napprove = (pd_final <= threshold).astype(int)\n"
         "groups = pd.DataFrame({'Sex': X_te.SEX.map({1: 'Male', 2: 'Female'}).to_numpy(),\n"
         "                       'Age': pd.cut(X_te.AGE, [0, 30, 45, 120], labels=['<30', '30-45', '45+']).astype(str).to_numpy()})\n"
         "frames = {}\nfor g in groups:\n"
         "    mf = MetricFrame(metrics={'approval_rate': selection_rate, 'true_positive_rate': true_positive_rate, 'false_positive_rate': false_positive_rate},\n"
         "                     y_true=y_te, y_pred=1 - approve, sensitive_features=groups[g])\n"
         "    t = mf.by_group.copy(); t['approval_rate'] = 1 - t['approval_rate']; t.index = [f'{g}: {i}' for i in t.index]; frames[g] = t\n"
         "    print(g, 'approval-rate ratio (min/max):', round(t.approval_rate.min() / t.approval_rate.max(), 3))\n"
         "fair = pd.concat(frames.values())\ncharts.fairness_chart(fair)\ndisplay(fair.round(3))\ndisplay(Image('../images/fairness.png'))"),
    md("## 9. Stability: Population Stability Index"),
    code("pd_train = calibrated.predict_proba(X_tr)[:, 1]\nprint(f'PSI train vs holdout scores: {psi(pd_train, pd_final):.4f}  (< 0.1 = stable)')\n"
         "# stress the RAW inputs, then rebuild the engineered features (perturbing X directly would leave derived features stale)\n"
         "stressed = clean(raw).loc[X_te.index].copy()\nstressed['LIMIT_BAL'] = stressed.LIMIT_BAL * 0.7\n"
         "late = np.random.default_rng(0).random(len(stressed)) < 0.15\nstressed.loc[late, 'PAY_1'] = stressed.loc[late, 'PAY_1'].clip(lower=0) + 2\n"
         "drift = xy(engineer(stressed))[0][X_te.columns]\n"
         "pd_stress = calibrated.predict_proba(drift)[:, 1]\nprint(f'PSI under a stress scenario (limits cut 30%, 15% of customers 2 months late): {psi(pd_final, pd_stress):.3f}  '\n      f'(mean PD {pd_final.mean():.1%} -> {pd_stress.mean():.1%})')"),
    md("## 10. Model card"),
    code("card = pd.Series({'Model': 'LightGBM (Optuna-tuned) + isotonic calibration', 'Training rows': len(X_tr), 'Holdout rows': len(X_te),\n"
         "    'Holdout ROC AUC': round(results.loc['LightGBM tuned + isotonic calibration', 'roc_auc'], 4),\n"
         "    'Gini': round(results.loc['LightGBM tuned + isotonic calibration', 'gini'], 4), 'KS': round(results.loc['LightGBM tuned + isotonic calibration', 'ks'], 4),\n"
         "    'Brier': round(results.loc['LightGBM tuned + isotonic calibration', 'brier'], 4), 'Recommended cut-off': f'approve safest {best}%',\n"
         "    'Top-decile capture': f\"{gains.cum_defaulters_captured_pct.iloc[0]:.1f}% of defaulters\", 'PSI (train vs holdout)': round(psi(pd_train, pd_final), 4),\n"
         "    'Known limits': 'Taiwan 2005 card data; no bureau/income fields; sex & age used only for auditing'})\n"
         "card.to_frame('value').to_csv('../reports/model_card.csv')\ncard.to_frame('value')"),
]
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
nbf.write(nb, "credit_risk_toolkit.ipynb")
