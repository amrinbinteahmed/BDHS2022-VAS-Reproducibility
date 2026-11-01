import pandas as pd,json,zipfile,sys
from pathlib import Path
R=Path('/mnt/data/VAS_Final_Verification')
r=pd.io.stata.StataReader('/mnt/data/BDKR81FL.DTA',convert_categoricals=False)
vl=r.value_labels()
mapping={'child_sex_code':'B4','maternal_education_code':'V106','paternal_education_code':'V701','paternal_occupation_group':'V705','maternal_work_code':'V714','wealth_quintile_code':'V190','residence_code':'V025','division_code':'V024','religion_group':'V130'}
rows=[]
for _,x in pd.read_csv(R/'Categorical_Reference_Codes.csv').iterrows():
 k=x['variable'];v=mapping.get(k)
 code=str(x['reference_code']).replace('.0','')
 label=vl.get(v,{}).get(int(code),'Derived or not in source') if v else 'Derived media exposure; 0=no weekly media'
 rows.append({'variable':k,'source_DHS_variable':v or 'Derived V157/V158/V159','reference_code':code,'reference_label':label,'all_levels_and_counts':x['levels'],'verification':'Source Stata value-label dictionary' if v else 'Derived indicator; inspect creation code'})
pd.DataFrame(rows).to_csv(R/'Verified_DHS_Reference_Labels.csv',index=False)
# Source label code 98 means unknown, must not be treated as substantive occupation category.
checks=[('source_dataset','7550 rows in validated data','PASS'),('reference_categories','DHS Stata value labels recovered for original coded variables','PASS'),('occupation_unknown','v705=98 (don’t know) excluded from regression, not pooled with occupations','PASS'),('paternal_education_unknown','v701=8 (don’t know) excluded','PASS'),('regression_model','n=7417; 674 PSU; 16 strata; design df=658','PASS'),('covariance_internal','two algebraic score-meat calculations agree; NOT external R survey verification','PASS - LIMITED'),('model_selection','XGBoost training nested-CV selected, holdout not used to select','PASS'),('tuning_search','2 prespecified candidates per model; not comprehensive','PASS'),('tuning_fold_configurations','fold-specific selected settings not fully verified from all final logs','OPEN'),('independent_R_survey','Rscript unavailable in current runtime; independent package replication not completed','OPEN'),('SHAP_interaction','no verified formal pairwise interaction estimates; do not claim','NOT REQUIRED / DISCLOSE'),('external_validation','not performed; limitation, not analysis error','NOT AVAILABLE'),('international_literature','requires verified literature synthesis before manuscript','OPEN')]
pd.DataFrame(checks,columns=['verification_item','evidence_or_decision','status']).to_csv(R/'Final_Verification_Status.csv',index=False)
old=pd.read_csv('/mnt/data/VAS_Regression_Corrected_Audit/Table_Corrected_CompleteCase_Survey_Logistic.csv');new=pd.read_csv(R/'Table_Corrected_CompleteCase_Survey_Logistic.csv')
compare=old.merge(new,on='term',suffixes=('_previous','_corrected'))
compare['absolute_aOR_change']=(compare.aOR_previous-compare.aOR_corrected).abs()
compare[['term','aOR_previous','aOR_corrected','absolute_aOR_change']].to_csv(R/'Regression_Change_Audit.csv',index=False)
readme='''# Final pre-manuscript verification checkpoint — NEW BDHS 2022 analysis only

## Newly found and corrected issue
The earlier complete-case regression (n=7426) incorrectly pooled v705=98 ("don't know") with rare *occupations*. The original DHS Stata value-label dictionary confirms 98 is missing/unknown, not an occupation. The regression was rerun excluding nine unknown-occupation records: **n=7417**. The primary ML dataset and XGBoost holdout (n=7550 and 1458) are unchanged.

## Reproducible findings
- Corrected regression: 674 PSUs, 16 strata, 658 design df, 34 coefficients; convergence achieved. Child age OR %.6f, maternal age OR %.6f. Survey Taylor-linearized score-sandwich SE with internal independent algebra check max difference %.3g.
- Reference labels confirmed from original DHS Stata value labels (see Verified_DHS_Reference_Labels.csv). Paternal occupation reference code 8 is *skilled manual*, not a missingness reference.
- The categorical-grouping choice (occupation 6+9, religion other than Islam/Hinduism) remains exploratory. Complete-case regression is a sensitivity epidemiological model, not primary ML model.
- Existing frozen XGBoost model selection, bootstrap discrimination/calibration and SHAP main-effect results remain unchanged.

## Scientific limitations that must not be hidden
1. An **independent external survey package (R survey)** variance replication was **not** completed. The two score-meat implementations are internal algebra checks, not independent software verification.
2. Exact final hyperparameter selections for every outer fold require a complete run-log archive. Verified performance files do not automatically prove every tuning parameter value.
3. No formal SHAP pairwise interactions or validated biological thresholds; do not claim them. Exploratory spline curves are descriptive and can overfit.
4. No external, temporal or geographical validation. Internal PSU-disjoint holdout is not external validation.
5. International VAS literature synthesis remains a writing/research step; do not claim completed.
6. Weighted GLM + custom Taylor score covariance is a reasonable design-aware approximation; confidence intervals assume specified model and design variables, no FPC. Compare with R survey before final submission if possible.

## Decision
Statistical gap-closure audit is ready to transition to manuscript drafting **with the above explicit qualifications**. Do not label all checks completed or promise reviewer satisfaction. Cite corrected regression n=7417 only, not n=7426.
'''%(new.loc[new.term=='child_age_months','aOR'].iloc[0],new.loc[new.term=='maternal_age_years','aOR'].iloc[0],json.loads((R/'Verification_Metadata.json').read_text())['max_meat_algebra_difference'])
(R/'FINAL_VERIFICATION_REPORT.md').write_text(readme)
with zipfile.ZipFile('/mnt/data/VAS_Final_Verification_Checkpoint.zip','w',zipfile.ZIP_DEFLATED) as z:
 for p in R.iterdir():
  if p.is_file() and p.name!='finalize_audit.py':z.write(p,p.name)
 z.write(R/'finalize_audit.py','finalize_audit.py')
print('Complete',len(rows),'reference labels; max OR change',compare.absolute_aOR_change.max())
