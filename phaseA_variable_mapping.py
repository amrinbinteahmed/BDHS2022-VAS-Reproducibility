import pandas as pd, csv
from pathlib import Path
base=Path('/mnt/data'); reader=pd.io.stata.StataReader(base/'BDKR81FL.DTA',convert_categoricals=False); d=reader.read(); labels=reader.variable_labels()
mapping=[('Vitamin_A','h34','Direct; check eligible ages and nonresponse'),('Primary_sampling_unit','v021','Design-only; never model predictor'),('Number_of_living_children','v218','Direct'),('Sex_of_child','b4','Recode 1=male 2=female; old 1/0 verify'),('Duration_of_breastfeeding','m4','Special codes 93/94/95; 36+ month records mostly missing: verify applicability'),('Mothersage','v012','Direct'),('Mothers_currently_working','v714','Direct'),('Mothers_educational_level','v106','Direct'),('Fathers_education_level','v701','Missing / special code 8 require handling'),('Fathers_Occupation','v705 / v704','VERIFY: husband occupation variable in file'),('Wealth_index','v190','Original codes 1-5, prior 0-4'),('Type_of_place_of_residence','v025','1 urban 2 rural; prior 0/1 recode'),('Division','v024','Direct, labels verify'),('Religion','v130','Special category 96'),('MediaAccess','v157 v158 v159','Derive weekly exposure; distinguish less-than-weekly code'),('Body_Mas_Index','v445','Divide by 100; missingness and special values; avoid pre-imputed BMI'),('child_age','b19','Verify age against alternative b8 if needed'),('child_age_cat','b19','Derived; redefine for 6-59'),('mother_age_cat','v012','Derived'),('BMI','v445','Derived after missing-code processing; threshold audit'),('Breastfeeding_status','m4 / m5','Derived; code 93/94/95 semantics must be checked'),('Religion_New','v130','Derived Muslim vs other; special code handling')]
with open(base/'PhaseA_Variable_Mapping.csv','w',newline='') as f:
 w=csv.writer(f);w.writerow(['Previous analysis variable','Candidate original DHS variable','Original variable label','Verification / caution'])
 for a,b,c in mapping:w.writerow([a,b,'; '.join(f'{k}: {labels.get(k,"NOT FOUND")}' for k in b.replace('/',' ').split()),c])
ages=d.b19.between(6,59); valid=d.h34.isin([0,1]); elig=d[ages]; s=d[ages&valid]; wt=s.v005/1e6
with open(base/'PhaseA_Verification_Report.md','w') as f:
 f.write('# Phase A — Original BDHS KR verification and variable mapping\n\n')
 f.write('Source: BDKR81FL.DTA; analysis conducted from raw file. Candidate variable mappings must be checked against the questionnaire and old processed records.\n\n')
 f.write(f'- Raw KR rows: **{len(d):,}**; columns: **{d.shape[1]:,}**.\n- Age 6–59: **{ages.sum():,}**; age 6–35: **{d.b19.between(6,35).sum():,}**; age 36–59: **{d.b19.between(36,59).sum():,}**.\n')
 f.write(f'- Within age 6–59, h34 responses: {elig.h34.value_counts(dropna=False).to_dict()}. Valid 0/1: **{len(s):,}**.\n')
 f.write(f'- Among valid 0/1: h34=1 **{int((s.h34==1).sum()):,}**; h34=0 **{int((s.h34==0).sum()):,}**; unweighted proportion **{(s.h34==1).mean():.5%}**; preliminary weighted proportion **{(wt*(s.h34==1)).sum()/wt.sum():.5%}**.\n')
 f.write('- Weight: v005 / 1,000,000; PSU: v021; strata: v022/v023. Weighted proportion is not a design-based CI.\n')
 f.write('- **Critical:** Verify whether h34 is asked of all 6–59-month-olds and whether missingness is structurally determined; check living-child eligibility, reference period, and DHS documentation.\n')
 f.write('- **Critical:** v445 is substantially missing and should not be filled with a constant; consider excluding BMI from primary ML model and retaining sensitivity analysis if appropriate.\n')
 f.write('- **Critical:** m4 has structural missingness for older children; previous breastfeeding variable cannot be copied to 6–59 cohort without redesign.\n')
 f.write('- **Do not** describe model results as rerun or reviewer comments as resolved at this phase.\n\n')
 f.write('## Candidate mapping\n\n| Old variable | Original candidate | Verification note |\n|---|---|---|\n')
 for a,b,c in mapping:f.write(f'| {a} | {b} | {c} |\n')
 f.write('\n## Next gate\nVerify original coding and age eligibility, investigate data availability for 36–59 months, decide predictors before modeling, then freeze reproducible sample flow and weighted Table 1.\n')
print('saved',len(s), (s.h34==1).mean(), (wt*(s.h34==1)).sum()/wt.sum())
