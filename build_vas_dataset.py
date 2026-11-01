import pandas as pd, numpy as np, json
from pathlib import Path
P=Path('/mnt/data'); d=pd.read_stata(P/'BDKR81FL.DTA',convert_categoricals=False)
a=d.loc[d.b19.between(6,59) & d.b5.eq(1) & d.h34.isin([0,1])].copy()
cols={'h34':'Vitamin_A','b19':'child_age_months','b4':'child_sex_code','v012':'maternal_age_years','v106':'maternal_education_code','v701':'paternal_education_code','v705':'paternal_occupation_code','v714':'maternal_currently_working_code','v218':'living_children_count','v190':'wealth_quintile_code','v025':'residence_code','v024':'division_code','v130':'religion_code','v157':'newspaper_frequency_code','v158':'radio_frequency_code','v159':'tv_frequency_code','v445':'maternal_BMI_raw_x100','m4':'breastfeeding_duration_code','m5':'breastfeeding_months_code','v005':'sampling_weight_raw','v021':'PSU','v022':'strata_v022','v023':'strata_v023','v001':'cluster_v001','v002':'household_number','b16':'child_line_number','bidx':'birth_index'}
cols={k:v for k,v in cols.items() if k in a.columns}
o=a[list(cols)].rename(columns=cols)
o.insert(0,'source_KR_row',a.index.to_numpy()+2)
o['sampling_weight']=o['sampling_weight_raw']/1_000_000
o['maternal_BMI']=o['maternal_BMI_raw_x100']/100
# Special BMI codes cannot be considered plausible anthropometry; retain raw separately.
o.loc[~o.maternal_BMI.between(10,60),'maternal_BMI']=np.nan
for c in ['paternal_education_code']:
 o.loc[o[c].isin([8,9,98,99]),c]=np.nan
for c in ['paternal_occupation_code']:
 o.loc[o[c].isin([98,99]),c]=np.nan
o['media_weekly_any']=((o[['newspaper_frequency_code','radio_frequency_code','tv_frequency_code']]==2).any(axis=1)).astype(int)
o['child_age_group_6_23_24_59']=np.where(o.child_age_months<24,'6-23','24-59')
o['maternal_age_group']=pd.cut(o.maternal_age_years,bins=[14,19,24,29,34,39,44,49],labels=['15-19','20-24','25-29','30-34','35-39','40-44','45-49']).astype('string')
o['BMI_category']=pd.cut(o.maternal_BMI,bins=[-np.inf,18.5,25,30,np.inf],labels=['Underweight','Normal','Overweight','Obese'],right=False).astype('string')
o['survey_cluster_id']=o['strata_v022'].astype('Int64').astype(str)+'_'+o['PSU'].astype('Int64').astype(str)
o['outcome_reference']='h34: vitamin A in last 6 months (verify official questionnaire wording)'
o.to_csv(P/'BDHS2022_VAS_Analysis_Ready_Provisional.csv',index=False)
flow={'raw_KR':len(d),'aged_6_59':int(d.b19.between(6,59).sum()),'age_6_59_deceased':int((d.b19.between(6,59)&d.b5.eq(0)).sum()),'age_6_59_alive':int((d.b19.between(6,59)&d.b5.eq(1)).sum()),'alive_h34_code_8':int((d.b19.between(6,59)&d.b5.eq(1)&d.h34.eq(8)).sum()),'final_binary_outcome':len(o),'positive':int(o.Vitamin_A.sum()),'negative':int((o.Vitamin_A==0).sum()),'weighted_coverage':float(np.average(o.Vitamin_A,weights=o.sampling_weight)),'unweighted_coverage':float(o.Vitamin_A.mean()),'BMI_missing':int(o.maternal_BMI.isna().sum()),'distinct_survey_clusters':int(o.survey_cluster_id.nunique())}
(P/'BDHS2022_VAS_Sample_Flow.json').write_text(json.dumps(flow,indent=2))
(P/'BDHS2022_VAS_README.md').write_text('''# Provisional analysis-ready BDHS 2022 Vitamin A dataset\n\nSource: `BDKR81FL.DTA` (Kids Recode). No imputation, ML model fitting, or feature selection has been performed. Original file is unchanged.\n\n**Selection:** b19=6–59 months; b5=1 (alive); h34 in {0,1}. Exclude h34=8 (do not know), and deceased children with structural missingness. Retain all missing predictor values as missing; no complete-case restriction.\n\n**Outcome:** `Vitamin_A` equals source h34, 1=yes, 0=no. Verify official questionnaire wording, age eligibility, and the appropriate denominator before journal submission.\n\n**Survey design:** `sampling_weight=v005/1e6`; `PSU=v021`; strata candidates v022/v023 retained. `survey_cluster_id` combines strata and PSU for grouped validation; inspect uniqueness and survey documentation before inferential use. Survey weights are **not** automatically applied by storing them in the dataset. Survey-weighted analyses must explicitly use them and account for clustering and strata.\n\n**BMI:** `maternal_BMI=v445/100`, values outside [10,60] treated as missing for modeling only, while raw code is retained. This is a provisional plausibility filter, not an official DHS missing-code mapping; revisit with the codebook. Very high BMI missingness makes it unsuitable as a required primary-model predictor without a defensible missingness plan.\n\n**Breastfeeding:** m4/m5 source codes retained unchanged; structurally unavailable for many older children. Do not use as primary ML predictor until questionnaire applicability is resolved.\n\n**Media exposure:** `media_weekly_any` is a provisional derived indicator (any of v157/v158/v159=2); check frequency value labels and handling of missing/unknown responses before analysis.\n\n**Other predictors:** Numeric category codes retained for interpretation using official value labels; paternal education code 8 and occupation 98/99 mapped to missing while original raw data remains available.\n\n**Important:** This is a reproducible **provisional** analysis-ready file, not a finalized weighted analysis or an approved final model matrix. No train/test split, scaling, imputation or encoding has been done to avoid leakage. Use original codebook and questionnaire for final signoff.\n''')
print(json.dumps(flow,indent=2));print(o.shape)
