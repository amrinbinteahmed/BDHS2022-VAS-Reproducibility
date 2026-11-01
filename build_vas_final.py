import pandas as pd, numpy as np, json, argparse
from pathlib import Path
parser=argparse.ArgumentParser(description='Prepare BDHS VAS private analysis dataset')
parser.add_argument('--input',required=True,help='Authorized BDKR81FL.DTA')
parser.add_argument('--out-dir',required=True,help='Private output directory; never commit')
args=parser.parse_args()
P=Path(args.out_dir).resolve(); P.mkdir(parents=True,exist_ok=True)
d=pd.read_stata(args.input,convert_categoricals=False)
q=d.loc[d.b19.between(6,59)&d.b5.eq(1)&d.h34.isin([0,1])].copy()
fields={'h34':'Vitamin_A','b19':'child_age_months','b4':'child_sex_code','v012':'maternal_age_years','v106':'maternal_education_code','v701':'paternal_education_code','v705':'paternal_occupation_code','v714':'maternal_work_code','v218':'living_children_count','v190':'wealth_quintile_code','v025':'residence_code','v024':'division_code','v130':'religion_code','v157':'newspaper_frequency_code','v158':'radio_frequency_code','v159':'tv_frequency_code','v445':'maternal_BMI_raw_x100','m4':'breastfeeding_duration_code','m5':'breastfeeding_months_code','v005':'sampling_weight_raw','v021':'PSU','v022':'strata','v023':'strata_alternative','v001':'cluster_v001','v002':'household_number','v003':'mother_line_number','b16':'child_line_number','bidx':'birth_index','v001':'cluster_v001'}
o=q[list(fields)].rename(columns=fields).copy();o.insert(0,'source_KR_row',q.index.to_numpy()+2)
o['sampling_weight']=o.sampling_weight_raw/1e6
o['maternal_BMI']=o.maternal_BMI_raw_x100/100
# Do not overwrite source or silently impute. Only finite plausible BMI in derived field.
o.loc[~o.maternal_BMI.between(10,60),'maternal_BMI']=np.nan
o['BMI_category']=pd.cut(o.maternal_BMI,[-np.inf,18.5,25,30,np.inf],right=False,labels=['Underweight','Normal','Overweight','Obese'])
media=o[['newspaper_frequency_code','radio_frequency_code','tv_frequency_code']]
o['media_weekly_any']=np.where(media.eq(2).any(axis=1),1,np.where(media.isin([0,1,2]).all(axis=1),0,np.nan))
o['child_age_group']=pd.cut(o.child_age_months,[5,11,23,35,59],labels=['6-11','12-23','24-35','36-59'])
o['maternal_age_group']=pd.cut(o.maternal_age_years,[14,19,24,29,34,39,44,49],labels=['15-19','20-24','25-29','30-34','35-39','40-44','45-49'])
o['survey_cluster_id']=o.strata.astype('Int64').astype(str)+'_'+o.PSU.astype('Int64').astype(str)
o['primary_model_BMI_included']=0
o['primary_model_breastfeeding_included']=0
# No categorical imputation or train/test transformations are performed here.
assert len(o)==7550 and o.Vitamin_A.sum()==5389 and o.Vitamin_A.isin([0,1]).all()
assert o.sampling_weight.gt(0).all() and o.strata.notna().all() and o.PSU.notna().all()
assert o.child_age_months.between(6,59).all()
assert not o.duplicated(['cluster_v001','household_number','mother_line_number','birth_index']).any()
out=P/'BDHS2022_VAS_FINAL_Source_Validated.csv';o.to_csv(out,index=False)
flow={'original_KR':len(d),'aged_6_59':int(d.b19.between(6,59).sum()),'aged_6_59_alive':int((d.b19.between(6,59)&d.b5.eq(1)).sum()),'alive_6_59_h34_unknown':int((d.b19.between(6,59)&d.b5.eq(1)&d.h34.eq(8)).sum()),'final_binary':len(o),'received':int(o.Vitamin_A.sum()),'not_received':int((o.Vitamin_A==0).sum()),'unweighted_coverage':float(o.Vitamin_A.mean()),'sampling_weighted_coverage':float(np.average(o.Vitamin_A,weights=o.sampling_weight)),'bmi_missing':int(o.maternal_BMI.isna().sum()),'strata':int(o.strata.nunique()),'cluster_ids':int(o.survey_cluster_id.nunique()),'media_missing':int(o.media_weekly_any.isna().sum())}
(P/'BDHS2022_VAS_FINAL_Validation.json').write_text(json.dumps(flow,indent=2))
(P/'BDHS2022_VAS_FINAL_README.md').write_text('''# BDHS 2022 Vitamin A — Source-validated final analysis dataset\n\n**Source:** BDKR81FL.DTA, Bangladesh DHS 2022 Kids Recode. Questionnaire Appendix F, woman's questionnaire, Section 6, Q601–605.\n\n**Population:** Living children aged 6–59 completed months, valid yes/no response to Q605 (Vitamin A dose during preceding six months). Q605 response 1=Yes, 2=No, 8=Don't know. In KR recode h34 1=Yes, 0=No, 8=Don't know. Missing and don't know outcomes are not assigned a class.\n\n**Outcome:** Vitamin_A 1=received, 0=not received. Original row reference and cluster/household/birth-index retained for reproducibility.\n\n**Survey design:** sampling_weight=v005/1,000,000, PSU=v021, strata=v022 (v023 retained for verification). Weighted prevalence requires explicit weighting and survey-aware confidence intervals need PSU and strata. Do not use PSU, strata, source row, sampling weight or child ID as predictors.\n\n**Media:** v157 newspaper, v158 radio, v159 TV. DHS recode 0=not at all, 1=less than once a week, 2=at least once a week; derived media_weekly_any is 1 if any weekly, 0 if all three known and none weekly, otherwise missing. Questionnaire Q119–121.\n\n**BMI:** v445 is scaled by 100. Original value retained. Derived BMI uses a documented conservative plausible interval 10–60 and is missing outside it; BMI is NOT a primary-model predictor due to substantial missingness. No BMI imputation performed.\n\n**Breastfeeding:** m4/m5 are retained as raw fields but NOT used in primary model due to structural availability and differing applicability by age.\n\n**Final here means final source-validated *dataset construction*, not final model matrix or final published analysis.** Missing predictor handling, model-specific encoding, training-fold imputation, cross-validation, survey-weighted uncertainty and model comparisons must be completed later. A DHS recode dictionary and survey report remain authoritative for final published coding choices.\n\n**Never use columns named source_KR_row, survey_cluster_id, cluster_v001, household_number, child_line_number, birth_index, PSU, strata, strata_alternative, sampling_weight_raw, sampling_weight, primary_model_BMI_included or primary_model_breastfeeding_included as predictors.**\n''')
print(json.dumps(flow,indent=2));print(out)
