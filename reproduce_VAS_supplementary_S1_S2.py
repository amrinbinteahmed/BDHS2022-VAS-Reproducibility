"""Reproduce BDHS 2022 VAS Supplementary Tables S1/S2.
Inputs: BDHS2022_VAS_FINAL_Source_Validated.csv,
VAS_Complete_Tuning_Log_Audit.zip, VAS_Tuning_Reproducibility_Audit.zip.
Run: python reproduce_VAS_supplementary_S1_S2.py
"""
from pathlib import Path
from zipfile import ZipFile
from io import BytesIO
import pandas as pd
ROOT=Path(__file__).resolve().parent
D=pd.read_csv(ROOT/'BDHS2022_VAS_FINAL_Source_Validated.csv')
N=len(D)
rows=[
 ('Child age','b19','Continuous','Completed months (6–59)','child_age_months',None),
 ('Maternal age','v012','Continuous','Years','maternal_age_years',None),
 ('Number of living children','v218','Continuous','Count','living_children_count',None),
 ('Child sex','b4','Categorical','Male; female','child_sex_code',None),
 ('Maternal education','v106','Categorical','No education; primary; secondary; higher','maternal_education_code',None),
 ('Paternal education','v701','Categorical',"No education (0); primary (1); secondary (2); higher (3); don't know (8); system missing",'paternal_education_code',8),
 ('Paternal occupation','v705','Categorical',"Did not work (0); professional/technical/managerial (1); sales (3); self-employed agriculture (4); agricultural employee (5); household/domestic (6); services (7); skilled manual (8); unskilled manual (9); don't know (98). Regression reference: skilled manual (8)",'paternal_occupation_code',98),
 ('Maternal employment','v714','Binary','Currently working: yes/no','maternal_work_code',None),
 ('Wealth quintile','v190','Categorical','Poorest; poorer; middle; richer; richest','wealth_quintile_code',None),
 ('Residence','v025','Binary','Urban; rural','residence_code',None),
 ('Administrative division','v024','Categorical','Barishal (1, regression reference); Chattogram (2); Dhaka (3); Khulna (4); Mymensingh (5); Rajshahi (6); Rangpur (7); Sylhet (8)','division_code',None),
 ('Religion','v130','Categorical','Islam (1, regression reference); Hinduism (2); Buddhist (3); Christianity (4); Others (96)','religion_code',None),
 ('Weekly media exposure','v157, v158, v159','Binary','At least weekly newspaper/radio/television exposure versus none of these weekly','media_weekly_any',None),
]
out=[]
for name,bdhs,typ,coding,col,special in rows:
 miss=int(D[col].isna().sum())
 ns=int(D[col].eq(special).sum()) if special is not None else 0
 out.append({'Predictor':name,'BDHS variable':bdhs,'Type':typ,'Coding / verification':coding,'Missing, n (%)':f'{miss} ({100*miss/N:.2f}%)','Non-substantive, n (%)':f'{ns} ({100*ns/N:.2f}%)'})
S1=pd.DataFrame(out)
S1.to_csv(ROOT/'VAS_Supplementary_Table_S1_Final.csv',index=False,encoding='utf-8-sig')
reg_exclude=D['paternal_education_code'].isna()|D['paternal_education_code'].eq(8)|D['paternal_occupation_code'].isna()|D['paternal_occupation_code'].eq(98)
assert N==7550 and int(reg_exclude.sum())==133 and int((~reg_exclude).sum())==7417
with ZipFile(ROOT/'VAS_Complete_Tuning_Log_Audit.zip') as z:
 selected=pd.read_csv(BytesIO(z.read('All_24_Selected_Configurations.csv')))
with ZipFile(ROOT/'VAS_Tuning_Reproducibility_Audit.zip') as z:
 space=pd.read_csv(BytesIO(z.read('Eight_Model_Tuning_Search_Space.csv')))
S2=selected.merge(space,left_on='model',right_on='model',how='left',validate='many_to_one')
assert len(S2)==24 and S2['fixed_settings'].notna().all()
S2=S2.rename(columns={'model':'Algorithm','outer_fold':'Outer fold','selected_parameters':'Selected parameters','inner_mean_auc':'Inner mean AUC','outer_auc':'Outer AUC','candidate_1':'Candidate 1','candidate_2':'Candidate 2','fixed_settings':'Fixed settings (audit)'})
S2=S2[['Algorithm','Outer fold','Selected parameters','Inner mean AUC','Outer AUC','Candidate 1','Candidate 2','Fixed settings (audit)']].sort_values(['Algorithm','Outer fold'])
S2['Inner mean AUC']=S2['Inner mean AUC'].round(4)
S2['Outer AUC']=S2['Outer AUC'].round(4)
S2.to_csv(ROOT/'VAS_Supplementary_Table_S2_Final.csv',index=False,encoding='utf-8-sig')
print('Saved S1:',len(S1),'predictors; missing/non-substantive regression exclusions:',int(reg_exclude.sum()))
print('Saved S2:',len(S2),'selected configurations')
print('WARNING: Logistic regression max_iter=500 in tuning audit but 400 in an earlier Phase E89 script; verify final executed version.')
print('WARNING: ANN batch_size, activation, solver and tolerance not explicitly pinned in tuning audit; check executed code/environment before reporting defaults as verified.')
