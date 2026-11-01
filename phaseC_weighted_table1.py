import pandas as pd, numpy as np, json
from pathlib import Path
BASE=Path('/mnt/data')
df=pd.read_csv(BASE/'BDHS2022_VAS_FINAL_Source_Validated.csv')
assert len(df)==7550 and df.Vitamin_A.isin([0,1]).all()
assert df[['sampling_weight','PSU','strata']].notna().all().all()
# Stratified cluster Taylor linearization of a domain ratio: contributions for out-of-domain records are zero.
def ratio_ci(domain, outcome):
    d=np.asarray(domain,dtype=bool); y=outcome.to_numpy(dtype=float); w=df.sampling_weight.to_numpy(dtype=float)
    den=np.sum(w*d)
    if den<=0:return (np.nan,np.nan,np.nan,np.nan)
    p=np.sum(w*d*y)/den
    residual=w*d*(y-p)/den
    work=pd.DataFrame({'strata':df.strata,'PSU':df.PSU,'u':residual})
    psu=work.groupby(['strata','PSU'],observed=True)['u'].sum().reset_index()
    var=0.; dof=0
    for _,g in psu.groupby('strata'):
        n=len(g)
        if n<2:continue
        var+=n/(n-1)*np.sum((g.u-g.u.mean())**2)
        dof+=n-1
    se=np.sqrt(var)
    from scipy.stats import t
    crit=t.ppf(.975,dof)
    return (p,se,max(0,p-crit*se),min(1,p+crit*se))
fields={'Child age group':'child_age_group','Child sex':'child_sex_code','Maternal age group':'maternal_age_group','Maternal education':'maternal_education_code','Paternal education':'paternal_education_code','Paternal occupation':'paternal_occupation_code','Maternal employment':'maternal_work_code','Living children':'living_children_count','Wealth quintile':'wealth_quintile_code','Residence':'residence_code','Division':'division_code','Religion':'religion_code','Weekly media exposure':'media_weekly_any','Maternal BMI category (descriptive only)':'BMI_category'}
rows=[]; miss=[]
for name,col in fields.items():
    x=df[col].astype('string').fillna('Missing')
    for category in sorted(x.unique()):
        mask=(x==category).to_numpy()
        p,se,lo,hi=ratio_ci(mask,df.Vitamin_A)
        weighted_n=df.loc[mask,'sampling_weight'].sum()
        rows.append({'Variable':name,'Category_code_or_label':str(category),'Unweighted_n':int(mask.sum()),'Unweighted_percent':round(100*mask.mean(),3),'Weighted_percent_all_sample':round(100*weighted_n/df.sampling_weight.sum(),3),'VAS_yes_n':int(df.loc[mask,'Vitamin_A'].sum()),'VAS_coverage_weighted_percent':round(100*p,3),'VAS_coverage_design_SE_percentage_points':round(100*se,3),'VAS_coverage_95CI_low':round(100*lo,3),'VAS_coverage_95CI_high':round(100*hi,3)})
    miss.append({'Variable':name,'Missing_n':int((x=='Missing').sum()),'Missing_percent':round(100*(x=='Missing').mean(),3)})
table=pd.DataFrame(rows);table.to_csv(BASE/'BDHS2022_VAS_Table1_SurveyWeighted.csv',index=False)
pd.DataFrame(miss).to_csv(BASE/'BDHS2022_VAS_Predictor_Missingness.csv',index=False)
p,se,lo,hi=ratio_ci(np.ones(len(df),dtype=bool),df.Vitamin_A)
summary={'n':len(df),'yes':int(df.Vitamin_A.sum()),'no':int((df.Vitamin_A==0).sum()),'weighted_prevalence':p,'design_based_standard_error':se,'design_based_95CI':[lo,hi],'strata':int(df.strata.nunique()),'psu_strata_pairs':int(df.groupby(['strata','PSU']).ngroups),'degrees_of_freedom':int(df.groupby(['strata','PSU']).ngroups-df.strata.nunique()),'variance_method':'Stratified PSU-level Taylor linearization; t CI, df=PSUs-strata; domain estimation uses zeroed out-of-domain influence contributions','caveat':'Weighted prevalence is for children with binary VAS response; nonresponse handling and choice of v022 vs v023 need final methodological review.'}
(BASE/'BDHS2022_VAS_PhaseC_Summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2));print('Table rows',len(table));print(pd.DataFrame(miss).to_string(index=False))
