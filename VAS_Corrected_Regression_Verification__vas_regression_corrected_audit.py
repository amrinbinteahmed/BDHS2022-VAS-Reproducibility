import pandas as pd,numpy as np,statsmodels.api as sm
from scipy.stats import t
from pathlib import Path
O=Path('/mnt/data/VAS_Regression_Corrected_Audit');O.mkdir(exist_ok=True)
d=pd.read_csv('/mnt/data/BDHS2022_VAS_FINAL_Source_Validated.csv')
# Primary epidemiological complete-case sensitivity avoids imputing paternal information or treating missingness as reference.
cols=['paternal_education_code','paternal_occupation_code']
a=d.dropna(subset=cols).copy();a=a[a.paternal_education_code!=8].copy()
a['paternal_occupation_group']=a.paternal_occupation_code.astype(int).astype(str).where(~a.paternal_occupation_code.isin([6,9,98]),'Other_rare')
a['religion_group']=a.religion_code.astype(str).where(a.religion_code.isin([1,2]),'Other')
continuous=['child_age_months','maternal_age_years','living_children_count']
catrefs={'child_sex_code':1,'maternal_education_code':0,'paternal_education_code':0,'paternal_occupation_group':'8','maternal_work_code':0,'wealth_quintile_code':1,'residence_code':1,'division_code':1,'religion_group':'1','media_weekly_any':0}
X=a[continuous].astype(float).copy();reference=[]
for c,ref in catrefs.items():
 s=a[c].astype(str).str.replace(r'\.0$','',regex=True)
 ref=str(ref);levels=[ref]+sorted(set(s)-{ref});assert ref in set(s)
 z=pd.get_dummies(pd.Categorical(s,categories=levels),prefix=c,drop_first=True,dtype=float)
 z.index=a.index;X=pd.concat([X,z],axis=1)
 reference.append({'variable':c,'reference_code':ref,'levels':'; '.join(f'{v}={int((s==v).sum())}' for v in levels),'reference_label_status':'Code verified; substantive label requires DHS recode check'})
X=sm.add_constant(X);y=a.Vitamin_A.to_numpy(dtype=float);w=a.sampling_weight.to_numpy(dtype=float);w/=w.mean()
fit=sm.GLM(y,X,family=sm.families.Binomial(),freq_weights=w).fit(maxiter=250)
mu=fit.predict(X).to_numpy();Z=X.to_numpy();scores=Z*(w*(y-mu))[:,None];bread=np.linalg.pinv((Z.T*(w*mu*(1-mu)))@Z)
meat=np.zeros((Z.shape[1],Z.shape[1]));stratum_psu=a[['strata','PSU']].reset_index(drop=True)
for _,idx in stratum_psu.groupby('strata').groups.items():
 ix=np.array(list(idx));s=pd.DataFrame(scores[ix]);s['psu']=stratum_psu.PSU.iloc[ix].to_numpy();tot=s.groupby('psu').sum().to_numpy();m=len(tot)
 if m>1:
  c=tot-tot.mean(axis=0);meat+=m/(m-1)*c.T@c
cov=bread@meat@bread;se=np.sqrt(np.maximum(np.diag(cov),0));b=fit.params.to_numpy();dfdesign=int(a.PSU.nunique()-a.strata.nunique());critical=t.ppf(.975,dfdesign)
result=pd.DataFrame({'term':X.columns,'log_odds':b,'aOR':np.exp(b),'survey_se':se,'CI95_lower':np.exp(b-critical*se),'CI95_upper':np.exp(b+critical*se),'p_value':2*t.sf(np.abs(b/se),dfdesign)})
result.to_csv(O/'Table_Corrected_CompleteCase_Survey_Logistic.csv',index=False);pd.DataFrame(reference).to_csv(O/'Categorical_Reference_Codes.csv',index=False)
# independent algebra check: score aggregation via groupby and cov centered PSU totals
scoreframe=pd.DataFrame(scores,columns=X.columns);scoreframe['strata']=a.strata.to_numpy();scoreframe['PSU']=a.PSU.to_numpy();ps=scoreframe.groupby(['strata','PSU']).sum().reset_index();M=np.zeros_like(meat)
for _,g in ps.groupby('strata'):
 z=g[X.columns].to_numpy();m=len(z)
 if m>1:M+=m*np.cov(z,rowvar=False,ddof=1)
err=float(np.max(np.abs(M-meat)));assert err<1e-6,err
import json
meta={'n_source':len(d),'n_primary':len(a),'excluded_missing_paternal':int(d[cols].isna().any(axis=1).sum()),'excluded_paternal_education_unknown_code8':int((d.paternal_education_code==8).sum()),'n_psu':int(a.PSU.nunique()),'n_strata':int(a.strata.nunique()),'design_df':dfdesign,'converged':bool(fit.converged),'parameters':len(b),'max_meat_algebra_difference':err,'note':'Complete-case survey-weighted associational logistic regression; rare religion and occupation codes grouped. Independent score covariance algebra agrees, but independent survey-package replication and category-label confirmation still pending. No causal claims.'}
(O/'Verification_Metadata.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta,indent=2));print(result.head(5).to_string(index=False))
