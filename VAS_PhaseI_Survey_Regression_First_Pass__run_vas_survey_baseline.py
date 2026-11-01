import numpy as np,pandas as pd,statsmodels.api as sm
from scipy.stats import norm
from pathlib import Path
out=Path('/mnt/data/VAS_Survey_Regression_PhaseI');out.mkdir(exist_ok=True)
df=pd.read_csv('/mnt/data/BDHS2022_VAS_FINAL_Source_Validated.csv')
continuous=['child_age_months','maternal_age_years','living_children_count']
categoricals=['child_sex_code','maternal_education_code','paternal_education_code','paternal_occupation_code','maternal_work_code','wealth_quintile_code','residence_code','division_code','religion_code','media_weekly_any']
X=df[continuous].copy()
for c in categoricals:
 s=df[c].map(lambda v:'Missing' if pd.isna(v) else str(int(v)))
 X=pd.concat([X,pd.get_dummies(s,prefix=c,drop_first=True,dtype=float)],axis=1)
X=sm.add_constant(X).astype(float)
y=df.Vitamin_A.astype(float).values
w=df.sampling_weight.values.astype(float)
# Weight scaling leaves coefficients and linearization invariant when all weights scaled equally.
w=w/np.mean(w)
fit=sm.GLM(y,X,family=sm.families.Binomial(),freq_weights=w).fit(maxiter=250)
mu=np.asarray(fit.predict(X)); V=mu*(1-mu)
A=(X.values.T*(w*V))@X.values
bread=np.linalg.pinv(A)
score=(X.values*(w*(y-mu))[:,None]); k=X.shape[1]
# Taylor linearization using within-stratum centered PSU totals; each stratum independent.
meat=np.zeros((k,k));groups=pd.DataFrame({'strata':df.strata.astype(str),'psu':df.PSU.astype(str)})
for s,ix in groups.groupby('strata').groups.items():
 tmp=pd.DataFrame(score[np.asarray(list(ix))]);tmp['psu']=df.PSU.iloc[list(ix)].values
 psu_scores=tmp.groupby('psu').sum().to_numpy();m=len(psu_scores)
 if m>1:
  centered=psu_scores-psu_scores.mean(axis=0)
  meat+=m/(m-1)*centered.T@centered
cov=bread@meat@bread
se=np.sqrt(np.maximum(0,np.diag(cov)))
b=np.asarray(fit.params); z=b/np.where(se>0,se,np.nan);p=2*norm.sf(np.abs(z))
result=pd.DataFrame({'term':X.columns,'coefficient_log_odds':b,'adjusted_odds_ratio':np.exp(np.clip(b,-100,100)),'se_taylor_psu_strata':se,'ci95_lower':np.exp(np.clip(b-1.96*se,-100,100)),'ci95_upper':np.exp(np.clip(b+1.96*se,-100,100)),'p_value_approx':p})
result.to_csv(out/'Table_I1_Survey_Weighted_Logistic_Regression.csv',index=False)
import json
info={'n':len(df),'weighted_prevalence':float(np.average(y,weights=w)),'psu':int(df.PSU.nunique()),'strata':int(df.strata.nunique()),'n_model_parameters':k,'converged':bool(fit.converged),'outcome':'Vitamin_A 1=receipt','uncertainty':'stratified PSU-centered Taylor-linearized sandwich with normal 1.96 CI','missing_paternal':'missing-indicator category for 109 records; sensitivity complete-case recommended','note':'Survey-weighted associational logistic regression; not a causal model. Verify DHS coding and all reference groups before submission.'}
(out/'Regression_Design_Audit.json').write_text(json.dumps(info,indent=2))
refs={c:sorted(df[c].dropna().unique().tolist())[0] for c in categoricals}
(out/'Categorical_Reference_Codes.json').write_text(json.dumps(refs,indent=2))
print(json.dumps(info,indent=2));print(result.head(12).to_string(index=False))
