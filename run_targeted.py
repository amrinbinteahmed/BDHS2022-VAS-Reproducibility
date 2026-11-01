import os,zipfile,json,warnings
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import t
import statsmodels.api as sm
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,balanced_accuracy_score,brier_score_loss,average_precision_score,recall_score,precision_score,f1_score
from xgboost import XGBClassifier
import shap
warnings.filterwarnings('ignore')
O=Path('/mnt/data/VAS_Reviewer13_14_Targeted_Analysis');O.mkdir(exist_ok=True)
P=Path('/mnt/data/vas_work/VAS_Sequential_Reproducibility_Package/11_ORIGINAL_ARCHIVES')
with zipfile.ZipFile(P/'PhaseE_Boruta_Run_Kit.zip') as z:
 d=pd.read_csv(z.open('BDHS2022_VAS_FINAL_Source_Validated.csv'))
with zipfile.ZipFile(P/'BDHS2022_VAS_PhaseD_Package.zip') as z:
 s=pd.read_csv(z.open('PhaseD_outputs/holdout_assignments.csv'))
d=d.merge(s[['source_KR_row','partition']],on='source_KR_row',validate='one_to_one')
assert len(d)==7550 and d.source_KR_row.is_unique
print('source',d.shape,'partition',d.partition.value_counts().to_dict(),flush=True)
num=['child_age_months','maternal_age_years','living_children_count'];cat=['child_sex_code','maternal_education_code','paternal_education_code','paternal_occupation_code','maternal_work_code','wealth_quintile_code','residence_code','division_code','religion_code','media_weekly_any'];cols=num+cat

def prep():return ColumnTransformer([('num',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),num),('cat',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),cat)])
def model():return XGBClassifier(n_estimators=90,learning_rate=.06,subsample=.85,colsample_bytree=.9,max_depth=2,eval_metric='logloss',n_jobs=1,random_state=42)
def fit_predict(tra,val,return_model=False):
 p=prep();xt=p.fit_transform(tra[cols]);xv=p.transform(val[cols]);y=tra.Vitamin_A.to_numpy();w=tra.sampling_weight.to_numpy();prev=np.average(y,weights=w);sw=w*np.where(y==1,.5/prev,.5/(1-prev));sw/=sw.mean();m=model();m.fit(xt,y,sample_weight=sw)
 if return_model:return p,m
 return m.predict_proba(xv)[:,1]
def metric(y,prob,threshold,w):
 pred=(prob>=threshold).astype(int)
 return {'n':len(y),'weighted_auc':roc_auc_score(y,prob,sample_weight=w),'weighted_average_precision_receipt':average_precision_score(y,prob,sample_weight=w),'weighted_balanced_accuracy':balanced_accuracy_score(y,pred,sample_weight=w),'weighted_brier':brier_score_loss(y,prob,sample_weight=w),'nonreceipt_sensitivity':recall_score(1-y,1-pred,sample_weight=w),'nonreceipt_precision':precision_score(1-y,1-pred,sample_weight=w,zero_division=0),'nonreceipt_f1':f1_score(1-y,1-pred,sample_weight=w,zero_division=0)}

# Reviewer 13: age-restricted survey coverage with fixed original design variables.
a=d[d.child_age_months>=12].copy();assert len(a)==6571
w=a.sampling_weight.to_numpy();y=a.Vitamin_A.to_numpy();coverage=np.average(y,weights=w)
# replicate stratified PSU bootstrap conditional on survey design
rng=np.random.default_rng(991);groups=a.reset_index(drop=True).groupby(['strata','PSU']).indices;by={}
for (st,psu),ix in groups.items():by.setdefault(st,[]).append(np.asarray(ix))
boot=[]
for _ in range(500):
 ix=np.concatenate([np.concatenate([cls[j] for j in rng.integers(0,len(cls),len(cls))]) for cls in by.values()]);boot.append(np.average(y[ix],weights=w[ix]))
coverage_out={'age':'12–59 months','n':len(a),'n_psus':int(a.PSU.nunique()),'weighted_coverage':float(coverage),'approximate_psu_bootstrap_ci95':list(map(float,np.quantile(boot,[.025,.975]))),'original_n':len(d),'original_weighted_coverage':float(np.average(d.Vitamin_A,weights=d.sampling_weight))}
(O/'R13_AgeRestricted_Coverage.json').write_text(json.dumps(coverage_out,indent=2));print('coverage',coverage_out,flush=True)
# Survey-weighted regression reproduce exact complete case specification on restricted cohort.
cc=a.dropna(subset=['paternal_education_code','paternal_occupation_code']).copy();cc=cc[(cc.paternal_education_code!=8)&(cc.paternal_occupation_code!=98)].copy();cc['paternal_occupation_group']=cc.paternal_occupation_code.astype(int).astype(str).where(~cc.paternal_occupation_code.isin([6,9]),'Other_rare');cc['religion_group']=cc.religion_code.astype(str).where(cc.religion_code.isin([1,2]),'Other')
catrefs={'child_sex_code':1,'maternal_education_code':0,'paternal_education_code':0,'paternal_occupation_group':'8','maternal_work_code':0,'wealth_quintile_code':1,'residence_code':1,'division_code':1,'religion_group':'1','media_weekly_any':0}
X=cc[num].astype(float).copy()
for c,ref in catrefs.items():
 s=cc[c].astype(str).str.replace(r'\.0$','',regex=True);ref=str(ref);levels=[ref]+sorted(set(s)-{ref});assert ref in set(s),(c,ref)
 z=pd.get_dummies(pd.Categorical(s,categories=levels),prefix=c,drop_first=True,dtype=float);z.index=cc.index;X=pd.concat([X,z],axis=1)
X=sm.add_constant(X);yy=cc.Vitamin_A.to_numpy(float);ww=cc.sampling_weight.to_numpy(float);ww/=ww.mean();fit=sm.GLM(yy,X,family=sm.families.Binomial(),freq_weights=ww).fit(maxiter=250)
mu=np.asarray(fit.predict(X));Z=X.to_numpy();scores=Z*(ww*(yy-mu))[:,None];bread=np.linalg.pinv((Z.T*(ww*mu*(1-mu)))@Z);meat=np.zeros((Z.shape[1],Z.shape[1]));design=cc[['strata','PSU']].reset_index(drop=True)
for _,ix in design.groupby('strata').groups.items():
 ix=np.array(list(ix));q=pd.DataFrame(scores[ix]);q['psu']=design.PSU.iloc[ix].to_numpy();tot=q.groupby('psu').sum().to_numpy();m=len(tot)
 if m>1:
  c=tot-tot.mean(axis=0);meat+=m/(m-1)*c.T@c
se=np.sqrt(np.maximum(np.diag(bread@meat@bread),0));beta=fit.params.to_numpy();df=int(cc.PSU.nunique()-cc.strata.nunique());crit=t.ppf(.975,df)
reg=pd.DataFrame({'term':X.columns,'aOR':np.exp(beta),'CI95_lower':np.exp(beta-crit*se),'CI95_upper':np.exp(beta+crit*se),'p_value':2*t.sf(np.abs(beta/se),df)})
reg.to_csv(O/'R13_AgeRestricted_Regression.csv',index=False);print('regression',len(cc),reg.head(4).to_string(index=False),flush=True)
# Age-restricted model: original PSU-disjoint assignments retained, model settings fixed, calibration and threshold reestimated on restricted development only.
tr=a[a.partition=='train'].reset_index(drop=True);te=a[a.partition=='test'].reset_index(drop=True)
assert set(tr.PSU).isdisjoint(set(te.PSU))
cv=StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=2468);oof=np.full(len(tr),np.nan)
for fold,(ti,vi) in enumerate(cv.split(tr[cols],tr.Vitamin_A,tr.survey_cluster_id),1):
 oof[vi]=fit_predict(tr.iloc[ti],tr.iloc[vi]);print('age OOF',fold,flush=True)
cal=LogisticRegression(C=1e6,max_iter=2000,random_state=42);cal.fit(oof.reshape(-1,1),tr.Vitamin_A,sample_weight=tr.sampling_weight);cp=cal.predict_proba(oof.reshape(-1,1))[:,1];ts=np.arange(.20,.801,.01);scores=[balanced_accuracy_score(tr.Vitamin_A,cp>=q,sample_weight=tr.sampling_weight) for q in ts];thr=float(ts[int(np.argmax(scores))]);raw=fit_predict(tr,te);p=cal.predict_proba(raw.reshape(-1,1))[:,1];m=metric(te.Vitamin_A.to_numpy(),p,thr,te.sampling_weight.to_numpy());m.update({'n_train':len(tr),'n_test':len(te),'threshold':thr,'method':'Fixed XGBoost specification; original PSU-disjoint partitions retained; calibration/threshold refitted within restricted development sample'})
# 500 bootstrap holdout uncertainty
rng=np.random.default_rng(991);groups=te.groupby(['strata','PSU']).indices;by={}
for (st,psu),ix in groups.items():by.setdefault(st,[]).append(np.asarray(ix))
boots=[];ty=te.Vitamin_A.to_numpy();tw=te.sampling_weight.to_numpy()
for _ in range(500):
 ix=np.concatenate([np.concatenate([cls[j] for j in rng.integers(0,len(cls),len(cls))]) for cls in by.values()]);
 if len(np.unique(ty[ix]))==2:boots.append(metric(ty[ix],p[ix],thr,tw[ix]))
m['ci95']={k:list(map(float,np.quantile([r[k] for r in boots],[.025,.975]))) for k in boots[0] if k!='n'}
(O/'R13_AgeRestricted_Model.json').write_text(json.dumps(m,indent=2));print('age model',m,flush=True)
# Reviewer14: fit final model on original development sample, TreeSHAP pairwise interactions on 1000 sampled development observations.
train=d[d.partition=='train'].reset_index(drop=True);processor,fitm=fit_predict(train,train,True);rng=np.random.default_rng(2026);idx=rng.choice(len(train),size=1000,replace=False);xt=processor.transform(train.iloc[idx][cols]);enc=list(processor.get_feature_names_out());ex=shap.TreeExplainer(fitm);ints=np.asarray(ex.shap_interaction_values(xt));print('interaction tensor',ints.shape,flush=True)
assert ints.ndim==3 and ints.shape[1]==ints.shape[2] and ints.shape[0]==len(idx)
# group feature indices, preserve original feature level and interactions of one-hot encodings
features={n:[] for n in cols}
for i,n in enumerate(enc):
 if n.startswith('num__'):
  original=n.removeprefix('num__')
 else:
  matches=[c for c in cat if n.startswith('cat__'+c+'_')];assert len(matches)==1,(n,matches);original=matches[0]
 features[original].append(i)
assert sum(map(len,features.values()))==len(enc)
weights=train.iloc[idx].sampling_weight.to_numpy();rows=[]
for i,c1 in enumerate(cols):
 for c2 in cols[i+1:]:
  val=ints[:,features[c1],:][:,:,features[c2]].sum(axis=(1,2));rows.append({'feature_1':c1,'feature_2':c2,'weighted_mean_abs_interaction_logodds':float(np.average(np.abs(val),weights=weights)),'weighted_mean_signed_interaction_logodds':float(np.average(val,weights=weights))})
inter=pd.DataFrame(rows).sort_values('weighted_mean_abs_interaction_logodds',ascending=False);inter.to_csv(O/'R14_Exploratory_SHAP_Pairwise_Interactions.csv',index=False);print('top interactions',inter.head(12).to_string(index=False),flush=True)
# plot
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family']='DejaVu Sans'
top=inter.head(12).iloc[::-1];fig,ax=plt.subplots(figsize=(9,6));ax.barh(top.feature_1+' × '+top.feature_2,top.weighted_mean_abs_interaction_logodds);ax.set_xlabel('Survey-weighted mean absolute SHAP interaction (raw-margin/log-odds)');ax.set_title('Exploratory pairwise interactions — selected XGBoost');fig.tight_layout();fig.savefig(O/'R14_SHAP_Interactions.png',dpi=300,bbox_inches='tight');plt.close(fig)
meta={'source':'uploaded corrected reproducibility package','n_primary':len(d),'n_age_restricted':len(a),'n_age_restricted_regression':len(cc),'n_age_restricted_train':len(tr),'n_age_restricted_test':len(te),'shap_training_sample':len(idx),'shap_model':'Reconstructed final XGBoost using locked PhaseE13 code, not saved fitted original artifact','shap_output_scale':'raw margin (log odds)','shap_interaction_scope':'Exploratory model attributions, not inferential tests or causal mechanisms','warning':'Reconstructed model may differ from originally fitted model if software versions differ; cross-check original SHAP importance before manuscript insertion'}
(O/'RUN_NOTES.json').write_text(json.dumps(meta,indent=2))

# Compare reconstructed original fitted model SHAP global values to the verified archive.
sv=np.asarray(ex.shap_values(xt));baseline=pd.read_csv('/mnt/data/vas_work/VAS_Sequential_Reproducibility_Package/12_VERIFIED_OUTPUTS/Table_H7_Weighted_SHAP_Importance.csv').set_index('predictor')
checks=[]
for f in cols:
 vals=sv[:,features[f]].sum(axis=1);observed=float(np.average(np.abs(vals),weights=weights));expected=float(baseline.loc[f,'survey_weighted_mean_absolute_SHAP_margin']);checks.append({'predictor':f,'reconstructed':observed,'verified_reference':expected,'absolute_difference':abs(observed-expected)})
pd.DataFrame(checks).to_csv(O/'R14_SHAP_Reconstruction_Check.csv',index=False)
print('SHAP reference max abs diff',max(v['absolute_difference'] for v in checks),flush=True)
# Independent original holdout check using the fixed refitted model and original development OOF calibration.
with zipfile.ZipFile(P/'BDHS2022_VAS_PhaseE13_Locked_Holdout.zip') as z:
 archived=pd.read_csv(z.open('holdout_predictions.csv'))
orig_train=d[d.partition=='train'].reset_index(drop=True);orig_test=d[d.partition=='test'].reset_index(drop=True)
cv2=StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=2468);oof2=np.full(len(orig_train),np.nan)
for ti,vi in cv2.split(orig_train[cols],orig_train.Vitamin_A,orig_train.survey_cluster_id):oof2[vi]=fit_predict(orig_train.iloc[ti],orig_train.iloc[vi])
cal2=LogisticRegression(C=1e6,max_iter=2000,random_state=42);cal2.fit(oof2.reshape(-1,1),orig_train.Vitamin_A,sample_weight=orig_train.sampling_weight)
raw2=fitm.predict_proba(processor.transform(orig_test[cols]))[:,1];cp2=cal2.predict_proba(raw2.reshape(-1,1))[:,1]
compare=orig_test[['source_KR_row']].copy();compare['raw']=raw2;compare['calibrated']=cp2;compare=compare.merge(archived[['source_KR_row','raw_probability','calibrated_probability']],on='source_KR_row',validate='one_to_one')
recon={'max_raw_probability_difference':float(np.max(np.abs(compare.raw-compare.raw_probability))),'max_calibrated_probability_difference':float(np.max(np.abs(compare.calibrated-compare.calibrated_probability))),'mean_raw_probability_difference':float(np.mean(np.abs(compare.raw-compare.raw_probability))),'reference_shap_max_difference':max(v['absolute_difference'] for v in checks)}
(O/'RECONSTRUCTION_VALIDATION.json').write_text(json.dumps(recon,indent=2));print('validation',recon,flush=True)

print('DONE',O,flush=True)
