import numpy as np,pandas as pd,json,time,zipfile
from pathlib import Path
from scipy.stats import binomtest
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score,average_precision_score,balanced_accuracy_score,recall_score,brier_score_loss
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
P=Path('/mnt/data');O=P/'PhaseE3_outputs';O.mkdir(exist_ok=True)
d=pd.read_csv(P/'BDHS2022_VAS_FINAL_Source_Validated.csv');a=pd.read_csv(P/'PhaseD_outputs/holdout_assignments.csv');d=d.merge(a[['source_KR_row','partition']],on='source_KR_row',validate='one_to_one');d=d[d.partition=='train'].copy()
num=['child_age_months','maternal_age_years','living_children_count'];cat=['child_sex_code','maternal_education_code','paternal_education_code','paternal_occupation_code','maternal_work_code','wealth_quintile_code','residence_code','division_code','religion_code','media_weekly_any'];cols=num+cat
for c in cols:d[c]=pd.to_numeric(d[c],errors='coerce')
for c in cat:d[c]=d[c].map(lambda v: str(int(v)) if pd.notna(v) else np.nan)
X=d[cols];y=d.Vitamin_A.astype(int).to_numpy();w=d.sampling_weight.to_numpy();g=d.survey_cluster_id.to_numpy()
def prep():return ColumnTransformer([('num',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),num),('cat',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),cat)],verbose_feature_names_out=False)
def boruta_style(X,y,w,seed,iterations=12):
 rng=np.random.default_rng(seed);p=X.shape[1];hits=np.zeros(p,dtype=int)
 for t in range(iterations):
  sh=np.column_stack([rng.permutation(X[:,j]) for j in range(p)])
  rf=RandomForestClassifier(n_estimators=80,max_depth=7,min_samples_leaf=12,n_jobs=2,random_state=seed+t)
  rf.fit(np.column_stack((X,sh)),y,sample_weight=w)
  imp=rf.feature_importances_;hits+=(imp[:p]>imp[p:].max())
 # Boruta-style repeated shadow comparisons, not the BorutaPy package's full iterative decision process
 selected=np.flatnonzero([binomtest(int(k),iterations,.5,alternative='greater').pvalue<.1 for k in hits])
 if len(selected)<3:selected=np.argsort(hits)[-max(3,min(10,p)):]
 return selected,hits
models={
 'Logistic Regression':LogisticRegression(C=.5,max_iter=500,random_state=42),
 'Decision Tree':DecisionTreeClassifier(max_depth=4,min_samples_leaf=25,random_state=42),
 'Random Forest':RandomForestClassifier(n_estimators=130,max_depth=8,min_samples_leaf=12,n_jobs=2,random_state=42),
 'SVM':SVC(C=1.0,kernel='rbf',probability=True,random_state=42),
 'AdaBoost':AdaBoostClassifier(n_estimators=100,learning_rate=.05,random_state=42),
 'XGBoost':XGBClassifier(n_estimators=130,max_depth=2,learning_rate=.05,n_jobs=2,eval_metric='logloss',random_state=42),
 'LightGBM':LGBMClassifier(n_estimators=130,max_depth=3,num_leaves=7,learning_rate=.05,verbosity=-1,n_jobs=2,random_state=42),
 'ANN':MLPClassifier(hidden_layer_sizes=(32,),alpha=1.0,max_iter=150,early_stopping=True,random_state=42)
}
cv=StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=42);rows=[];sels=[];start=time.time()
for fold,(ti,vi) in enumerate(cv.split(X,y,g),1):
 assert set(g[ti]).isdisjoint(set(g[vi]));p=prep();xt=p.fit_transform(X.iloc[ti]);xv=p.transform(X.iloc[vi]);sel,hits=boruta_style(xt,y[ti],w[ti],seed=500+fold)
 names=p.get_feature_names_out();sels.extend([dict(fold=fold,feature=n,shadow_wins=int(h),selected=bool(i in sel)) for i,(n,h) in enumerate(zip(names,hits))])
 for name,model in models.items():
  model.fit(xt[:,sel],y[ti],sample_weight=w[ti]) if name!='ANN' else model.fit(xt[:,sel],y[ti])
  prob=model.predict_proba(xv[:,sel])[:,1];pred=(prob>=.5).astype(int)
  rows.append(dict(fold=fold,model=name,n_train=len(ti),n_val=len(vi),n_features=len(sel),weighted_training=(name!='ANN'),roc_auc=roc_auc_score(y[vi],prob,sample_weight=w[vi]),average_precision=average_precision_score(y[vi],prob,sample_weight=w[vi]),balanced_accuracy=balanced_accuracy_score(y[vi],pred,sample_weight=w[vi]),sensitivity=recall_score(y[vi],pred,sample_weight=w[vi]),specificity=recall_score(1-y[vi],1-pred,sample_weight=w[vi]),brier=brier_score_loss(y[vi],prob,sample_weight=w[vi])))
 print('fold',fold,'features',len(sel),'elapsed',round(time.time()-start,1),flush=True)
r=pd.DataFrame(rows);r.to_csv(O/'eight_models_foldwise.csv',index=False);pd.DataFrame(sels).to_csv(O/'shadow_feature_selection.csv',index=False)
s=r.groupby('model').agg(roc_auc=('roc_auc','mean'),roc_auc_sd=('roc_auc','std'),average_precision=('average_precision','mean'),balanced_accuracy=('balanced_accuracy','mean'),specificity=('specificity','mean'),sensitivity=('sensitivity','mean'),brier=('brier','mean')).sort_values('roc_auc',ascending=False);s.to_csv(O/'eight_models_summary.csv')
meta={'status':'DEVELOPMENT SCREENING ONLY','n_train':len(d),'models':list(models),'cv':'3-fold StratifiedGroupKFold, group-exclusive','feature_selection':'repeated shadow-feature significance screening (Boruta-style; NOT full canonical Boruta)','iterations':12,'model_training_weights':'7/8 models use sample_weight; sklearn MLP ANN unweighted','evaluation':'survey-weighted validation metrics; threshold 0.5','holdout':'NOT ACCESSED','limitations':['No nested hyperparameter tuning','Feature selection is Boruta-inspired, not full BorutaPy','ANN fit unweighted, not directly comparable as fully survey-weighted','No threshold tuning','3-fold averages do not provide survey-design confidence intervals','All metrics exploratory, not manuscript-ready'],'runtime_seconds':round(time.time()-start,1)}
(O/'audit.json').write_text(json.dumps(meta,indent=2));print(s.to_string(),flush=True)
with zipfile.ZipFile(P/'BDHS2022_VAS_PhaseE3_Eight_Model_Development.zip','w',zipfile.ZIP_DEFLATED) as z:
 for f in O.iterdir():z.write(f,arcname=f.name)
 z.write(__file__,arcname=Path(__file__).name)
