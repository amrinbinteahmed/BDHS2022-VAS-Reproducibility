from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import chi2
import statsmodels.api as sm
from patsy import dmatrix
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from zipfile import ZipFile,ZIP_DEFLATED
R=Path('/mnt/data'); O=R/'VAS_SHAP_Nonlinearity_Audit';O.mkdir(exist_ok=True)
a=pd.read_csv(R/'PhaseE15_F_outputs/F_SHAP_Training_Subset_Values.csv');d=pd.read_csv(R/'BDHS2022_VAS_FINAL_Source_Validated.csv')
x=a[['source_KR_row','sampling_weight','child_age_months','maternal_age_years']].merge(d[['source_KR_row','child_age_months','maternal_age_years']],on='source_KR_row',suffixes=('_shap',''),validate='one_to_one')
rows=[]
for v in ['child_age_months','maternal_age_years']:
 xx=x[v].to_numpy(float);yy=x[v+'_shap'].to_numpy(float);ww=x.sampling_weight.to_numpy(float)
 z=(xx-xx.mean())/xx.std();linear=sm.WLS(yy,sm.add_constant(z),weights=ww).fit()
 spline=dmatrix('bs(x, df=5, degree=3, include_intercept=False)',{'x':xx},return_type='dataframe');curved=sm.WLS(yy,spline,weights=ww).fit()
 # descriptive goodness of fit only: SHAP points are fixed outputs of fitted ML model
 r2lin=1-np.sum(ww*(yy-linear.predict(sm.add_constant(z)))**2)/np.sum(ww*(yy-np.average(yy,weights=ww))**2)
 r2spline=1-np.sum(ww*(yy-curved.predict(spline))**2)/np.sum(ww*(yy-np.average(yy,weights=ww))**2)
 q=np.unique(np.quantile(xx,np.linspace(0,1,7)));bins=pd.cut(xx,q,include_lowest=True,duplicates='drop')
 for group in bins.categories:
  sel=np.asarray(bins==group);rows.append({'feature':v,'bin':str(group),'n':int(sel.sum()),'weighted_mean_shap_logodds':np.average(yy[sel],weights=ww[sel]),'weighted_mean_age':np.average(xx[sel],weights=ww[sel]),'linear_weighted_R2':r2lin,'spline_weighted_R2':r2spline,'spline_minus_linear_R2':r2spline-r2lin})
 grid=np.linspace(xx.min(),xx.max(),180);smat=dmatrix('bs(x, df=5, degree=3, include_intercept=False)',{'x':grid},return_type='dataframe')
 # avoid spline extrapolation and inconsistent knots: use design_info from original fit
 from patsy import build_design_matrices
 smat=build_design_matrices([spline.design_info],{'x':grid})[0]
 fig,ax=plt.subplots(figsize=(8,5));ax.scatter(xx,yy,s=10,alpha=.18,label='Training-record SHAP values');ax.plot(grid,curved.predict(smat),linewidth=2,label='Descriptive spline smoother');ax.axhline(0,color='gray',lw=.7);ax.set(xlabel=v.replace('_',' ').title(),ylabel='SHAP contribution (log-odds)',title='Exploratory SHAP dependence (not causal)');ax.legend();fig.tight_layout();fig.savefig(O/f'{v}_dependence_spline.png',dpi=250);plt.close(fig)
pd.DataFrame(rows).to_csv(O/'Age_SHAP_Binned_Profiles_And_Curvature.csv',index=False)
notes='''# SHAP nonlinear and interaction feasibility audit\n\nSource: saved Phase F TreeSHAP values on 1,000 training records, not the holdout. No new XGBoost model fitted.\n\n**Nonlinear patterns:** Age dependence curves, survey-weighted binned means and descriptive weighted spline-versus-linear R² were calculated. A curvature pattern in fixed-model SHAP values is not evidence of a biological threshold, population association or causal effect. No optimal cutoff is estimated.\n\n**Interactions:** Ordinary per-feature SHAP values do NOT identify SHAP interaction values. To make pairwise interaction claims, one needs the exact fitted final XGBoost model, its preprocessing pipeline and corresponding feature matrix, then recompute TreeSHAP interaction values (with model-output scale specified). The available saved 1,000-row SHAP matrix alone cannot support this. Correlated features can also complicate attribution. Reviewer response should transparently state formal interaction analysis was not performed, unless the complete model is recovered and independently evaluated.\n\n**Publication decision:** Present dependence plots as exploratory supplement only; discuss nonlinear associations cautiously, without numeric thresholds. Avoid mechanistic/biological claims inferred from SHAP. Interactions are NOT VERIFIED.\n'''
(O/'README.md').write_text(notes)
with ZipFile(R/'VAS_SHAP_Nonlinearity_Audit.zip','w',ZIP_DEFLATED) as z:
 for f in O.iterdir():z.write(f,arcname=f.name)
 z.write(R/'shap_nonlinearity_audit.py',arcname='reproduce.py')
print(pd.DataFrame(rows).groupby('feature').first()[['linear_weighted_R2','spline_weighted_R2','spline_minus_linear_R2']]);print('saved')
