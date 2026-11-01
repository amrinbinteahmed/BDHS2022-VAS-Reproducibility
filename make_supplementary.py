"""Reproduce publication-formatted Supplementary Figure S9 and Tables S7/S8.
Uses archived computed results, does not refit models or recompute SHAP.
Run: python make_supplementary.py
Requirements: pandas matplotlib python-docx openpyxl numpy
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs';OUT.mkdir(exist_ok=True)
LABELS={
'child_age_months':'Child Age','maternal_age_years':'Maternal Age',
'division_code':'Administrative Division','media_weekly_any':'Weekly Media Exposure',
'residence_code':'Place of Residence','living_children_count':'Living Children',
'wealth_quintile_code':'Household Wealth','paternal_education_code':'Paternal Education',
'paternal_occupation_code':'Paternal Occupation','maternal_education_code':'Maternal Education',
'child_sex_code':'Child Sex','maternal_employment_code':'Maternal Employment','maternal_work_code':'Maternal Employment',
'religion_code':'Religion'}
raw=pd.read_csv(ROOT/'R14_Exploratory_SHAP_Pairwise_Interactions.csv')
unknown=(set(raw.feature_1)|set(raw.feature_2))-set(LABELS)
if unknown:raise ValueError(f'Unmapped predictor labels: {sorted(unknown)}')
raw=raw.sort_values('weighted_mean_abs_interaction_logodds',ascending=False).reset_index(drop=True)
raw['Predictor 1']=raw.feature_1.map(LABELS)
raw['Predictor 2']=raw.feature_2.map(LABELS)
raw['Survey-weighted mean absolute interaction']=raw.weighted_mean_abs_interaction_logodds
raw['Survey-weighted mean signed interaction']=raw.weighted_mean_signed_interaction_logodds
clean=raw[['Predictor 1','Predictor 2','Survey-weighted mean absolute interaction','Survey-weighted mean signed interaction']]
assert len(clean)==78 and clean[['Predictor 1','Predictor 2']].duplicated().sum()==0
clean.to_csv(OUT/'Supplementary_Table_S8.csv',index=False,float_format='%.6f')

# Figure S9: lollipop design matching Figure 7 (navy labels, pastel lines, circles, right-side values).
N=15
sub=raw.head(N).iloc[::-1]
values=sub.weighted_mean_abs_interaction_logodds.to_numpy()
labels=[f'{a} × {b}' for a,b in zip(sub['Predictor 1'],sub['Predictor 2'])]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.linewidth':1.2,'savefig.facecolor':'white'})
fig,ax=plt.subplots(figsize=(12.6,8.0))
y=np.arange(len(sub))
maxval=float(values.max())
for i,(yi,v) in enumerate(zip(y,values)):
    highlight=(i==len(y)-1)
    ax.hlines(yi,0,v,color='#f3b8a8' if highlight else '#a7c6d5',lw=3.8,alpha=.85,zorder=1)
    ax.scatter(v,yi,s=125,color='#f47f63' if highlight else '#4d86ae',edgecolor='#203b60',lw=1.5,zorder=3)
    ax.text(v+maxval*.015,yi,f'{v:.4f}',va='center',ha='left',fontsize=10.5,color='#203b60',fontweight='bold')
ax.set_yticks(y,labels,fontsize=10.5,fontweight='bold')
ax.set_xlim(0,maxval*1.19)
ax.xaxis.set_major_locator(MultipleLocator(.0025))
ax.xaxis.set_major_formatter(FormatStrFormatter('%.4f'))
ax.grid(axis='x',linestyle=':',color='#cdd8e1',alpha=.9)
ax.set_axisbelow(True)
ax.set_xlabel('Survey-weighted mean absolute SHAP interaction (log-odds scale)',fontweight='bold',fontsize=11.5,labelpad=12)
for spine in ('top','right'):ax.spines[spine].set_visible(False)
ax.spines['left'].set_color('#202020');ax.spines['bottom'].set_color('#202020')
ax.tick_params(axis='y',length=5,width=1.1)
fig.subplots_adjust(left=.37,right=.97,top=.97,bottom=.12)
for ext in ('png','tiff','pdf'):
    fig.savefig(OUT/f'Supplementary_Figure_S9.{ext}',dpi=300,bbox_inches='tight',pad_inches=.16)
plt.close(fig)

# Tables S7: archived results transcribed verbatim from previously supplied cleaned supplementary DOCX.
# These CSVs are the frozen source for formatted tables; values are not re-estimated here.
for filename in ('Supplementary_Table_S7_Overview.csv','Supplementary_Table_S7_Performance.csv','Supplementary_Table_S7_Regression.csv'):
    if not (ROOT/filename).exists():raise FileNotFoundError(filename)

def shade(cell,fill):
    tcPr=cell._tc.get_or_add_tcPr(); shd=OxmlElement('w:shd');shd.set(qn('w:fill'),fill);tcPr.append(shd)

def add_table(doc,df):
    t=doc.add_table(rows=1,cols=len(df.columns));t.style='Table Grid';t.alignment=WD_TABLE_ALIGNMENT.CENTER
    for j,col in enumerate(df.columns):
        c=t.rows[0].cells[j];c.text=str(col);shade(c,'DDE8F4')
        for run in c.paragraphs[0].runs:run.bold=True;run.font.size=Pt(8.5)
    for i,row in enumerate(df.itertuples(index=False,name=None)):
        cells=t.add_row().cells
        for j,val in enumerate(row):
            cells[j].text=str(val);cells[j].vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
            for p in cells[j].paragraphs:
                p.paragraph_format.space_after=Pt(0)
                for run in p.runs:run.font.size=Pt(8)
            if i%2:shade(cells[j],'F4F7FA')
    return t

doc=Document();sec=doc.sections[0];sec.top_margin=Inches(.8);sec.bottom_margin=Inches(.75);sec.left_margin=Inches(.8);sec.right_margin=Inches(.8)
style=doc.styles['Normal'];style.font.name='Times New Roman';style.font.size=Pt(10)
p=doc.add_paragraph('Additional Supplementary Materials — Vitamin A Supplementation (BDHS 2022)');p.style='Title'
doc.add_heading('Supplementary Figure S9',level=1)
doc.add_picture(str(OUT/'Supplementary_Figure_S9.png'),width=Inches(6.8))
p=doc.add_paragraph('Figure S9. Fifteen largest exploratory predictor-level pairwise SHAP interaction strengths in the selected XGBoost model. Survey-weighted mean absolute aggregated SHAP interaction values are reported on the raw-margin (log-odds) scale. Values are descriptive model attributions, not tests of statistical interaction or evidence of causal effects. All 78 pairs are reported in Supplementary Table S8.')
p.style='Caption'
doc.add_page_break()
doc.add_heading('Supplementary Table S7',level=1)
doc.add_paragraph('Age-restricted (12–59 months) sensitivity analysis: survey-weighted coverage, adjusted associations, and XGBoost holdout performance.')
for title,file in [('A. Population and coverage','Supplementary_Table_S7_Overview.csv'),('B. Restricted holdout performance','Supplementary_Table_S7_Performance.csv'),('C. Survey-weighted multivariable regression','Supplementary_Table_S7_Regression.csv')]:
    doc.add_heading(title,level=2)
    df=pd.read_csv(ROOT/file,dtype=str).fillna('')
    add_table(doc,df)
    doc.add_paragraph('')
doc.add_paragraph('Notes: The restricted analysis excludes children aged 6–11 months. Confidence intervals and performance metrics are reproduced from the archived targeted analysis; the script formats rather than recalculates these estimates. aOR = adjusted odds ratio; PSU = primary sampling unit; ROC-AUC = area under the receiver operating characteristic curve.')
doc.add_page_break()
doc.add_heading('Supplementary Table S8',level=1)
doc.add_paragraph('Complete exploratory predictor-level pairwise SHAP interaction estimates (78 pairs).')
add_table(doc,clean.assign(**{'Survey-weighted mean absolute interaction':clean['Survey-weighted mean absolute interaction'].map(lambda v:f'{v:.6f}'),'Survey-weighted mean signed interaction':clean['Survey-weighted mean signed interaction'].map(lambda v:f'{v:.6f}')}))
doc.add_paragraph('Notes: One-hot encoded components were aggregated as signed interaction values at the observation level before calculating survey-weighted means. Estimates are on the raw-margin (log-odds) scale and should not be interpreted as formal statistical tests or causal interactions.')
doc.save(OUT/'Supplementary_S7_S8_S9_Clean.docx')
print('Generated:',*[str(p) for p in OUT.iterdir()],sep='\n')
