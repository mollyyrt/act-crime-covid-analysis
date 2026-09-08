import pandas as pd
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from covid_dates import add_covid_col

def datetime(df):
    y_q = df['Year'].astype(str) + '-' + df['Quarter']
    df['Date'] = pd.PeriodIndex(y_q, freq='Q').to_timestamp()
    return df


def crime_subplots(df, plot_func, x, y, h, x_label=True, y_label=True, **kwargs):
    fig, axs = plt.subplots(3,4, figsize=(12,7))
    for i, ax in enumerate(axs.flat):
        crime_title = df['Crime'].unique()[i]
        crime = df[df['Crime'] == crime_title]
        if i == 0:
            ax.set_title(crime_title, fontweight='bold')
        else:
            ax.set_title(crime_title)
        plot_func(data=crime, x=x, y=y, hue=h, ax=ax, **kwargs)
        ax.tick_params(axis='y', labelsize=8)
        if x == 'Date':
            ax.tick_params(axis='x', labelsize=6)
        else:
            ax.tick_params(axis='y', labelsize=8)
        if not x_label:
            ax.set_xlabel('')
        if not y_label:
            ax.set_ylabel('')
    if h == 'Covid':
        labels = ['Pre', 'During', 'Post']
        handles = [Line2D([0], [0], color=pal[i], linestyle='-', linewidth=2)
                   for i in range(len(labels))]

        fig.legend(handles=handles, labels=labels, loc='lower center', 
                           ncol=len(labels), bbox_to_anchor=(0.5, 0.005), title='Covid Period')
        


crime_region = pd.read_csv('data/processed/final/region/crime_grouped_region.csv')

# calculate act-wide crime
crime_act = crime_region.groupby(['Crime', 'Year', 'Quarter'], as_index=False).sum().drop(columns=['Region', 'Covid'])
crime_act_all = crime_act.groupby(['Year', 'Quarter', 'Population'], as_index=False).sum().drop(columns=['Crime', 'Rate'])
# specify time in relation to covid
crime_act = add_covid_col(crime_act)
crime_act_all = add_covid_col(crime_act_all)
# recalculate rates
crime_act['Rate'] = crime_act['Number']/(crime_act['Population']/1000)
crime_act_all['Rate'] = crime_act_all['Number']/(crime_act_all['Population']/1000)
# add datetime format column
crime_act = datetime(crime_act)
crime_act_all = datetime(crime_act_all)
#combine data
crime_act_all.insert(0, 'Crime', ['All Crime'] * len(crime_act_all))
crime_act_total = pd.concat([crime_act_all, crime_act])
# distribution of covid periods
quarter_count = crime_act_total[crime_act_total['Crime']=='All Crime']['Covid'].value_counts()
for idx, val in quarter_count.items():
    print(f'Number of {idx} Covid quarters: {val}')


sns.set_palette('Set2', 6)
pal = sns.color_palette('Set2', 6)

# examine crime rates over time
crime_subplots(crime_act_total, sns.lineplot, 'Date', 'Rate', 'Covid', False, False, legend=False)
plt.suptitle('ACT Crime Rates per 1000 People', fontsize=14)
plt.tight_layout(rect=[0, 0.07, 1, 1])
plt.show()
plt.close('all')

# calculate percentage change in mean crime rates between covid periods
cov_mean = crime_act_total[['Crime', 'Rate', 'Covid']].groupby(['Crime', 'Covid'], as_index=False, observed=True).mean()
cov_mean_change = cov_mean.copy()
cov_mean_change['Percentage Change'] = cov_mean_change.groupby(['Crime'], as_index=False)['Rate'].pct_change() * 100
cov_mean_change = cov_mean_change.drop(columns='Rate')

cov_mean = cov_mean.pivot(index='Crime', columns='Covid', values='Rate')
cov_mean_change = cov_mean_change.pivot(index='Crime', columns='Covid', values='Percentage Change').drop(columns='Pre')
cov_mean_change.columns = ['Change During Covid (%)', 'Change Post-Covid (%)']
cov_mean.columns = ['Pre-Covid Mean Crime Rate', 'Covid Mean Crime Rate', 'Post-Covid Mean Crime Rate']
pct_change = pd.merge(cov_mean, cov_mean_change, left_index=True, right_index=True, how='inner')
print(pct_change)

## explore seasonality trends
seasonal = crime_act_total[['Crime', 'Quarter', 'Rate', 'Covid']].groupby(['Crime', 'Quarter', 'Covid'], as_index=False, observed=True).mean()
seasonal.columns = ['Crime', 'Quarter', 'Covid', 'Mean Quarter Rate']
crime_subplots(seasonal, sns.lineplot, 'Quarter', 'Mean Quarter Rate', 'Covid', x_label=True, y_label=True, legend=False)
plt.suptitle('Quarterly Mean Crime Rate', fontsize=14)
plt.tight_layout(rect=[0, 0.07, 1, 1])
plt.show()
plt.close('all')

#calculate quarterly deviation from mean
mean = crime_act_total[['Crime','Rate', 'Covid']].groupby(['Crime', 'Covid'], as_index=False, observed=True).mean()
mean.columns = ['Crime', 'Covid', 'Mean Period Rate']
seasonal = pd.merge(seasonal, mean, on=['Crime', 'Covid'], how='inner')
seasonal['Percentage Deviation'] = ((seasonal['Mean Quarter Rate'] - seasonal['Mean Period Rate'])/seasonal['Mean Period Rate']) * 100

min_deviation = seasonal[['Crime', 'Covid', 'Percentage Deviation']].groupby(['Crime', 'Covid'], as_index=False, observed=True).min()
max_deviation = seasonal[['Crime', 'Covid', 'Percentage Deviation']].groupby(['Crime', 'Covid'], as_index=False, observed=True).max()
deviation_range = min_deviation[['Crime', 'Covid']].copy()
deviation_range['Deviation Range'] = max_deviation['Percentage Deviation'] - min_deviation['Percentage Deviation']
crime_subplots(deviation_range, sns.barplot, None, 'Deviation Range', 'Covid', x_label=True, y_label=True, legend=False)
plt.suptitle('Quarterly Crime Rate Deviations', fontsize=14)
plt.tight_layout(rect=[0, 0.07, 1, 1])
plt.show()
plt.close('all')
