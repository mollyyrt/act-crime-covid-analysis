import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.eda_funcs import (
    seasonal_amplitude,
    relative_difference,
    covid_pct_change,
    column_iqr,
    crime_rate_bump, 
    crime_subplots,
    region_subplots,
    regional_vs_suburb_plot
)


source_path = Path('data/processed')

crime_act = pd.read_csv(source_path / 'crime_act.csv')
crime_act['Date'] = pd.to_datetime(crime_act['Date'], format='%Y-%m-%d')
crime_act['COVID'] = pd.Categorical(crime_act['COVID'], categories=['Pre', 'During', 'Post'], ordered=True)

crime_region = pd.read_csv(source_path / 'crime_region.csv')
crime_region['Date'] = pd.to_datetime(crime_region['Date'], format='%Y-%m-%d')
crime_region['COVID'] = pd.Categorical(crime_region['COVID'], categories=['Pre', 'During', 'Post'], ordered=True)

crime_suburb = pd.read_csv(source_path / 'crime_suburb.csv')
crime_suburb['Date'] = pd.to_datetime(crime_suburb['Date'], format='%Y-%m-%d')
crime_suburb['COVID'] = pd.Categorical(crime_suburb['COVID'], categories=['Pre', 'During', 'Post'], ordered=True)

sns.set_palette('Set2', 6)


## ACT-WIDE EDA ##

print('ACT crime rate statistics between 2014-2025')
quarter_summary_act = crime_act.groupby('Crime').agg({'Rate': ['mean', 'median', 'std', 'min', 'max']})
print(quarter_summary_act)

# examine crime rates over time
crime_subplots(crime_act, sns.lineplot, 'Date', 'Rate', 'COVID', False, False, legend=False)

plt.suptitle('ACT Crime Rates', fontsize=14)
act_rates_fig = plt.gcf()
act_rates_fig.text(0.5, 0.94, 'Number of Crimes per 1000 People', ha='center', va='top', fontsize=10, color='grey')
plt.tight_layout(rect=[0, 0.07, 1, 0.98])
plt.show()
plt.close('all')

# calculate percentage change during and after COVID
print('\nACT-wide crime rate changes by COVID period:')
crime_act_pct, act_mean_rate = covid_pct_change(crime_act, ['Crime', 'Rate', 'COVID'])
crime_subplots(crime_act_pct, sns.barplot, None, 'Rate', 'COVID', False, True, legend=False, width=0.9)

fig = plt.gcf()
for i, ax in enumerate(fig.get_axes()):
    #print(round(crime_act_pct.iloc[(3*i)+1, 3], 1))
    changes = [round(crime_act_pct.iloc[(3*i)+1, 3], 1), round(crime_act_pct.iloc[(3*i)+2, 3], 1)]
    str_changes = [f'{x}%' if x < 0 else f'+{x}%' for x in changes]

    ax.text(0.5, 0.025, str_changes[0], transform=ax.transAxes, ha='center', size=9, weight='bold', c='white')
    ax.text(0.8, 0.025, str_changes[1], transform=ax.transAxes, ha='center', size=9, weight='bold', c='white')
plt.suptitle('Mean ACT Crime Rates and Percentage Changes for Each COVID Period', fontsize=14)
plt.tight_layout(rect=[0, 0.07, 1, 1])
plt.show()
plt.close('all')

## explore seasonality trends
act_seasonal, act_seasonal_amp = seasonal_amplitude(crime_act, ['Crime', 'Quarter', 'COVID'])
print(act_seasonal.head())
print(act_seasonal_amp.head())

crime_subplots(act_seasonal, sns.lineplot, 'Quarter', 'Mean Quarter Rate', 'COVID', x_label=True, y_label=False, legend=False)
plt.suptitle('ACT Mean Quarterly Crime Rates by COVID Period', fontsize=12)
plt.tight_layout(rect=[0, 0.07, 1, 1])
plt.show()
plt.close('all')

crime_subplots(act_seasonal_amp, sns.barplot, None, 'Seasonal Amplitude', 'COVID', x_label=True, y_label=True, legend=False)
plt.suptitle('ACT Quarterly Crime Rate Deviations by COVID Period', fontsize=12)
plt.tight_layout(rect=[0, 0.07, 1, 1])
plt.show()
plt.close('all')


## REGION-LEVEL EDA ##

# calculate relative difference between regional and act crime rates for each crime
region_rel_diff = relative_difference(crime_act, crime_region)

# calculate mean regional rates and relative difference by COVID period
covid_region_comparison = region_rel_diff.drop(columns=['Quarter', 'Year'])
covid_region_comparison = covid_region_comparison.groupby(['Crime', 'Region','COVID'], as_index=False, observed=True).mean()


# display five max and min rates
covid_region_comparison = covid_region_comparison.sort_values(by='Rate')
print('\nGreatest absolute regional mean crime rates by COVID period:')
print(covid_region_comparison.head())
print(covid_region_comparison.tail())

region_subplots(covid_region_comparison.sort_values(by='Region'), 'All Crime', sns.barplot, None, 'Rate', 'COVID', x_label=True, y_label=True, legend=False)
plt.suptitle('Mean Regional Crime Rates by COVID Period', fontsize=14)
plt.tight_layout(rect=[0, 0.07, 1, 1])
plt.show()
plt.close('all')

# display five max and min relative differences
covid_region_comparison = covid_region_comparison.sort_values(by='Relative Difference')
print('\nGreatest absolute regional mean relative crime rate differences COVID period:')
print(covid_region_comparison.head())
print(covid_region_comparison.tail())

region_subplots(covid_region_comparison.sort_values(by='Region'), 'All Crime', sns.barplot, None, 'Relative Difference', 'COVID', x_label=True, y_label=True, legend=False)
plt.suptitle('Mean Regional Differences to Total ACT Crime Rate by COVID Period', fontsize=14)
plt.tight_layout(rect=[0, 0.07, 1, 1])
plt.show()
plt.close('all')

# find relative difference outliers
print('\nNumber of relative difference outliers')
covid_region_comparison_individual = covid_region_comparison[covid_region_comparison['Crime'] != 'All Crime']
h_region_rel_diff, l_region_rel__diff = column_iqr(covid_region_comparison_individual, ['Crime', 'COVID'], 'Relative Difference')
print('High outliers:')
print(h_region_rel_diff.value_counts(subset=['Crime', 'COVID']).sort_index())
print(h_region_rel_diff.value_counts(subset=['Region', 'COVID']).sort_index())
print('Low outliers:')
print(l_region_rel__diff.value_counts(subset=['Crime', 'COVID']).sort_index())
print(l_region_rel__diff.value_counts(subset=['Region', 'COVID']).sort_index())

# calculate percentage change during and after COVID
print('\nRegional crime rate changes by COVID period:')
crime_region_pct, region_mean_rate = covid_pct_change(crime_region, ['Crime', 'Region', 'Rate', 'COVID'])
crime_rate_bump(region_mean_rate, 'Region', 'All Crime')
plt.title('Regional Rankings for Total Crime Rate by COVID Period')
plt.tight_layout(pad=3)
plt.show()
plt.close('all')


# regional seasonality
region_seasonal, region_seasonal_amp = seasonal_amplitude(crime_region, ['Crime', 'Region', 'Quarter', 'COVID'])

region_subplots(region_seasonal, 'All Crime', sns.lineplot, 'Quarter', 'Mean Quarter Rate', 'COVID', x_label=True, y_label=False, legend=False)
plt.suptitle('Regional Mean Quarterly Total Crime Rate by Covid Period', fontsize=12)
plt.tight_layout(rect=[0, 0.07, 1, 1])
plt.show()
plt.close('all')

g = sns.catplot(region_seasonal_amp[region_seasonal_amp['Crime'] == 'All Crime'], x='Region', y='Seasonal Amplitude', row='COVID', hue='Region', kind='bar', height=2.4, aspect=4, zorder=2, legend=False)
for ax in g.axes.flatten():
    ax.grid(axis='y', zorder=1, linestyle='--', linewidth=0.5)
plt.xticks(rotation=25)
plt.suptitle('Regional Quarterly Seasonal Amplitude for All Crime by COVID Period', fontsize=12)
plt.tight_layout()
plt.show()
plt.close('all')



## SUBURB-LEVEL EDA ##

# explore suburb level crimes over time
crime_subplots(crime_suburb, sns.lineplot, 'Date', 'Rate', 'COVID', False, False, legend=False)
plt.suptitle('Suburb Crime Rates over Time')
plt.tight_layout(rect=[0, 0.07, 1, 1])
plt.show()
plt.close('all')

# display number of outliers for each crime
crime_suburb_covid = crime_suburb.drop(columns=['Quarter', 'Population', 'Date'])
crime_suburb_covid = crime_suburb_covid.groupby(['Crime', 'Region', 'Suburb', 'COVID'], as_index=False, observed=True).mean()
high_suburb_rate, low_suburb_rate = column_iqr(crime_suburb_covid, ['Crime', 'COVID'], 'Rate')

plt.figure(figsize=(12, 5))
sns.countplot(high_suburb_rate, x='Crime', hue='COVID', zorder=2)
plt.grid(axis='y', zorder=1, linestyle='--', linewidth=0.5)
plt.xticks(rotation=30)
plt.suptitle('Number of Unusually High Suburb Crime Rates by COVID Period', fontsize=14)
plt.title('Outliers identified among quarterly observations across 103 suburbs', fontsize=10, color='grey')
plt.tight_layout()
plt.show()
plt.clf()
sys.exit()


# compare total crime rates/number vs population within suburbs
g = sns.jointplot(data=crime_suburb[crime_suburb['Crime'] != 'All Crime'], x='Population', y='Rate', hue='Covid')
plt.suptitle('Suburb Crime Rates by Population and COVID Period', fontsize=12)
g.figure.text(0.5, 0.94, 'Quarterly Crime Rates Across 103 Suburbs', ha='center', va='top', fontsize=10, color='grey')
plt.tight_layout(rect=[0, 0.07, 1, 0.98])
plt.show()
plt.close('all')

g = sns.jointplot(data=crime_suburb[crime_suburb['Crime'] != 'All Crime'], x='Population', y='Number', hue='COVID')
plt.suptitle('Suburb Crime Occurences by Population and COVID Period', fontsize=12)
g.figure.text(0.5, 0.94, 'Quarterly Crime Counts Across 103 Suburbs', ha='center', va='top', fontsize=10, color='grey')
plt.tight_layout(rect=[0, 0.07, 1, 0.98])
plt.show()
plt.close('all')

# add flag for suburbs (e.g. industrial/commercial) with low populations which are skewing data
suburb_mean_pop = crime_suburb[['Suburb', 'Population', 'Rate']].groupby('Suburb').mean()
low_pop = suburb_mean_pop[suburb_mean_pop['Population'] < 300].index.tolist()
crime_suburb['Low Population'] = np.where(crime_suburb['Suburb'].isin(low_pop), 1, 0)

g = sns.FacetGrid(crime_suburb[crime_suburb['Crime'] != 'All Crime'], col='COVID',  row='Low Population', sharex=False, sharey='row')
g.map_dataframe(sns.scatterplot,  x='Population', y='Rate', hue='COVID')
plt.suptitle('Suburb Crime Rates by Population and COVID Period', fontsize=12)
g.figure.text(0.5, 0.94, 'Separated by Mean Populations of Under 300', ha='center', va='top', fontsize=10, color='grey')
plt.tight_layout(rect=[0, 0.07, 1, 0.98])
plt.show()
plt.close('all')


# calculate relative difference
suburb_rel_diff = relative_difference(crime_act, crime_suburb)

# compare suburb-level relative differences to regional means
fig, axs = plt.subplots(3, 1, figsize=(10, 10))
regional_vs_suburb_plot(suburb_rel_diff, covid_region_comparison, 'All Crime', 'Belconnen', 'Pre', 'Relative Difference', axs[0])
regional_vs_suburb_plot(suburb_rel_diff, covid_region_comparison, 'All Crime', 'Belconnen', 'During', 'Relative Difference', axs[1])
regional_vs_suburb_plot(suburb_rel_diff, covid_region_comparison, 'All Crime', 'Belconnen', 'Post', 'Relative Difference', axs[2])
axs[0].tick_params(labelbottom=False)
axs[1].tick_params(labelbottom=False)
axs[2].tick_params(axis='x', labelrotation=45, labelsize=8)
plt.suptitle('Belconnen: Comparison of Suburb Relative Differences and Region Mean by COVID Period')
plt.tight_layout(rect=[0, 0.07, 1, 1])
plt.show()
plt.close('all')

gungahlin_filtered = suburb_rel_diff[~suburb_rel_diff['Suburb'].isin(low_pop)]
fig, axs = plt.subplots(3, 2, figsize=(16, 10))
regional_vs_suburb_plot(suburb_rel_diff, covid_region_comparison, 'All Crime', 'Gungahlin', 'Pre', 'Relative Difference', axs[0][0])
regional_vs_suburb_plot(suburb_rel_diff, covid_region_comparison, 'All Crime', 'Gungahlin', 'During', 'Relative Difference', axs[1][0])
regional_vs_suburb_plot(suburb_rel_diff, covid_region_comparison, 'All Crime', 'Gungahlin', 'Post', 'Relative Difference', axs[2][0])
regional_vs_suburb_plot(gungahlin_filtered, covid_region_comparison, 'All Crime', 'Gungahlin', 'Pre', 'Relative Difference', axs[0][1])
regional_vs_suburb_plot(gungahlin_filtered, covid_region_comparison, 'All Crime', 'Gungahlin', 'During', 'Relative Difference', axs[1][1])
regional_vs_suburb_plot(gungahlin_filtered, covid_region_comparison, 'All Crime', 'Gungahlin', 'Post', 'Relative Difference', axs[2][1])
for i in [0,1]:
    axs[0][i].tick_params(labelbottom=False)
    axs[1][i].tick_params(labelbottom=False)
    axs[2][i].tick_params(axis='x', labelrotation=45, labelsize=8)
for i in [0,1,2]:
    axs[i][1].set_ylabel(None)
axs[0][0].annotate('All Suburbs', xy=(0.5, 1.05), xycoords='axes fraction', fontsize=10, va='center', ha='center')
axs[0][1].annotate('Not Including Low-Population Suburbs', xy=(0.5, 1.05), xycoords='axes fraction', fontsize=10, va='center', ha='center')
plt.suptitle('Gungahlin: Comparison of Suburb Relative Differences and Region Mean by COVID Period')
plt.tight_layout(rect=[0, 0.07, 1, 0.95])
plt.show()
plt.close('all')

# relative difference outliers
suburb_comparison_covid = suburb_rel_diff.drop(columns=['Quarter', 'ACT Rate', 'Year'])
suburb_comparison_covid = suburb_comparison_covid.groupby(['Crime', 'Region', 'Suburb', 'COVID'], as_index=False, observed=True).mean()

high_suburb_rel_diff, low_suburb_rel_diff = column_iqr(suburb_comparison_covid, ['Crime', 'COVID'], 'Relative Difference')
print('\nNumber of extreme suburb relative differences in each COVID period')
print('High outliers:')
print(high_suburb_rel_diff.value_counts(subset=['Crime', 'Covid']).sort_index())
print(high_suburb_rel_diff.value_counts(subset=['Covid']).sort_index())
print('Low outliers:')
print(low_suburb_rel_diff.value_counts(subset=['Crime', 'Covid']).sort_index())
print(low_suburb_rel_diff.value_counts(subset=['Covid']).sort_index())


# percentage change between suburb and ACT-wide crime rates
suburb_pct, suburb_mean_rate = covid_pct_change(crime_suburb, ['Crime', 'Region', 'Suburb', 'COVID', 'Rate'])

suburb_pct_change_only = suburb_pct[suburb_pct['COVID'] != 'Pre']
h_suburb_pct_change, l_suburb_pct_change= column_iqr(suburb_pct[suburb_pct['COVID'] != 'Pre'], ['Crime', 'COVID'], 'Percentage Change')
print('\nNumber of extreme percentage change values between covid periods')
print('High outliers:')
print(h_suburb_pct_change.value_counts(subset=['Crime', 'COVID']).sort_index())
print(h_suburb_pct_change.value_counts(subset=['COVID']).sort_index())
print('Low outliers:')
print(l_suburb_pct_change.value_counts(subset=['Crime', 'COVID']).sort_index())
print(l_suburb_pct_change.value_counts(subset=['COVID']).sort_index())

crime_rate_bump(suburb_mean_rate, 'Suburb', 'All Crime')
plt.title('Suburb Rankings for Total Crime Rate by COVID Period')
plt.tight_layout(pad=3)
plt.show()
plt.close('all')

crime_subplots(suburb_pct_change_only, sns.kdeplot, 'Percentage Change', None, 'COVID', x_label=True, y_label=True, legend=False)
plt.suptitle('Mean Percentage Change in Suburb Crime Rates by COVID Period')
plt.tight_layout(rect=[0, 0.07, 1, 1])
plt.show()
plt.close('all')


# suburb-level seasonality
suburb_seasonal, suburb_seasonal_amp = seasonal_amplitude(crime_suburb, ['Crime', 'Region', 'Suburb', 'Quarter', 'COVID'])

crime_subplots(suburb_seasonal, sns.lineplot, 'Quarter', 'Mean Quarter Rate', 'COVID', x_label=True, y_label=True, legend=False)
plt.suptitle('Suburb Crime Seasonal Amplitudes by COVID Period')
plt.tight_layout(rect=[0, 0.07, 1, 1])
plt.show()
plt.close('all')

crime_subplots(suburb_seasonal_amp, sns.boxplot, None, 'Seasonal Amplitude', 'COVID', x_label=True, y_label=True, legend=False)
plt.suptitle('Suburb Crime Seasonal Amplitudes by COVID Period')
plt.tight_layout(rect=[0, 0.07, 1, 1])
plt.show()
plt.close('all')

