import pandas as pd
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from covid_dates import add_covid_col

def datetime(df):
    """
    Get datetime column from 'Year' and 'Quarter' columns

    Args:
        df: (pd.Dataframe) dataframe containing 'Year' and 'Quarter' columns

    Returns:
        df: (pd.Dataframe) Updated dataframe including pd.datetime ('Date') column
    """
    y_q = df['Year'].astype(str) + '-' + df['Quarter']
    df['Date'] = pd.PeriodIndex(y_q, freq='Q').to_timestamp()
    return df


def add_total_crime(df):
    """
    Formats crime dataframe columns and includes statistics for all crime types

    Args:
        df: (pd.Dataframe) Processed crime dataframe
        
    Returns:
        df_total: (pd.Dataframe) Updated dataframe with formatted 'Date' and 'Covid' columns
                                    with 'All Crime' included in crime types
    """
    if 'Covid' in df.columns:
        df = df.drop(columns='Covid')
    group_cols = df.columns.tolist()
    group_cols.remove('Crime')
    group_cols.remove('Number')
    group_cols.remove('Rate')
    df_all = df.groupby(group_cols, as_index=False).sum().drop(columns=['Crime', 'Rate'])
    # specify time in relation to covid
    df = add_covid_col(df)
    df_all = add_covid_col(df_all)
    # recalculate rates if necessary
    if 'Region' not in group_cols:
        df['Rate'] = df['Number']/(df['Population']/1000)
    df_all['Rate'] = df_all['Number']/(df_all['Population']/1000)
    # add datetime format column
    df = datetime(df)
    df_all = datetime(df_all)
    #combine data
    df_all.insert(0, 'Crime', ['All Crime'] * len(df_all))
    df_total = pd.concat([df_all, df])
    return df_total

def covid_pct_change(df, cols):
    """
    Calculates and displays percentage change in crime rates by covid period

    Args:
        df: (pd.Dataframe) Crime dataframe
        cols: (list(str)) df columns containing (at minimum) ['Crime', 'Rate', 'Covid'], used for filtering and grouping
        
    """
    group_cols = cols.copy()
    group_cols.remove('Rate')
    mean_rate = df[cols]
    mean_rate = mean_rate[mean_rate['Crime'] != 'All Crime']
    mean_rate = mean_rate.groupby(group_cols, as_index=False, observed=True).mean()

    mean_rate_change = mean_rate.copy()
    mean_rate_change['Percentage Change'] = mean_rate_change.groupby(['Crime'], as_index=False)['Rate'].pct_change() * 100
    mean_rate_change = mean_rate_change.drop(columns='Rate')

    group_cols.remove('Covid')
    mean_rate = mean_rate.pivot(index=group_cols, columns='Covid', values='Rate')
    mean_rate_change = mean_rate_change.pivot(index=group_cols, columns='Covid', values='Percentage Change').drop(columns='Pre').fillna(0) # remove division by 0 error
    mean_rate.columns = ['Pre-Covid Mean Crime Rate', 'Covid Mean Crime Rate', 'Post-Covid Mean Crime Rate']
    mean_rate_change.columns = ['Change During Covid (%)', 'Change Post-Covid (%)']
    
    pct_change = pd.merge(mean_rate, mean_rate_change, left_index=True, right_index=True, how='inner')
    print(pct_change)


def crime_subplots(df, plot_func, x, y, h, x_label=True, y_label=True, **kwargs):
    """
    Creates and formats a subplot per crime  

    Args:
        df: (pd.Dataframe) Crime dataframe
        plot_func: (seaborn function) axis-level plot type
        x: (str/NoneType) column corresponding to x-axis
        y: (str/NoneType) column corresponding to y-axis
        h: (str/NoneType) column corresponding to hue
        x_label: (Boolean) specify whether to display x-axis label
        y_label: (Boolean) specify whether to display y-axis label
        **kwargs: optional additional keyword arguments passed to plot_func
    """
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
        
def region_subplots(df, crime_name, plot_func, x, y, h, x_label=True, y_label=True, **kwargs):
    """
    Creates and formats a subplot region for a given crime type  

    Args:
        df: (pd.Dataframe) Crime dataframe
        crime_name: (str) specify which value of 'Crime' column to plot
        plot_func: (seaborn function) axis-level plot type
        x: (str/NoneType) column corresponding to x-axis
        y: (str/NoneType) column corresponding to y-axis
        h: (str/NoneType) column corresponding to hue
        x_label: (Boolean) specify whether to display x-axis label
        y_label: (Boolean) specify whether to display y-axis label
        **kwargs: optional additional keyword arguments passed to plot_func
    """
    fig, axs = plt.subplots(2,4, figsize=(12,6))
    for i, ax in enumerate(axs.flat):
        reg_title = df['Region'].unique()[i]
        region = df[(df['Crime'] == crime_name) & (df['Region'] == reg_title)]
        ax.set_title(reg_title)
        plot_func(data=region, x=x, y=y, hue=h, ax=ax, **kwargs)
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
crime_act = crime_region.groupby(['Crime', 'Year', 'Quarter'], as_index=False).sum().drop(columns=['Region'])


crime_region_total = add_total_crime(crime_region)
crime_act_total = add_total_crime(crime_act)

sns.set_palette('Set2', 6)
pal = sns.color_palette('Set2', 6)



## ACT-WIDE EDA ##

# examine crime rates over time
crime_subplots(crime_act_total, sns.lineplot, 'Date', 'Rate', 'Covid', False, False, legend=False)
plt.suptitle('ACT Crime Rates per 1000 People', fontsize=14)
plt.tight_layout(rect=[0, 0.07, 1, 1])
plt.show()
plt.close('all')

# calculate percentage change during and after covid
covid_pct_change(crime_act_total, ['Crime', 'Rate', 'Covid'])


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



## REGION-LEVEL EDA ##

# calculate relative difference between regional and ACT-wide crime rates for each crime
crime_act_total = crime_act_total.rename(columns={'Rate': 'ACT Rate'})
region_comparison = pd.merge(crime_region_total, crime_act_total[['Crime', 'Year', 'Quarter', 'ACT Rate']], on=['Crime', 'Year', 'Quarter'], how='inner')
region_comparison = region_comparison.drop(columns=['Population', 'Number', 'Date'])
region_comparison['Relative Difference'] = ((region_comparison['Rate'] - region_comparison['ACT Rate']) / region_comparison['ACT Rate']) * 100
# remove division by 0 errors
region_comparison['Relative Difference'] = region_comparison['Relative Difference'].fillna(0)

covid_region_comparison = region_comparison.drop(columns=['Quarter', 'Year'])
covid_region_comparison = covid_region_comparison.groupby(['Crime', 'Region','Covid'], as_index=False, observed=True).mean()
covid_region_comparison = covid_region_comparison.sort_values(by='Relative Difference')
# display five max and min relative differences
print(covid_region_comparison.head())
print(covid_region_comparison.tail())

region_subplots(covid_region_comparison.sort_values(by='Region'), 'All Crime', sns.barplot, None, 'Relative Difference', 'Covid', x_label=True, y_label=True, legend=False)
plt.suptitle('Regional Differences to ACT-Wide Crime Rate for All Crime', fontsize=14)
plt.tight_layout(rect=[0, 0.07, 1, 1])
plt.show()
plt.close('all')

crime_subplots(covid_region_comparison.sort_values(by='Crime'), sns.kdeplot, 'Relative Difference', None, 'Covid', x_label=True, y_label=True, legend=False)
plt.suptitle('Mean Regional Differences to ACT-Wide Crime Rate', fontsize=14)
plt.tight_layout(rect=[0, 0.07, 1, 1])
plt.show()
plt.close('all')


# calculate percentage change during and after covid
covid_pct_change(crime_region_total, ['Crime', 'Region', 'Rate', 'Covid'])