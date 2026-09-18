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


def seasonal_amplitude(crime_df, group_cols):
    """
    Calculates percentage deviation between mean quarterly rates and period rates for each covid period
    Args:
        crime_df: (pd.Dataframe) Crime dataframe (containing at minimum 'Rate' column and columns in group_cols)
        group_cols: (list(str)) Column names used for grouping when calculating mean quarterly rates (must contain 'Quarter' and 'Covid')
    Returns:
            seasonal: (pd.Dataframe) Mean quarterly crime rate for each column value in group_cols
            deviation_range: (pd.Dataframe) Seasonal amplitude (percentage deviation range) 
    """
    filter_cols = group_cols + ['Rate']
    # calculate quarterly mean rates within a covid period
    seasonal = crime_df[filter_cols].groupby(group_cols, as_index=False, observed=True).mean()
    seasonal.columns = group_cols + ['Mean Quarter Rate']
    # calculate covid period mean rates
    period_group = [x for x in group_cols if x not in ['Quarter']]
    period_filter = [x for x in filter_cols if x not in ['Quarter']]
    period_mean = crime_df[period_filter].groupby(period_group, as_index=False, observed=True).mean()
    period_mean.columns = period_group + ['Mean Period Rate']
    # calculate quarterly percentage deviation from period mean
    df = pd.merge(seasonal, period_mean, on=period_group, how='inner')
    df['Percentage Deviation'] = ((df['Mean Quarter Rate'] - df['Mean Period Rate'])/df['Mean Period Rate']) * 100
    min_deviation = df[period_group + ['Percentage Deviation']].groupby(period_group, as_index=False, observed=True).min()
    max_deviation = df[period_group + ['Percentage Deviation']].groupby(period_group, as_index=False, observed=True).max()
    deviation_range = min_deviation[period_group].copy()
    deviation_range['Seasonal Amplitude'] = max_deviation['Percentage Deviation'] - min_deviation['Percentage Deviation']
    return seasonal, deviation_range


def relative_difference(act_df, area_df):
    """
    Calculates relative difference between act-wide crime rates and an area subset of crime rates for each crime, year, and quarter

    Args:
        act_df: (pd.Dataframe) ACT-wide crime dataframe
        area_df: (pd.Dataframe) sub-area crime dataframe (e.g. region- or suburb-level)
    Returns:
            df: (pd.Dataframe) contains sub-area crime rate and relative difference to ACT-wide crime rate for each crime, year, and quarter
    """
    act_df = act_df.rename(columns={'Rate': 'ACT Rate'})
    df = pd.merge(area_df, act_df[['Crime', 'Year', 'Quarter', 'ACT Rate']], on=['Crime', 'Year', 'Quarter'], how='inner')
    df = df.drop(columns=['Population', 'Number', 'Date'])
    # calculate relative difference between regional and ACT-wide crime rates for each crime
    df['Relative Difference'] = ((df['Rate'] - df['ACT Rate']) / df['ACT Rate']) * 100
    # remove division by 0 errors
    df['Relative Difference'] = df['Relative Difference'].fillna(0)
    return df


def covid_pct_change(df, cols):
    """
    Calculates and displays percentage change in crime rates by covid period (not including 'All Crime')

    Args:
        df: (pd.Dataframe) Crime dataframe
        cols: (list(str)) df columns containing (at minimum) 'Crime' and 'Covid', used for grouping, and 'Rate' used for filtering
    Returns:
            pct_change_long: (pd.Dataframe) Long-form dataframe listing 'Rate' and 'Percentage Change' for covid periods under specified columns
            mean_rate: (pd.Dataframe) Long-form dataframe displaying each covid period crime rate under specified columns
    """
    group_cols = cols.copy()
    group_cols.remove('Rate')
    mean_rate = df[cols]
    mean_rate = mean_rate.groupby(group_cols, as_index=False, observed=True).mean()

    mean_rate_change = mean_rate.copy()
    group_cols.remove('Covid')
    mean_rate_change['Percentage Change'] = mean_rate_change.groupby(group_cols, as_index=False)['Rate'].pct_change() * 100
    mean_rate_change = mean_rate_change.drop(columns='Rate')

    pct_change_long = pd.merge(mean_rate, mean_rate_change, on=mean_rate.columns[:-1].tolist(), how='inner')
    mean_rate = mean_rate.pivot(index=group_cols, columns='Covid', values='Rate')
    mean_rate_change = mean_rate_change.pivot(index=group_cols, columns='Covid', values='Percentage Change').drop(columns='Pre').fillna(0) # remove division by 0 error
    mean_rate.columns = ['Pre-Covid Mean Crime Rate', 'Covid Mean Crime Rate', 'Post-Covid Mean Crime Rate']
    mean_rate_change.columns = ['Change During Covid (%)', 'Change Post-Covid (%)']
    
    pct_change = pd.merge(mean_rate, mean_rate_change, left_index=True, right_index=True, how='inner')
    print(pct_change)
    return pct_change_long, mean_rate.reset_index()


def column_iqr(df, group_cols, val_col):
    """
    Calculates high and low outliers using interquartile ranges for a given column under a group
    Displays the number of outliers found for each combination of specified summary columns

    Args:
        df: (pd.Dataframe) Crime dataframe
        group_cols: (list(str)) df column names used for grouping in the aggregation
        val_col: (str) df column name for which outliers should be determined
    """    
    # calculate outlier value ranges using interquartile range
    quantiles = df.groupby(group_cols, as_index=False, observed=True)[val_col].agg(
                                                            Q1=lambda x: x.quantile(0.25),Q3=lambda x: x.quantile(0.75))
    quantiles['IQR'] = quantiles['Q3'] - quantiles['Q1']
    quantiles['Lower'] = quantiles['Q1'] - (1.5 * quantiles['IQR'])
    quantiles['Upper'] = quantiles['Q3'] + (1.5 * quantiles['IQR'])
    quantiles_cols = group_cols + ['Lower', 'Upper']

    # get rows containging high and low outlier values
    extremes = pd.merge(df, quantiles[quantiles_cols], on=group_cols, how='inner')
    high_outlier = extremes[extremes[val_col] > extremes['Upper']]
    low_outlier = extremes[extremes[val_col] < extremes['Lower']]
    return high_outlier, low_outlier


def crime_rate_bump(mean_rate_df, label_col, crime_name):
    """
    Displays a bump chart of crime rate rankings for each covid-period

    Args:
        mean_rate_df: (pd.Dataframe) Long-form dataframe displaying each covid period crime rate
        group_cols: (list(str)) df column names used for grouping in the aggregation
        val_col: (str) df column name for which outliers should be determined
    """       
    mean_rate_all = mean_rate_df[mean_rate_df['Crime'] == crime_name].copy()
    mean_rate_all['Pre Rank'] = mean_rate_all['Pre-Covid Mean Crime Rate'].rank()
    mean_rate_all['During Rank'] = mean_rate_all['Covid Mean Crime Rate'].rank()
    mean_rate_all['Post Rank'] = mean_rate_all['Post-Covid Mean Crime Rate'].rank()
    
    regions = mean_rate_all['Region'].unique()
    region_pal = sns.color_palette('Set2', n_colors=len(regions))
    region_map = dict(zip(regions, region_pal))

    rank_cols = ['Pre Rank', 'During Rank', 'Post Rank']
    suburb_labels = []
    if label_col == 'Suburb':
        f_size = 5
        fig, axs = plt.subplots( figsize=(7, 20))
    elif label_col == 'Region':
        f_size = 7
        fig, axs = plt.subplots(figsize=(7, 8))
    else:
        raise ValueError('Incorrect label given')

    for _, row in mean_rate_all.iterrows():
        ranks = [row[col] for col in rank_cols]
        colour = region_map[row['Region']]

        axs.plot(range(len(rank_cols)), ranks,
                    linewidth=1, alpha=0.6, marker='o', markersize=4, color=colour)

        # Add label
        axs.text(-0.1, ranks[0], row[label_col], ha='right', va='center', fontsize=f_size)

    axs.set_xticks(range(len(rank_cols)))
    axs.set_xticklabels(['Pre', 'During', 'Post'])
    axs.set_xlabel('Covid')
    axs.set_xlim(-0.8, len(rank_cols)-0.5)
    axs.set_ylim(-0.4, len(mean_rate_all)+1)

    axs.yaxis.set_visible(False) 
    axs.set_title(f'Mean {crime_name} Rankings')


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
    fig, axs = plt.subplots(2,4, figsize=(12,7))
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
crime_suburb = pd.read_csv('data/processed/final/suburb/crime_grouped_suburb.csv')
crime_act = crime_region.groupby(['Crime', 'Year', 'Quarter'], as_index=False).sum().drop(columns=['Region'])

crime_region_total = add_total_crime(crime_region)
crime_act_total = add_total_crime(crime_act)
crime_suburb_total = add_total_crime(crime_suburb)

sns.set_palette('Set2', 6)
pal = sns.color_palette('Set2', 6)


## ACT-WIDE EDA ##

print('ACT crime rate statistics between 2014-2025')
quarter_summary_act = crime_act_total.groupby('Crime').agg({'Rate': ['mean', 'median', 'std', 'min', 'max']})
print(quarter_summary_act)

# examine crime rates over time
crime_subplots(crime_act_total, sns.lineplot, 'Date', 'Rate', 'Covid', False, False, legend=False)

plt.suptitle('ACT Crime Rates', fontsize=14)
act_rates_fig = plt.gcf()
act_rates_fig.text(0.5, 0.94, 'Number of Crimes per 1000 People', ha='center', va='top', fontsize=10, color='grey')
plt.tight_layout(rect=[0, 0.07, 1, 0.98])
plt.show()
plt.close('all')

# calculate percentage change during and after covid
print('\nACT-wide crime rate changes by covid period:')
crime_act_pct, act_mean_rate = covid_pct_change(crime_act_total, ['Crime', 'Rate', 'Covid'])
crime_subplots(crime_act_pct, sns.barplot, None, 'Rate', 'Covid', False, True, legend=False, width=0.9)

fig = plt.gcf()
for i, ax in enumerate(fig.get_axes()):
    #print(round(crime_act_pct.iloc[(3*i)+1, 3], 1))
    changes = [round(crime_act_pct.iloc[(3*i)+1, 3], 1), round(crime_act_pct.iloc[(3*i)+2, 3], 1)]
    str_changes = [f'{x}%' if x < 0 else f'+{x}%' for x in changes]

    ax.text(0.5, 0.02, str_changes[0], transform=ax.transAxes, ha='center', size=9, weight='bold', c='white')
    ax.text(0.8, 0.02, str_changes[1], transform=ax.transAxes, ha='center', size=9, weight='bold', c='white')
plt.suptitle('Mean ACT Crime Rates and Percentage Changes for Each Covid Period', fontsize=14)
plt.tight_layout(rect=[0, 0.07, 1, 1])
plt.show()
plt.close('all')


## explore seasonality trends
act_seasonal, act_seasonal_amp = seasonal_amplitude(crime_act_total, ['Crime', 'Quarter', 'Covid'])
print(act_seasonal.head())
print(act_seasonal_amp.head())

crime_subplots(act_seasonal, sns.lineplot, 'Quarter', 'Mean Quarter Rate', 'Covid', x_label=True, y_label=False, legend=False)
plt.suptitle('ACT Mean Quarterly Crime Rates by Covid Period', fontsize=12)
plt.tight_layout(rect=[0, 0.07, 1, 1])
plt.show()
plt.close('all')

crime_subplots(act_seasonal_amp, sns.barplot, None, 'Seasonal Amplitude', 'Covid', x_label=True, y_label=True, legend=False)
plt.suptitle('ACT Quarterly Crime Rate Deviations by Covid Period', fontsize=12)
plt.tight_layout(rect=[0, 0.07, 1, 1])
plt.show()
plt.close('all')


## REGION-LEVEL EDA ##

# calculate relative difference between regional and act crime rates for each crime
region_rel_diff = relative_difference(crime_act_total, crime_region_total)

# calculate mean regional rates and relative difference by covid period
covid_region_comparison = region_rel_diff.drop(columns=['Quarter', 'Year'])
covid_region_comparison = covid_region_comparison.groupby(['Crime', 'Region','Covid'], as_index=False, observed=True).mean()


# display five max and min rates
covid_region_comparison = covid_region_comparison.sort_values(by='Rate')
print('\nGreatest absolute regional mean crime rates by covid period:')
print(covid_region_comparison.head())
print(covid_region_comparison.tail())

region_subplots(covid_region_comparison.sort_values(by='Region'), 'All Crime', sns.barplot, None, 'Rate', 'Covid', x_label=True, y_label=True, legend=False)
plt.suptitle('Mean Regional Crime Rates by Covid Period', fontsize=14)
plt.tight_layout(rect=[0, 0.07, 1, 1])
plt.show()
plt.close('all')

# display five max and min relative differences
covid_region_comparison = covid_region_comparison.sort_values(by='Relative Difference')
print('\nGreatest absolute regional mean relative crime rate differences covid period:')
print(covid_region_comparison.head())
print(covid_region_comparison.tail())

region_subplots(covid_region_comparison.sort_values(by='Region'), 'All Crime', sns.barplot, None, 'Relative Difference', 'Covid', x_label=True, y_label=True, legend=False)
plt.suptitle('Mean Regional Differences to Total ACT Crime Rate by Covid Period', fontsize=14)
plt.tight_layout(rect=[0, 0.07, 1, 1])
plt.show()
plt.close('all')

# find relative difference outliers
print('\nNumber of relative difference outliers')
covid_region_comparison_individual = covid_region_comparison[covid_region_comparison['Crime'] != 'All Crime']
h_region_rel_diff, l_region_rel__diff = column_iqr(covid_region_comparison_individual, ['Crime', 'Covid'], 'Relative Difference')
print('High outliers:')
print(h_region_rel_diff.value_counts(subset=['Crime', 'Covid']).sort_index())
print(h_region_rel_diff.value_counts(subset=['Region', 'Covid']).sort_index())
print('Low outliers:')
print(l_region_rel__diff.value_counts(subset=['Crime', 'Covid']).sort_index())
print(l_region_rel__diff.value_counts(subset=['Region', 'Covid']).sort_index())

# calculate percentage change during and after covid
print('\nRegional crime rate changes by covid period:')
crime_region_pct, region_mean_rate = covid_pct_change(crime_region_total, ['Crime', 'Region', 'Rate', 'Covid'])
crime_rate_bump(region_mean_rate, 'Region', 'All Crime')
plt.title('Regional Rankings for Total Crime Rate by Covid Period')
plt.tight_layout(pad=3)
plt.show()
plt.close('all')