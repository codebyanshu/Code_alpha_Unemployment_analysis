import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set(style='whitegrid')

DATA = Path('archive') / 'Unemployment_Rate_upto_11_2020.csv'
OUTDIR = Path('.') / 'analysis_outputs'
OUTDIR.mkdir(exist_ok=True)

def load_clean(path=DATA):
    df = pd.read_csv(path)
    
    
    # Normalize column names
    df.columns = [c.strip() for c in df.columns]
    
    
    # Parse dates (day-first in these CSVs)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['Date'])
    
    
    # Convert unemployment to numeric
    df['Estimated Unemployment Rate (%)'] = pd.to_numeric(df['Estimated Unemployment Rate (%)'], errors='coerce')
    df['Estimated Employed'] = pd.to_numeric(df['Estimated Employed'], errors='coerce')
    
    
    # Keep month period for grouping
    df['Month'] = df['Date'].dt.to_period('M').dt.to_timestamp()
    return df


def national_weighted_rate(df):
    
    # weight by employed to get closer to national aggregate
    grp = df.groupby('Month').apply(lambda g: (g['Estimated Unemployment Rate (%)'] * g['Estimated Employed']).sum() / g['Estimated Employed'].sum())
    
    ser = grp.rename('national_unemployment')
    ser.index = pd.to_datetime(ser.index)
    
    return ser.sort_index()


def covid_impact_metrics(national_series):
    
    pre = national_series['2020-01':'2020-03'].mean()
    peak = national_series['2020-04':'2020-05'].max()
    peak_month = national_series['2020-04':'2020-05'].idxmax()
    change_abs = peak - pre
    change_pct = (change_abs / pre) * 100 if pre != 0 else np.nan
    return dict(pre_covid_avg=pre, peak=peak, peak_month=peak_month.strftime('%Y-%m'), abs_change=change_abs, pct_change=change_pct)


def plot_national(series):
    plt.figure(figsize=(10,5))
    series.plot(marker='o')
    plt.title('Estimated National Unemployment Rate (weighted by employed)')
    plt.ylabel('Unemployment Rate (%)')
    plt.tight_layout()
    out = OUTDIR / 'national_unemployment.png'
    plt.savefig(out)
    plt.close()
    return out


def top_states_during_peak(df, months=['2020-04','2020-05'], top_n=10):
    # select months and compute average unemployment per region
    mask = df['Month'].dt.strftime('%Y-%m').isin(months)
    sel = df[mask]
    avg = sel.groupby('Region')['Estimated Unemployment Rate (%)'].mean().sort_values(ascending=False)
    return avg.head(top_n)


def seasonal_summary(df):
    # month-of-year averages across available months
    df['month_num'] = df['Date'].dt.month
    mom = df.groupby('month_num')['Estimated Unemployment Rate (%)'].mean()
    return mom


def main():
    df = load_clean()
    print('Rows loaded:', len(df))
    national = national_weighted_rate(df)
    metrics = covid_impact_metrics(national)
    print('Covid impact metrics:', metrics)

    out_plot = plot_national(national)
    print('Saved national plot to', out_plot)

    top_states = top_states_during_peak(df)
    print('Top affected states (Apr-May 2020):')
    print(top_states)

    season = seasonal_summary(df)
    print('Month-of-year average unemployment (all regions):')
    print(season)

    (OUTDIR / 'covid_metrics.txt').write_text(str(metrics))
    top_states.to_csv(OUTDIR / 'top_states_apr_may_2020.csv')
    season.to_csv(OUTDIR / 'monthly_mean_unemployment.csv')
    print('Analysis outputs written to', OUTDIR)

if __name__ == '__main__':
    main()
