import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import warnings

warnings.filterwarnings('ignore')

from preprocessing import load_and_clean_auxiliary_data, merge_and_impute_data

DATA_PATH = "."

def extract_required_time_features(df):
    df['date'] = pd.to_datetime(df['date'])
    df['Year'] = df['date'].dt.year
    df['Month'] = df['date'].dt.month
    df['DayOfWeek'] = df['date'].dt.dayofweek
    df['day_type'] = np.where(df['DayOfWeek'].isin([5, 6]), 'Weekend', 'Weekday')
    return df

def descriptive_stats_and_quality_check(df):
    print("## Descriptive Statistics and Data Quality Check")
    print("--------------------------------------------------")
    
    print(f"Merged Data Shape (Rows, Columns): {df.shape}")
    
    missing_values = df.isnull().sum()
    missing_percent = (missing_values[missing_values > 0] / len(df)) * 100
    missing_df = pd.DataFrame({'Missing Count': missing_values[missing_values > 0],
                               'Missing %': missing_percent.round(2)})
    
    if not missing_df.empty:
        print("\nMissing Values per Column:")
        print(missing_df.sort_values(by='Missing %', ascending=False))
    else:
        print("\nNo missing values in the merged data.")
    
    print("\nDescriptive statistics for numerical columns:")
    print(df[['sales', 'onpromotion', 'transactions', 'dcoilwtico']].describe().T)

def sales_aggregation_plots(train):
    df_day_type = train.groupby(['date', 'day_type'])['sales'].sum().reset_index()
    df_day_type = df_day_type.groupby('day_type').agg(mean_sales=('sales','mean'), max_sales=('sales','max')).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.barplot(x='day_type', y='mean_sales', data=df_day_type, ax=axes[0])
    axes[0].set_title("Average Sales per Day Type", pad=10)
    axes[0].set_xlabel("Day Type")
    axes[0].set_ylabel("Average Sales")
    for container in axes[0].containers:
        axes[0].bar_label(container, fmt="%.2f", padding=3)
    axes[0].set_ylim(0, df_day_type['mean_sales'].max() * 1.2)
    
    sns.barplot(x='day_type', y='max_sales', data=df_day_type, ax=axes[1])
    axes[1].set_title("Max Sales per Day Type", pad=10)
    axes[1].set_xlabel("Day Type")
    axes[1].set_ylabel("Max Sales")
    for container in axes[1].containers:
        axes[1].bar_label(container, fmt="%.2f", padding=3)
    axes[1].set_ylim(0, df_day_type['max_sales'].max() * 1.2)
    
    plt.tight_layout()
    plt.show()

    df_store_type = train.groupby(['date', 'store_type'])['sales'].sum().reset_index()
    df_store_type = df_store_type.groupby('store_type').agg(mean_sales=('sales','mean'), max_sales=('sales','max')).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.barplot(x='store_type', y='mean_sales', data=df_store_type, ax=axes[0])
    axes[0].set_title("Average Sales per Store Type", pad=10)
    axes[0].set_xlabel("Store Type")
    axes[0].set_ylabel("Average Sales")
    for container in axes[0].containers:
        axes[0].bar_label(container, fmt="%.2f", padding=3)
    axes[0].set_ylim(0, df_store_type['mean_sales'].max() * 1.2)
    
    sns.barplot(x='store_type', y='max_sales', data=df_store_type, ax=axes[1])
    axes[1].set_title("Max Sales per Store Type", pad=10)
    axes[1].set_xlabel("Store Type")
    axes[1].set_ylabel("Max Sales")
    for container in axes[1].containers:
        axes[1].bar_label(container, fmt="%.2f", padding=3)
    axes[1].set_ylim(0, df_store_type['max_sales'].max() * 1.2)
    
    plt.tight_layout()
    plt.show()

def sales_trend_analysis(train):
    df_yearly_store_type = train.groupby(['Year', 'store_type'])['sales'].sum().reset_index()
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    axes = axes.flatten()

    temp_list = df_yearly_store_type['store_type'].unique().tolist()
    
    for idx, store in enumerate(temp_list):
        temp = df_yearly_store_type[df_yearly_store_type['store_type'] == store]
        
        sns.barplot(x='Year', y='sales', data=temp, ax=axes[idx])
        
        axes[idx].set_title(f'Total Sales of Store Type {store} across Years', pad=10)
        axes[idx].set_xlabel('Year')
        axes[idx].set_ylabel('Total Sales')
        
        for container in axes[idx].containers:
            axes[idx].bar_label(container, fmt="%.2f", padding=3)

        axes[idx].set_ylim(0, df_yearly_store_type['sales'].max() * 1.2)

    for j in range(len(temp_list), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

    df_sales_timeline = train.groupby(['Year', 'Month'])['sales'].sum().reset_index()
    plt.figure(figsize=(20,5))
    df_sales_timeline['Year-Month'] = df_sales_timeline['Year'].astype(str) + '-' + df_sales_timeline['Month'].astype(str).str.zfill(2)

    custom_palette = {
        2013: "#1f77b4",
        2014: "#ff7f0e",
        2015: "#2ca02c",
        2016: "#d62728",
        2017: "#9467bd"
    }

    sns.lineplot(x='Year-Month', y='sales', hue='Year', data=df_sales_timeline, marker="o", palette = custom_palette)

    plt.xticks(rotation=45)
    plt.title("Sales Timeline by Year")
    plt.xlabel("Year-Month")
    plt.ylabel("Total Sales")
    plt.legend(title="Year")
    plt.show()

def oil_price_analysis(train):
    df_oil_timeline = train.groupby(['Year', 'Month'])['dcoilwtico'].mean().reset_index()
    plt.figure(figsize=(20,5))
    df_oil_timeline['Year-Month'] = df_oil_timeline['Year'].astype(str) + '-' + df_oil_timeline['Month'].astype(str).str.zfill(2)

    custom_palette = {
        2013: "#1f77b4",
        2014: "#ff7f0e",
        2015: "#2ca02c",
        2016: "#d62728",
        2017: "#9467bd"
    }

    sns.lineplot(x='Year-Month', y='dcoilwtico', hue='Year', data=df_oil_timeline, marker="o", palette = custom_palette)

    plt.xticks(rotation=45)
    plt.title("dcoilwtico Timeline by Year")
    plt.xlabel("Year-Month")
    plt.ylabel("dcoilwtico price")
    plt.legend(title="Year")
    plt.show()

    df_oil_summary = train[['Year', 'Month', 'dcoilwtico']].copy()
    df_oil_summary = df_oil_summary.groupby(['Year', 'Month'])['dcoilwtico'].mean().unstack()

    df_oil_summary['Avg'] = df_oil_summary.mean(axis=1)
    df_oil_summary['Max month'] = df_oil_summary.idxmax(axis=1)
    df_oil_summary['Max price'] = df_oil_summary.drop(columns=['Max month']).max(axis=1)
    df_oil_summary['Min month'] = df_oil_summary.drop(columns=['Max month', 'Max price']).idxmin(axis=1)
    df_oil_summary['Min price'] = df_oil_summary.drop(columns=['Max month', 'Max price', 'Min month']).min(axis=1)
    df_oil_summary.index.name='Year'
    df_oil_summary.columns.name='Month'
    print(df_oil_summary)

def promotions_and_location_analysis(train):
    df_promo_corr = train.groupby(['Year', 'Month']).agg({'onpromotion': 'sum', 'sales': 'sum'}).reset_index()
    sns.regplot(x='onpromotion', y='sales', data=df_promo_corr, ci=None)
    plt.title('Correlation between Promotions and Sales (Monthly Aggregation)')
    plt.show()

    df_sales_per_promo = train.groupby(['Year', 'Month']).agg({'onpromotion': 'sum', 'sales': 'sum'}).reset_index()
    df_sales_per_promo = df_sales_per_promo.groupby(['Month']).agg({'onpromotion': 'mean', 'sales': 'mean'}).reset_index()
    df_sales_per_promo['sales_per_promo'] = df_sales_per_promo['sales']/df_sales_per_promo['onpromotion']
    print(df_sales_per_promo.sort_values(by='sales_per_promo', ascending=False).reset_index(drop=True))

    sort_state = train.groupby('state')['sales'].sum().sort_values(ascending=False)
    fig = px.bar(x=sort_state.index, y=sort_state.values,
             labels={'x':'state','y':'sales'},
             title='Total Sales by State',
             text=sort_state.values
            )
    fig.update_traces(textposition='outside')
    fig.update_yaxes(tickformat=".2s")
    fig.show()

def run_full_eda_pipeline(data_path="."):
    try:
        train_raw, test_raw, stores, transactions, holidays_cleaned, oil_cleaned = load_and_clean_auxiliary_data(data_path)
    except Exception as e:
        print(f"Failed to load data. Ensure raw files are present. Error: {e}")
        return

    train_merged = merge_and_impute_data(train_raw.copy(), stores, transactions, oil_cleaned, holidays_cleaned)
    train_fe = extract_required_time_features(train_merged)
    
    descriptive_stats_and_quality_check(train_fe)
        
    sales_aggregation_plots(train_fe)
    sales_trend_analysis(train_fe)
    oil_price_analysis(train_fe)
    promotions_and_location_analysis(train_fe)

if __name__ == "__main__":
    try:
        run_full_eda_pipeline(DATA_PATH)
    except NameError:
        print("Error: Imported functions not found.")
    except Exception as e:
        print(f"Unexpected error: {e}")
