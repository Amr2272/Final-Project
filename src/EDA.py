import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

from preprocessing import load_and_clean_auxiliary_data, merge_auxiliary_data


def extract_required_time_features(df):
    """Extract time features without overwriting day_type from holidays"""
    df['date'] = pd.to_datetime(df['date'])
    df['Year'] = df['date'].dt.year
    df['Month'] = df['date'].dt.month
    df['DayOfWeek'] = df['date'].dt.dayofweek
    return df


def descriptive_stats_and_quality_check(df):
    print("\n=== Data Info ===")
    print(df.shape)

    missing_values = df.isnull().sum()
    missing_percent = (missing_values / len(df)) * 100
    missing_df = pd.DataFrame({
        "Missing Count": missing_values,
        "Missing %": missing_percent.round(2)
    })
    print("\n=== Missing Values ===")
    print(missing_df[missing_df["Missing Count"] > 0])

    print("\n=== Numerical Summary ===")
    print(df[['sales', 'onpromotion', 'dcoilwtico']].describe().T)


def sales_aggregation_plots(train):
    """Plot sales by holiday day types"""
    print("\nGenerating Sales Aggregation Plots...")

    all_types = ['Holiday', 'Transfer', 'Additional', 'Bridge', 'Work Day', 'Event']

    try:
        # Check if day_type exists
        if 'day_type' not in train.columns:
            print("Warning: day_type column not found. Skipping sales aggregation plots.")
            return

        # Aggregate total and mean sales per day_type
        df_agg = train.groupby('day_type')['sales'].agg(['sum', 'mean']).reindex(all_types, fill_value=0).reset_index()
        df_agg.rename(columns={'sum': 'total_sales', 'mean': 'avg_sales'}, inplace=True)

        # Create side-by-side plots
        sns.set_style("whitegrid")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        sns.barplot(x='day_type', y='total_sales', data=df_agg, ax=axes[0], palette='Blues_d')
        axes[0].set_title("Total Sales per Day Type")
        axes[0].set_xlabel("")
        axes[0].set_ylabel("Total Sales")
        axes[0].tick_params(axis='x', rotation=45)
        for p in axes[0].patches:
            h = p.get_height()
            axes[0].annotate(f'{int(h):,}', (p.get_x() + p.get_width() / 2., h),
                             ha='center', va='bottom', fontsize=9, xytext=(0, 6), textcoords='offset points')

        sns.barplot(x='day_type', y='avg_sales', data=df_agg, ax=axes[1], palette='Greens_d')
        axes[1].set_title("Average Sales per Day Type")
        axes[1].set_xlabel("")
        axes[1].set_ylabel("Average Sales")
        axes[1].tick_params(axis='x', rotation=45)
        for p in axes[1].patches:
            h = p.get_height()
            axes[1].annotate(f'{h:,.2f}', (p.get_x() + p.get_width() / 2., h),
                             ha='center', va='bottom', fontsize=9, xytext=(0, 6), textcoords='offset points')

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error in sales_aggregation_plots: {e}")


def sales_trend_analysis(train):
    print("\nGenerating Sales Trend Plots...")

    df_sales_timeline = train.groupby(['Year', 'Month'])['sales'].sum().reset_index()
    df_sales_timeline['Year-Month'] = (
        df_sales_timeline['Year'].astype(str) + '-' +
        df_sales_timeline['Month'].astype(str).str.zfill(2)
    )

    plt.figure(figsize=(12, 4))
    sns.lineplot(
        x='Year-Month',
        y='sales',
        hue='Year',
        data=df_sales_timeline,
        marker="o"
    )
    plt.xticks(rotation=45)
    plt.title("Sales Timeline by Year")
    plt.tight_layout()
    plt.show()


def oil_price_analysis(train):
    print("\nGenerating Oil Price Analysis...")

    df_oil = train.groupby(['Year', 'Month'])['dcoilwtico'].mean().reset_index()
    df_oil['Year-Month'] = (
        df_oil['Year'].astype(str) + "-" +
        df_oil['Month'].astype(str).str.zfill(2)
    )

    plt.figure(figsize=(12, 4))
    sns.lineplot(
        x='Year-Month',
        y='dcoilwtico',
        hue='Year',
        data=df_oil,
        marker="o"
    )
    plt.xticks(rotation=45)
    plt.title("Oil Price Trend")
    plt.tight_layout()
    plt.show()


def promotions_and_location_analysis(train):
    print("\nGenerating Promotions vs Sales Plot...")

    df_promo_corr = train.groupby(['Year', 'Month']).agg(
        {'onpromotion': 'sum', 'sales': 'sum'}
    ).reset_index()

    plt.figure(figsize=(10, 6))
    sns.regplot(x='onpromotion', y='sales', data=df_promo_corr, ci=None)
    plt.title('Correlation between Promotions and Sales (Monthly)')
    plt.tight_layout()
    plt.show()


def run_full_eda_pipeline():
    print("Loading data...")

    try:
        train_raw, test_raw, stores, holidays_cleaned, oil_cleaned = load_and_clean_auxiliary_data()
    except Exception as e:
        print(f"Failed to load data: {e}")
        return

    print("Merging data...")
    train_merged = merge_auxiliary_data(train_raw.copy(), stores, holidays_cleaned, oil_cleaned)

    print("Extracting features...")
    train_fe = extract_required_time_features(train_merged)

    descriptive_stats_and_quality_check(train_fe)
    sales_aggregation_plots(train_fe)
    sales_trend_analysis(train_fe)
    oil_price_analysis(train_fe)
    promotions_and_location_analysis(train_fe)


if __name__ == "__main__":
    run_full_eda_pipeline()
