import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import dcc, html
import warnings

from preprocessing import load_and_clean_auxiliary_data, merge_and_impute_data 

warnings.simplefilter(action='ignore', category=FutureWarning)

# ==============================================================================
# I. VISUALIZATION FUNCTIONS 
# ==============================================================================

def create_sales_timeline_fig(df):
    daily_global = df.groupby('date', as_index=False)['sales'].sum()
    top_stores = df.groupby('store_nbr')['sales'].sum().nlargest(5).index
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily_global['date'], y=daily_global['sales'], mode='lines', name="Global"))
    for store in top_stores:
        df_store = df[df['store_nbr'] == store].groupby('date', as_index=False)['sales'].sum()
        fig.add_trace(go.Scatter(x=df_store['date'], y=df_store['sales'], mode='lines', name=f"Store {store}"))
    fig.update_layout(title="Daily Sales (Global + Top 5 Stores)", xaxis_title="Date", yaxis_title="Sales", height=450)
    return fig

def create_family_sales_fig(df):
    top_fam = df.groupby('family')['sales'].sum().nlargest(12).index
    fig = go.Figure()
    for fam in top_fam:
        df_f = df[df['family'] == fam].groupby('date', as_index=False)['sales'].sum()
        fig.add_trace(go.Scatter(x=df_f['date'], y=df_f['sales'], mode='lines', name=fam))
    fig.update_layout(title="Daily Sales (Top 12 Families)", xaxis_title="Date", yaxis_title="Sales", height=450)
    return fig

def create_monthly_box_fig(df):
    monthly = df.copy()
    monthly['YearMonth'] = monthly['date'].dt.to_period('M').dt.to_timestamp()
    month_agg = monthly.groupby(['YearMonth'], as_index=False)['sales'].sum()
    month_agg['month'] = month_agg['YearMonth'].dt.month
    fig = px.box(month_agg, x='month', y='sales', points='outliers', title='Distribution of Monthly Sales by Month', height=450)
    fig.update_xaxes(tickmode='array', tickvals=list(range(1,13)))
    return fig

def create_weekday_analysis_fig(df):
    weekday = df.copy()
    weekday['Week_day'] = pd.Categorical(weekday['Week_day'],
        categories=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'], ordered=True)
    weekday['year'] = weekday['date'].dt.year
    weekday_avg = weekday.groupby(['year','Week_day'], as_index=False)['sales'].mean()
    weekday_all = weekday.groupby(['Week_day'], as_index=False)['sales'].mean()
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=weekday_all['Week_day'], y=weekday_all['sales'], name="All Years Average"))
    for y in sorted(weekday_avg['year'].unique()):
        df_year = weekday_avg[weekday_avg['year']==y]
        fig.add_trace(go.Scatter(x=df_year['Week_day'], y=df_year['sales'], mode='lines+markers', name=str(y)))
        
    fig.update_layout(title="Average Sales by Weekday", xaxis_title="Weekday", yaxis_title="Average Sales", height=450)
    return fig

def create_oil_correlation_figs(df):
    oil_daily = df[['date','dcoilwtico','sales']].dropna().groupby('date', as_index=False).agg({'dcoilwtico':'mean','sales':'sum'})
    
    oil_scatter_fig = px.scatter(oil_daily, x='dcoilwtico', y='sales', trendline='ols', title='Oil Price vs Sales (Scatter)', height=450)
    
    oil_timeseries_fig = go.Figure()
    oil_timeseries_fig.add_trace(go.Scatter(x=oil_daily['date'], y=oil_daily['sales'], name='Sales'))
    oil_timeseries_fig.add_trace(go.Scatter(x=oil_daily['date'], y=oil_daily['dcoilwtico'], name='Oil Price', yaxis="y2"))
    oil_timeseries_fig.update_layout(title='Sales & Oil Price Over Time',
                      yaxis=dict(title='Sales'),
                      yaxis2=dict(title='Oil Price', overlaying='y', side='right'),
                      height=450)
    return oil_scatter_fig, oil_timeseries_fig

def create_geographic_sales_figs(df):
    city_sum = df.groupby('city', as_index=False)['sales'].sum().sort_values('sales', ascending=False)
    city_treemap_fig = px.treemap(city_sum, path=[px.Constant("All Cities"), 'city'], values='sales', title='Sales Treemap by City', height=450)
    
    state_sum = df.groupby('state', as_index=False)['sales'].sum().sort_values('sales', ascending=False)
    state_bar_fig = px.bar(state_sum.head(15), x='state', y='sales', title='Top 15 States by Sales', height=450)
    return city_treemap_fig, state_bar_fig

def create_store_type_fig(df):
    ct = df.groupby(['date','store_type'], as_index=False)['sales'].sum()
    fig = go.Figure()
    for st in ct['store_type'].unique():
        df_st = ct[ct['store_type'] == st]
        fig.add_trace(go.Scatter(x=df_st['date'], y=df_st['sales'], mode='lines', name=f"Store Type {st}"))
    fig.update_layout(title="Sales Over Time by Store Type", xaxis_title="Date", yaxis_title="Sales", height=450)
    return fig

def create_monthly_trends_fig(df):
    cal = df.groupby('date', as_index=False)['sales'].sum()
    cal['year'] = cal['date'].dt.year
    cal['month'] = cal['date'].dt.month
    monthly = cal.groupby(['year','month'], as_index=False)['sales'].sum()
    monthly['month_name'] = pd.to_datetime(monthly['month'], format='%m').dt.strftime('%b')
    month_order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    monthly_trends_fig = px.line(monthly, x='month_name', y='sales', color='year', markers=True,
                                 category_orders={"month_name": month_order}, title="Monthly Sales Trends by Year", height=450)
    return monthly_trends_fig

def create_time_series_metrics_fig(df):
    time_df = df[['date','sales','onpromotion','dcoilwtico']].groupby('date', as_index=False).mean()
    fig = go.Figure()
    for col in ['sales','onpromotion','dcoilwtico']:
        fig.add_trace(go.Scatter(x=time_df['date'], y=time_df[col], mode='lines', name=col))
    fig.update_layout(title="Time Series Metrics (Sales, Onpromotion, Oil Price)", xaxis_title="Date", yaxis_title="Value", height=450)
    return fig

def create_kpi_fig(df):
    total_sales = df['sales'].sum()
    median_sales = df['sales'].median()
    mean_sales = df['sales'].mean()
    max_sale = df['sales'].max()
    
    kpi_fig = go.Figure()
    kpi_fig.add_trace(go.Indicator(mode="number", value=total_sales, title={"text": "Total Sales"}, domain={'row': 0, 'column': 0}, number={'valueformat': ',.0f'}))
    kpi_fig.add_trace(go.Indicator(mode="number", value=median_sales, title={"text": "Median Sale"}, domain={'row': 0, 'column': 1}))
    kpi_fig.add_trace(go.Indicator(mode="number", value=mean_sales, title={"text": "Mean Sale"}, domain={'row': 0, 'column': 2}, number={'valueformat': '.2f'}))
    kpi_fig.add_trace(go.Indicator(mode="number", value=max_sale, title={"text": "Max Sale"}, domain={'row': 0, 'column': 3}, number={'valueformat': ',.0f'}))
    kpi_fig.update_layout(grid={'rows': 1, 'columns': 4}, title="Quick KPIs", height=150)
    return kpi_fig

# ==============================================================================
# II. MAIN EXECUTION AND DASH APP
# ==============================================================================

def generate_dashboard():
    
    train_raw, stores, transactions, holidays_cleaned, oil_cleaned = load_and_clean_auxiliary_data()
    
    df = merge_and_impute_data(train_raw, stores, transactions, oil_cleaned, holidays_cleaned)
    
    kpi_fig = create_kpi_fig(df)
    sales_timeline_fig = create_sales_timeline_fig(df)
    family_sales_fig = create_family_sales_fig(df)
    monthly_box_fig = create_monthly_box_fig(df)
    weekday_analysis_fig = create_weekday_analysis_fig(df)
    oil_scatter_fig, oil_timeseries_fig = create_oil_correlation_figs(df)
    city_treemap_fig, state_bar_fig = create_geographic_sales_figs(df)
    store_type_fig = create_store_type_fig(df)
    monthly_trends_fig = create_monthly_trends_fig(df)
    time_series_metrics_fig = create_time_series_metrics_fig(df)
    
    app = dash.Dash(__name__)
    
    app.layout = html.Div([
        html.H1("Sales Analytics Dashboard", style={'text-align': 'center', 'margin-bottom': '30px', 'color': '#2a3f5f'}),
        
        html.Div([
            html.H2("Key Performance Indicators", style={'color': '#4c78a8'}), 
            dcc.Graph(id='kpi-chart', figure=kpi_fig)
        ], style={'margin-bottom': '30px', 'border': '1px solid #ddd', 'padding': '15px'}),
        
        html.Div([
            html.H2("Sales Timeline Analysis", style={'color': '#4c78a8'}), 
            dcc.Graph(id='sales-timeline', figure=sales_timeline_fig)
        ], style={'margin-bottom': '30px', 'border': '1px solid #ddd', 'padding': '15px'}),
        
        html.Div([
            html.H2("Product Family Sales", style={'color': '#4c78a8'}), 
            dcc.Graph(id='family-sales', figure=family_sales_fig)
        ], style={'margin-bottom': '30px', 'border': '1px solid #ddd', 'padding': '15px'}),
        
        html.Div([
            html.H2("Sales Distribution Analysis", style={'color': '#4c78a8'}),
            html.Div([
                dcc.Graph(id='monthly-distribution', figure=monthly_box_fig, style={'width': '50%', 'display': 'inline-block', 'padding-right': '10px'}),
                dcc.Graph(id='weekday-analysis', figure=weekday_analysis_fig, style={'width': '50%', 'display': 'inline-block'})
            ], style={'display': 'flex'})
        ], style={'margin-bottom': '30px', 'border': '1px solid #ddd', 'padding': '15px'}),

        html.Div([
            html.H2("Oil Price vs Sales Correlation", style={'color': '#4c78a8'}),
            html.Div([
                dcc.Graph(id='oil-scatter', figure=oil_scatter_fig, style={'width': '50%', 'display': 'inline-block', 'padding-right': '10px'}),
                dcc.Graph(id='oil-timeseries', figure=oil_timeseries_fig, style={'width': '50%', 'display': 'inline-block'})
            ], style={'display': 'flex'})
        ], style={'margin-bottom': '30px', 'border': '1px solid #ddd', 'padding': '15px'}),

        html.Div([
            html.H2("Geographic Sales Analysis", style={'color': '#4c78a8'}),
            html.Div([
                dcc.Graph(id='city-treemap', figure=city_treemap_fig, style={'width': '50%', 'display': 'inline-block', 'padding-right': '10px'}),
                dcc.Graph(id='state-bar', figure=state_bar_fig, style={'width': '50%', 'display': 'inline-block'})
            ], style={'display': 'flex'})
        ], style={'margin-bottom': '30px', 'border': '1px solid #ddd', 'padding': '15px'}),
        
        html.Div([
            html.H2("Store Performance and Trends", style={'color': '#4c78a8'}),
            html.Div([
                dcc.Graph(id='store-type-timeline', figure=store_type_fig, style={'width': '50%', 'display': 'inline-block', 'padding-right': '10px'}),
                dcc.Graph(id='monthly-trends', figure=monthly_trends_fig, style={'width': '50%', 'display': 'inline-block'})
            ], style={'display': 'flex'})
        ], style={'margin-bottom': '30px', 'border': '1px solid #ddd', 'padding': '15px'}),

        html.Div([
            html.H2("General Time Series Metrics", style={'color': '#4c78a8'}), 
            dcc.Graph(id='time-series-metrics', figure=time_series_metrics_fig)
        ], style={'margin-bottom': '30px', 'border': '1px solid #ddd', 'padding': '15px'}),

    ], style={'padding': '20px', 'font-family': 'Arial, sans-serif', 'max-width': '1400px', 'margin': 'auto', 'background-color': '#f8f9fa'})

    if __name__ == "__main__":
        app.run_server(debug=True)

if __name__ == "__main__":
    generate_dashboard()