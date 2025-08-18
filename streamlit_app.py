import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import warnings
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Rice Production & Yield Analysis",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2E8B57;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_preprocess_data(file_path):
    """Load and preprocess the rice data"""
    try:
        df = pd.read_excel(file_path)
        df['Year'] = df['Year'].apply(lambda x: pd.to_datetime(x.split('-')[0] + '-01-01'))
        df.set_index('Year', inplace=True)
        df = df.dropna(subset=['Area(Hectare)', 'Production(Tonnes)', 'Yield(Tonne/Hectare)'], how='all')
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def check_stationarity(timeseries):
    """Check stationarity using Augmented Dickey-Fuller test"""
    result = adfuller(timeseries, autolag='AIC')
    return {
        'adf_statistic': result[0],
        'p_value': result[1],
        'critical_values': result[4],
        'is_stationary': result[1] <= 0.05
    }

def difference_series(series):
    """Difference the time series"""
    return series.diff().dropna()

def create_arima_forecast(data, metric_type, forecast_steps=5):
    """Create ARIMA forecast for the given data"""
    try:
        # Check stationarity
        stationarity_result = check_stationarity(data)
        
        if not stationarity_result['is_stationary']:
            # Apply differencing
            data_diff = difference_series(data)
            d = 1
            while not check_stationarity(data_diff)['is_stationary'] and d < 2:
                data_diff = difference_series(data_diff)
                d += 1
        else:
            data_diff = data
            d = 0
        
        # Determine ARIMA parameters
        max_lags = max(1, int(len(data_diff) * 0.3) - 1)
        p = min(1, max_lags)
        q = min(1, max_lags)
        
        # Fit ARIMA model
        model = ARIMA(data, order=(p, d, q))
        results = model.fit()
        
        # Generate forecast
        forecast = results.forecast(steps=forecast_steps)
        
        # Calculate performance metrics if enough data
        performance_metrics = {}
        if len(data) >= 5:
            train_size = int(len(data) * 0.8)
            train, test = data[:train_size], data[train_size:]
            
            model_test = ARIMA(train, order=(p, d, q))
            results_test = model_test.fit()
            forecast_test = results_test.forecast(steps=len(test))
            
            rmse = np.sqrt(mean_squared_error(test, forecast_test))
            mape = mean_absolute_percentage_error(test, forecast_test) * 100
            
            performance_metrics = {
                'rmse': rmse,
                'mape': mape,
                'p': p,
                'd': d,
                'q': q
            }
        
        return {
            'forecast': forecast,
            'model_results': results,
            'performance_metrics': performance_metrics,
            'stationarity_result': stationarity_result,
            'd': d
        }
        
    except Exception as e:
        st.error(f"Error in ARIMA modeling: {str(e)}")
        return None

def create_forecast_plot(historical_data, forecast_data, title, ylabel):
    """Create an interactive forecast plot using Plotly"""
    # Historical data
    historical_trace = go.Scatter(
        x=historical_data.index,
        y=historical_data.values,
        mode='lines+markers',
        name='Historical Data',
        line=dict(color='#2E8B57', width=3),
        marker=dict(size=8)
    )
    
    # Forecast data
    last_date = historical_data.index[-1]
    future_dates = pd.date_range(start=last_date, periods=len(forecast_data) + 1, freq='Y')[1:]
    
    forecast_trace = go.Scatter(
        x=future_dates,
        y=forecast_data,
        mode='lines+markers',
        name='Forecast',
        line=dict(color='#FF6B6B', width=3, dash='dash'),
        marker=dict(size=8, color='#FF6B6B')
    )
    
    # Create figure
    fig = go.Figure()
    fig.add_trace(historical_trace)
    fig.add_trace(forecast_trace)
    
    fig.update_layout(
        title=title,
        xaxis_title='Year',
        yaxis_title=ylabel,
        hovermode='x unified',
        template='plotly_white',
        height=500
    )
    
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">🌾 Rice Production & Yield Analysis</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    st.sidebar.header("📊 Analysis Configuration")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "Upload Rice Data (Excel file)",
        type=['xlsx', 'xls'],
        help="Upload your rice production data file"
    )
    
    if uploaded_file is None:
        st.info("👆 Please upload a rice data file to begin analysis")
        st.stop()
    
    # Load data
    with st.spinner("Loading and preprocessing data..."):
        df = load_and_preprocess_data(uploaded_file)
    
    if df is None:
        st.error("Failed to load data. Please check your file format.")
        st.stop()
    
    # Display data info
    st.sidebar.success(f"✅ Data loaded successfully!")
    st.sidebar.write(f"**Total records:** {len(df)}")
    st.sidebar.write(f"**Date range:** {df.index.min().year} - {df.index.max().year}")
    
    # Get unique states and districts
    states = sorted(df['State'].unique())
    districts = sorted(df['District'].unique())
    
    # Sidebar selections
    selected_state = st.sidebar.selectbox("Select State", states)
    selected_district = st.sidebar.selectbox("Select District", districts)
    
    # Analysis type selection
    analysis_type = st.sidebar.radio(
        "Select Analysis Type",
        ["Production Analysis", "Yield Analysis", "Area Analysis"],
        help="Choose what metric to analyze and forecast"
    )
    
    # Forecast steps
    forecast_steps = st.sidebar.slider(
        "Forecast Steps (Years)",
        min_value=1,
        max_value=10,
        value=5,
        help="Number of years to forecast"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**About:** This app uses ARIMA time series analysis to forecast rice production, yield, and area data.")
    
    # Main content
    st.header(f"📈 {analysis_type} for {selected_district}, {selected_state}")
    
    # Filter data for selected state and district
    filtered_data = df[(df['State'] == selected_state) & (df['District'] == selected_district)].copy()
    
    if filtered_data.empty:
        st.error(f"No data found for {selected_district}, {selected_state}")
        st.stop()
    
    # Display basic statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", len(filtered_data))
    
    with col2:
        st.metric("Date Range", f"{filtered_data.index.min().year} - {filtered_data.index.max().year}")
    
    with col3:
        if analysis_type == "Production Analysis":
            metric_value = filtered_data['Production(Tonnes)'].mean()
            st.metric("Avg Production", f"{metric_value:,.0f} tonnes")
        elif analysis_type == "Yield Analysis":
            metric_value = filtered_data['Yield(Tonne/Hectare)'].mean()
            st.metric("Avg Yield", f"{metric_value:.2f} t/ha")
        else:
            metric_value = filtered_data['Area(Hectare)'].mean()
            st.metric("Avg Area", f"{metric_value:,.0f} ha")
    
    with col4:
        if analysis_type == "Production Analysis":
            metric_value = filtered_data['Production(Tonnes)'].std()
            st.metric("Std Dev", f"{metric_value:,.0f} tonnes")
        elif analysis_type == "Yield Analysis":
            metric_value = filtered_data['Yield(Tonne/Hectare)'].std()
            st.metric("Std Dev", f"{metric_value:.2f} t/ha")
        else:
            metric_value = filtered_data['Area(Hectare)'].std()
            st.metric("Std Dev", f"{metric_value:,.0f} ha")
    
    # Select the appropriate column based on analysis type
    if analysis_type == "Production Analysis":
        data_column = filtered_data['Production(Tonnes)']
        ylabel = "Production (Tonnes)"
        metric_name = "Production"
    elif analysis_type == "Yield Analysis":
        data_column = filtered_data['Yield(Tonne/Hectare)']
        ylabel = "Yield (Tonne/Hectare)"
        metric_name = "Yield"
    else:
        data_column = filtered_data['Area(Hectare)']
        ylabel = "Area (Hectares)"
        metric_name = "Area"
    
    # Data preview
    st.subheader("📋 Data Preview")
    st.dataframe(filtered_data[[col for col in filtered_data.columns if col != 'Year']], use_container_width=True)
    
    # Historical trend
    st.subheader("📊 Historical Trend")
    fig_historical = px.line(
        x=data_column.index.year,
        y=data_column.values,
        title=f"Historical {metric_name} Trend",
        labels={'x': 'Year', 'y': ylabel}
    )
    fig_historical.update_traces(line_color='#2E8B57', line_width=3)
    st.plotly_chart(fig_historical, use_container_width=True)
    
    # ARIMA Analysis
    st.subheader("🔮 ARIMA Forecast Analysis")
    
    if len(data_column) < 3:
        st.warning("Insufficient data for ARIMA analysis. Need at least 3 data points.")
        st.stop()
    
    with st.spinner("Performing ARIMA analysis..."):
        arima_results = create_arima_forecast(data_column, metric_name, forecast_steps)
    
    if arima_results is None:
        st.error("Failed to perform ARIMA analysis. Please check your data.")
        st.stop()
    
    # Display stationarity results
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("**Stationarity Test Results**")
        st.write(f"ADF Statistic: {arima_results['stationarity_result']['adf_statistic']:.4f}")
        st.write(f"P-value: {arima_results['stationarity_result']['p_value']:.4f}")
        st.write(f"Stationary: {'✅ Yes' if arima_results['stationarity_result']['is_stationary'] else '❌ No'}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("**ARIMA Parameters**")
        st.write(f"P (AR): {arima_results['performance_metrics'].get('p', 'N/A')}")
        st.write(f"D (Differencing): {arima_results['d']}")
        st.write(f"Q (MA): {arima_results['performance_metrics'].get('q', 'N/A')}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Performance metrics
    if arima_results['performance_metrics']:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("**Model Performance**")
            st.write(f"RMSE: {arima_results['performance_metrics']['rmse']:.2f}")
            st.write(f"MAPE: {arima_results['performance_metrics']['mape']:.2f}%")
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Forecast plot
    st.subheader("🎯 Forecast Results")
    forecast_fig = create_forecast_plot(
        data_column,
        arima_results['forecast'],
        f"{metric_name} Forecast for {selected_district}, {selected_state}",
        ylabel
    )
    st.plotly_chart(forecast_fig, use_container_width=True)
    
    # Forecast table
    st.subheader("📅 Detailed Forecast")
    last_date = data_column.index[-1]
    future_dates = pd.date_range(start=last_date, periods=len(arima_results['forecast']) + 1, freq='Y')[1:]
    
    forecast_df = pd.DataFrame({
        'Year': [date.year for date in future_dates],
        f'Forecasted {metric_name}': arima_results['forecast']
    })
    
    if analysis_type == "Production Analysis":
        forecast_df[f'Forecasted {metric_name}'] = forecast_df[f'Forecasted {metric_name}'].round(2)
    elif analysis_type == "Yield Analysis":
        forecast_df[f'Forecasted {metric_name}'] = forecast_df[f'Forecasted {metric_name}'].round(3)
    else:
        forecast_df[f'Forecasted {metric_name}'] = forecast_df[f'Forecasted {metric_name}'].round(2)
    
    st.dataframe(forecast_df, use_container_width=True)
    
    # Download forecast results
    csv = forecast_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Forecast Results (CSV)",
        data=csv,
        file_name=f"{selected_district}_{selected_state}_{metric_name}_forecast.csv",
        mime="text/csv"
    )
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            <p>🌾 Rice Production & Yield Analysis Tool | Built with Streamlit & ARIMA</p>
            <p>This tool provides time series analysis and forecasting for agricultural data</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
