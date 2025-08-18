# 🌾 Rice Production & Yield Analysis - Streamlit App

A comprehensive Streamlit web application for analyzing and forecasting rice production, yield, and area data using ARIMA time series analysis.

## 🚀 Features

- **Interactive Data Analysis**: Upload Excel files and analyze rice production data
- **Multiple Analysis Types**: Production, Yield, and Area analysis
- **ARIMA Forecasting**: Advanced time series forecasting with customizable parameters
- **Stationarity Testing**: Augmented Dickey-Fuller test for time series validation
- **Interactive Visualizations**: Beautiful charts using Plotly
- **Performance Metrics**: RMSE and MAPE calculations
- **Data Export**: Download forecast results as CSV
- **Responsive Design**: Modern, user-friendly interface

## 📋 Prerequisites

- Python 3.8 or higher
- pip package manager

## 🛠️ Installation

1. **Clone or download the project files**
   ```bash
   # If using git
   git clone <repository-url>
   cd ISI_Internship-main
   ```

2. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Running the App Locally

1. **Navigate to the project directory**
   ```bash
   cd ISI_Internship-main
   ```

2. **Run the Streamlit app**
   ```bash
   streamlit run streamlit_app.py
   ```

3. **Open your browser**
   - The app will automatically open in your default browser
   - Usually at: `http://localhost:8501`

## 📊 How to Use

1. **Upload Data**: Use the sidebar to upload your rice production Excel file
2. **Select Parameters**: Choose state, district, and analysis type
3. **Configure Forecast**: Set the number of years to forecast
4. **View Results**: Explore historical trends, ARIMA analysis, and forecasts
5. **Download Results**: Export forecast data as CSV

## 📁 Data Format Requirements

Your Excel file should contain the following columns:
- `Year`: Year of the data (format: YYYY or YYYY-YY)
- `State`: State name
- `District`: District name
- `Area(Hectare)`: Area under rice cultivation
- `Production(Tonnes)`: Rice production in tonnes
- `Yield(Tonne/Hectare)`: Yield per hectare

## 🌐 Deployment Options

### Option 1: Streamlit Cloud (Recommended for beginners)
1. Push your code to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Deploy automatically

### Option 2: Heroku
1. Create a `Procfile`:
   ```
   web: streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0
   ```
2. Deploy using Heroku CLI or GitHub integration

### Option 3: Docker
1. Create a `Dockerfile`:
   ```dockerfile
   FROM python:3.9-slim
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   COPY . .
   EXPOSE 8501
   CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
   ```
2. Build and run:
   ```bash
   docker build -t rice-analysis-app .
   docker run -p 8501:8501 rice-analysis-app
   ```

### Option 4: Local Server
1. Run the app locally
2. Configure your firewall to allow external connections
3. Access via your server's IP address

## 🔧 Configuration

### Environment Variables
- `STREAMLIT_SERVER_PORT`: Custom port (default: 8501)
- `STREAMLIT_SERVER_ADDRESS`: Server address (default: localhost)

### Customization
- Modify the CSS in the app for custom styling
- Adjust ARIMA parameters in the `create_arima_forecast` function
- Add new analysis types by extending the main function

## 📈 Technical Details

- **ARIMA Model**: Auto-regressive Integrated Moving Average
- **Stationarity Testing**: Augmented Dickey-Fuller test
- **Performance Metrics**: RMSE (Root Mean Square Error) and MAPE (Mean Absolute Percentage Error)
- **Visualization**: Plotly for interactive charts
- **Data Processing**: Pandas for data manipulation

## 🐛 Troubleshooting

### Common Issues:
1. **Port already in use**: Change port with `streamlit run streamlit_app.py --server.port 8502`
2. **Package conflicts**: Use virtual environment: `python -m venv venv && source venv/bin/activate`
3. **Memory issues**: Reduce forecast steps or use smaller datasets

### Error Messages:
- **"No data found"**: Check your Excel file format and column names
- **"Insufficient data"**: Ensure you have at least 3 data points
- **"Failed to load data"**: Verify Excel file format and data structure

## 🤝 Contributing

Feel free to contribute by:
- Reporting bugs
- Suggesting new features
- Improving documentation
- Optimizing code performance

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Built with Streamlit
- ARIMA implementation using statsmodels
- Data visualization with Plotly
- Agricultural data analysis for rice production

---

**Happy Forecasting! 🌾📊**
