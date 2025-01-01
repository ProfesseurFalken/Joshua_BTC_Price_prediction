# Bitcoin Price Predictor by Emmanuel Jean Louis Wojcik wojcikej@orange.fr

A sophisticated ensemble machine learning system that predicts Bitcoin price movements using multiple models including Prophet, Gradient Boosting, and Neural Networks.

## Features

- Real-time Bitcoin price data fetching using yfinance
- Technical indicator calculations (RSI, MACD, Bollinger Bands)
- Ensemble prediction combining:
  - Facebook Prophet for time series forecasting
  - Gradient Boosting Regressor
  - Multi-layer Perceptron Neural Network
- Trading recommendations for both technical and non-technical users
- 24-hour price forecasting
- Visualization of predictions and historical data

## Requirements

```
numpy
pandas
yfinance
scikit-learn
prophet
matplotlib
```

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the main script:
```bash
python bitcoin_predictor.py
```

The system will:
1. Fetch recent Bitcoin price data
2. Calculate technical indicators
3. Train multiple prediction models
4. Generate ensemble predictions
5. Display performance metrics and visualizations
6. Provide trading recommendations
7. Show 24-hour price forecast

## Technical Details

### Data Processing
- Fetches 90 days of hourly Bitcoin price data
- Handles missing values and validates data consistency
- Calculates technical indicators including SMA, EMA, RSI, MACD, and Bollinger Bands

### Models
- **Prophet**: Handles seasonality and trends in time series data
- **Gradient Boosting**: Captures non-linear relationships in feature space
- **Neural Network**: MLP with 2 hidden layers for complex pattern recognition
- **Ensemble**: Combines predictions from all models for improved accuracy

### Performance Metrics
- RMSE (Root Mean Square Error) for each model
- Comparative visualization of predictions vs actual prices
- Model performance evaluation on test data

## Output

The system provides:
1. Technical analysis based on RSI and price forecasts
2. User-friendly recommendations for non-experts
3. Detailed 24-hour price predictions
4. Performance visualizations
5. Logging of all key metrics and predictions

## License

MIT License

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss proposed modifications.
