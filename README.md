# Air Pollution and Respiratory Health Impact Prediction

This project uses machine learning to predict respiratory illness cases based on air pollution and weather data. It demonstrates a complete pipeline: synthetic data generation, exploratory analysis, model training (Linear Regression, Random Forest, XGBoost), evaluation, and feature importance analysis.

## Dataset

The script generates synthetic data that mimics real-world relationships:

- **Air pollutants**: PM2.5, NO2, Ozone  
- **Weather**: Temperature, Humidity  
- **Target**: Respiratory illness cases (simulated as a linear combination of pollutants with added noise)

You can replace the synthetic data with real datasets (e.g., from public health or environmental agencies) by providing a CSV with the same columns.

## Requirements

- Python 3.8+
- NumPy
- pandas
- Matplotlib
- seaborn
- scikit-learn
- XGBoost
- joblib (optional, for model saving)

All dependencies are listed in `requirements.txt`.


