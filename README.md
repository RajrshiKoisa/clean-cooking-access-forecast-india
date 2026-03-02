# Forecasting Access to Clean Cooking Fuels in India using Machine Learning

This project predicts national-level access to clean cooking fuels in India using
macroeconomic and development indicators from the World Bank – World Development Indicators.

## Data
- Source: World Bank – World Development Indicators
- Country: India
- Unit of analysis: Year
- Target variable:
  Access to clean fuels and technologies for cooking (% of population)

## Methodology
- Reshaped raw indicator data into a yearly panel dataset
- Removed indicators with high missing values
- Performed linear interpolation for remaining missing data
- Applied a time-aware (chronological) train–test split

## Models
- Linear Regression (baseline)
- Random Forest Regressor

## Evaluation
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² score

## Explainability
- Feature importance extracted from the Random Forest model
- Scenario analysis performed by increasing the most important feature by 10% and
  observing the change in predicted clean cooking access

## Key outcome
The Random Forest model achieves better predictive performance, highlighting
non-linear relationships between development indicators and access to clean cooking fuels.

## How to run
1. Install required libraries:
   pandas, numpy, matplotlib, scikit-learn
2. Update the dataset file path in the script.
3. Run the Python script to reproduce preprocessing, modelling and visualisation.
