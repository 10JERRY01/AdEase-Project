# Project Documentation: AdEase Wikipedia Page View Forecasting

## 1. Introduction

### 1.1. Project Goal

The primary goal of this project is to analyze and forecast daily page views for a diverse set of Wikipedia articles across multiple languages. This forecasting capability aims to support AdEase in optimizing digital advertising strategies for its clients by identifying pages with high anticipated traffic, allowing for more effective and economical ad placements.

### 1.2. Business Context

AdEase provides an ad infrastructure platform designed to simplify digital advertising for businesses. By leveraging AI, AdEase aims to maximize ad clicks at minimum cost. Understanding and predicting web traffic patterns on potential ad placement locations (like Wikipedia pages) is crucial for optimizing ad spend, targeting relevant audiences, and measuring campaign effectiveness across different regions and languages.

### 1.3. Data Overview

-   **`train_1.csv`**: Time series data where each row represents a unique Wikipedia page, and columns represent dates. Cell values indicate the number of views on that specific date. The `Page` column contains metadata string including title, language, access type (e.g., desktop, mobile), and access origin (e.g., spider, agent).
-   **`Exog_Campaign_eng.csv`**: Exogenous variable data indicating dates with significant campaigns or events that might influence traffic, specifically for English language pages.

## 2. Methodology

The analysis follows a standard time series forecasting workflow implemented within the `AdEase_Analysis.ipynb` Jupyter Notebook.

### 2.1. Data Loading and Preprocessing

-   **Loading:** Datasets (`train_1.csv`, `Exog_Campaign_eng.csv`) are loaded using pandas.
-   **Null Value Handling:** Null values in the view counts (likely indicating zero views or missing data) are identified and filled with 0.
-   **Metadata Extraction:** The `Page` column string is parsed using regular expressions to extract:
    -   Article Title
    -   Language Code (e.g., 'en', 'ja', 'de')
    -   Access Type (e.g., 'desktop', 'mobile-app', 'mobile-web')
    -   Access Origin (e.g., 'spider', 'agent')
-   **Data Reshaping:** The training data is transformed from a wide format (dates as columns) to a long format (one row per page per date) using `pd.melt` for easier aggregation and visualization. Date columns are converted to datetime objects.

### 2.2. Exploratory Data Analysis (EDA)

-   **Overall Trend:** The total daily views across all pages are calculated and plotted to observe the overall traffic pattern and identify potential long-term trends or major shifts.
-   **Categorical Analysis:** Average daily views are calculated and plotted grouped by:
    -   Language (focusing on the top N most frequent languages)
    -   Access Type
    -   Access Origin
    This helps understand how traffic differs across these dimensions.
-   **Distribution Analysis:** Value counts for Language, Access Type, and Access Origin are examined.

### 2.3. Stationarity Analysis

-   **Target Series:** The analysis primarily focuses on the `total_daily_views` series initially.
-   **Decomposition:** Seasonal decomposition (`statsmodels.tsa.seasonal.seasonal_decompose`) is used (assuming weekly seasonality, period=7) to visualize the trend, seasonal, and residual components. Both additive and potentially multiplicative models are considered.
-   **Augmented Dickey-Fuller (ADF) Test:** The ADF test (`statsmodels.tsa.stattools.adfuller`) is performed on the original series to statistically check for stationarity (specifically, the presence of a unit root).
-   **Differencing:** If the original series is found to be non-stationary (as expected), first-order differencing (`.diff().dropna()`) is applied.
-   **Post-Differencing Check:** The ADF test is repeated on the differenced series to confirm if stationarity has been achieved.
-   **ACF/PACF Plots:** Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) plots (`statsmodels.graphics.tsaplots.plot_acf`, `plot_pacf`) are generated for both the original and differenced series. These plots help identify the potential order (p, q) for ARIMA models and reveal autocorrelation patterns.

### 2.4. Time Series Modeling

A train-test split is performed (forecasting the last `forecast_days`, e.g., 30 days). Models are evaluated using Mean Absolute Percentage Error (MAPE). Walk-forward validation (refitting the model at each step in the test set) is used for ARIMA/SARIMAX evaluation, which is robust but computationally intensive.

-   **ARIMA (AutoRegressive Integrated Moving Average):**
    -   An `ARIMA` model (`statsmodels.tsa.arima.model.ARIMA`) is trained on the `total_daily_views` series.
    -   The order (p, d, q) is initially set based on ACF/PACF analysis (e.g., (7, 1, 7) assuming weekly patterns and first differencing).
-   **SARIMAX (Seasonal ARIMA with eXogenous variables):**
    -   A `SARIMAX` model (`statsmodels.tsa.statespace.sarimax.SARIMAX`) is trained, incorporating the campaign data from `Exog_Campaign_eng.csv` as an exogenous variable.
    -   It uses both a non-seasonal order (p, d, q) and a seasonal order (P, D, Q, s), e.g., (1, 1, 1, 7) for weekly seasonality.
    -   *Limitation Noted:* The provided exogenous data is only for English pages, while the initial model is run on total views. Ideally, this model should be applied specifically to English page data.
-   **Prophet:**
    -   Facebook's `Prophet` model (`prophet.Prophet`) is trained. Data is formatted into 'ds' (datetime) and 'y' (value) columns.
    -   Prophet automatically handles trend and seasonality (weekly, yearly).
    -   The possibility of adding campaign dates as custom holidays is included as a commented-out option.

### 2.5. Parameter Tuning (Example)

-   A grid search structure (`grid_search_arima`) is provided as an example for finding optimal ARIMA (p, d, q) parameters.
-   It iterates through combinations of p, d, q values, evaluates each using walk-forward validation (`evaluate_arima_model`), and identifies the order with the lowest MAPE.
-   *Note:* This grid search is commented out by default due to its high computational cost. Using `pmdarima.auto_arima` is recommended as a more efficient alternative.

### 2.6. Multi-Series Pipeline (Example)

-   A basic pipeline function (`forecast_pipeline`) is defined to demonstrate applying a forecasting model (ARIMA in this case) to multiple series.
-   The pipeline is applied to the aggregated daily views for selected languages (e.g., 'en', 'ja').
-   *Note:* This example uses a fixed ARIMA order. A production pipeline should incorporate dynamic order selection (e.g., `auto_arima`) or use models better suited for large numbers of series (like Prophet or global ML models).

## 3. Key Findings / Expected Results

*(Based on the code structure and typical web traffic patterns - actual results require running the notebook)*

-   **Data Quality:** The dataset contains null values, primarily addressed by filling with zero. The `Page` column requires parsing to extract useful features.
-   **Traffic Patterns:** Visualizations are expected to show significant weekly seasonality (lower traffic on weekends) and potentially yearly patterns. English pages likely dominate traffic volume. Mobile vs. desktop access patterns might differ.
-   **Stationarity:** Raw page view data is typically non-stationary. First-order differencing is often sufficient to achieve stationarity for the mean.
-   **Model Performance:**
    -   All three models (ARIMA, SARIMAX, Prophet) are expected to provide reasonable forecasts.
    -   SARIMAX, incorporating seasonality and exogenous campaign data (when applied to relevant English series), might outperform basic ARIMA, especially during campaign periods.
    -   Prophet is often robust and easier to tune, potentially yielding good results with less manual effort for order selection.
    -   MAPE values in the target range (e.g., 4-8% as mentioned in the prompt, though this depends heavily on the specific series' predictability) would indicate good model performance.
-   **Parameter Tuning:** Grid search or auto_arima would likely identify more optimal (p, d, q) / (P, D, Q, s) orders than the initial guesses, potentially improving ARIMA/SARIMAX accuracy.

## 4. Questionnaire Answers

Detailed answers based on the analysis steps are provided in the final markdown cell of the `AdEase_Analysis.ipynb` notebook.

## 5. Limitations and Future Work

-   **Computational Cost:** Walk-forward validation and grid search for ARIMA/SARIMAX are very slow, especially for many series.
-   **Exogenous Data Scope:** Campaign data is only for English pages; its direct application to total views or non-English pages is inaccurate.
-   **Model Scope:** Models were primarily demonstrated on aggregated total views or aggregated language views. Applying them to all 145k individual page series requires significant scaling and likely different modeling approaches (e.g., global models, clustering).
-   **Parameter Optimization:** The grid search example is limited. More sophisticated methods (`auto_arima`, Bayesian optimization) or model-specific tuning (Prophet hyperparameters) should be explored.
-   **Feature Engineering:** Additional features could be engineered (e.g., day of the week, month, holidays, rolling statistics, lag features) especially for ML-based models.
-   **Alternative Models:** Explore global forecasting models (e.g., DeepAR) or machine learning approaches (LightGBM, XGBoost) that might scale better to a large number of series.
-   **Error Analysis:** Deeper analysis of forecast errors (residuals) is needed to identify model weaknesses.
