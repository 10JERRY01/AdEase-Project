# AdEase Wikipedia Page View Forecasting

## Project Overview

This project analyzes Wikipedia page view data to forecast future views for various articles across different languages. The goal is to help AdEase, an ad infrastructure company, optimize ad placements for its clients by predicting page popularity.

The analysis involves:
- Exploratory Data Analysis (EDA) of page view trends, languages, access types, etc.
- Time series decomposition and stationarity testing.
- Forecasting using ARIMA, SARIMAX (incorporating campaign data), and Facebook Prophet models.
- Evaluation of model performance using Mean Absolute Percentage Error (MAPE).
- A basic pipeline structure for applying forecasting to multiple time series (e.g., different languages).

## Data

The dataset consists of two main files:

1.  `train_1.csv`: Contains daily page view counts for ~145k Wikipedia articles over 550 days. The 'Page' column encodes article title, language, access type, and origin.
2.  `Exog_Campaign_eng.csv`: Contains binary flags indicating significant campaign days or events potentially affecting views for *English* pages.

Data Source (Example Link - Replace if necessary):
[https://drive.google.com/drive/folders/1mdgQscjqnCtdg7LGItomyK0abN6lcHBb](https://drive.google.com/drive/folders/1mdgQscjqnCtdg7LGItomyK0abN6lcHBb)

*(Note: You might need to download the data from the provided source and place the CSV files in the project root directory.)*

## Setup and Usage

1.  **Clone the repository (if applicable) or ensure you have the project files.**
2.  **Install required Python libraries:**
    ```bash
    pip install pandas numpy matplotlib seaborn statsmodels prophet notebook jupyter
    ```
    *(Note: `prophet` installation might require additional steps depending on your OS. Refer to the official Prophet installation guide: [https://facebook.github.io/prophet/docs/installation.html](https://facebook.github.io/prophet/docs/installation.html))*
3.  **Ensure data files (`train_1.csv`, `Exog_Campaign_eng.csv`) are in the project directory.**
4.  **Launch Jupyter Notebook:**
    ```bash
    jupyter notebook
    ```
5.  **Open and run the `AdEase_Analysis.ipynb` notebook.** The notebook contains all the analysis steps, visualizations, model training, and evaluation.

## Files

-   `AdEase_Analysis.ipynb`: Jupyter Notebook containing the main analysis code.
-   `train_1.csv`: Training data with page views.
-   `Exog_Campaign_eng.csv`: Exogenous campaign data for English pages.
-   `readme.md`: This file.
-   `documentation.md`: Detailed project documentation.
