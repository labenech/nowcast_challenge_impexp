
# 🇪🇺 Nowcasting EU Import/Export Indicators

This repository contains a Python-based pipeline designed to nowcast monthly import and export statistics for EU countries. The project leverages a combination of econometric models and machine learning techniques, including XGBoost, AutoARIMA, and ETS, to forecast trade indicators using Eurostat data and COVID-19 stringency indices.

## 📦 Project Overview

The pipeline performs the following key steps:

- **Data Acquisition**: Downloads trade data from Eurostat and COVID-19 stringency indices.
- **Data Transformation**: Processes and reshapes the data into a suitable format for modeling.
- **Feature Engineering**: Creates lag features, rolling means, and encodes cyclical time features.
- **Standardization**: Applies standardization to the target variable by country.
- **Model Training**: Trains multiple models (e.g., XGBoost, AutoARIMA) for each country.
- **Evaluation**: Assesses model performance using metrics like RMSE and MAPE.
- **Prediction**: Generates forecasts for future periods.
- **Result Storage**: Saves predictions in a structured JSON format.

## 🧰 Requirements

Ensure you have the following installed:

- Python 3.8 or higher
- pip

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## 📁 Repository Structure

```
nowcast_challenge_impexp/
├── data/
│   └── results/               # Directory for storing output JSON files
├── extra/
│   ├── constants.py           # Contains constant variables used across scripts
│   └── functions.py           # Utility functions for data processing and modeling
├── main.py                    # Main script to execute the pipeline
├── requirements.txt           # List of required Python packages
└── README.md                  # Project documentation
```

## 🚀 Getting Started

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/labenech/nowcast_challenge_impexp.git
   cd nowcast_challenge_impexp
   ```

2. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Main Script**:

   Execute the main script to start the data processing and modeling pipeline:

   ```bash
   python main.py
   ```

   This will perform data acquisition, preprocessing, model training, and generate forecasts.

## ⚙️ Configuration

The script uses the following key variables:

- `date_2_pred`: The target date for prediction, set to the first day of the current month.
- `date_train`: 20 months before date_2_pred, used to specify length of train data.
- `max_steps_default`: Define how many epochs for NHITS and TFT models, 2500  by  default.
- `nb_lags`: how many lags to generate, 13 by  default.


These can be modified in the `constants.py` script as needed.

## 📈 Output

The forecasts are saved in the `data/results/` directory as JSON files, named using the pattern:

```
<date_2_pred>_<indicator>.json
```

Each JSON file contains entries like:

```json
{
  "entry_1": {
    "AT": 1234.5,
    "BE": 2345.6
  },
  "entry_2": {
    "AT": 1250.7,
    "BE": 2360.8
  }
}
```

These entries represent forecasts from different models or model combinations for each country.

## 🧪 Model Evaluation

The pipeline evaluates model performance using metrics such as:

- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error

These metrics help in selecting the best-performing model for each country.

## 🧩 Extending the Project

To add new models or indicators:

1. **Update `extra/functions.py`**: Implement the new model's training and prediction functions.
2. **Modify `main.py`**: Integrate the new model into the pipeline, ensuring it follows the existing structure.
3. **Adjust `extra/constants.py`**: Add any new constants or configurations required for the new model or indicator.

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your enhancements.
