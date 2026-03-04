# UNISIM-I Proxy Model

Machine learning proxy model to emulate ECLIPSE reservoir simulation results for the UNISIM-I benchmark.

## Project Structure
UNISIM_PROXY_PROJECT/
│
├── data/
│ ├── raw
│ ├── processed
│
├── src/
│ ├── extract
│ ├── datasets
│ ├── models
│ └── validation
│
├── models/
├── reports/
└── notebooks

## Pipeline

1. Extract simulation results from ECLIPSE
2. Build tabular dataset
3. Parse case parameters
4. Build ML dataset
5. Train proxy model (XGBoost)
6. Validate predictions

## Example metrics


MAE : 17.69
RMSE : 50.22
R² : 0.966


## Applications

- reservoir optimization
- uncertainty analysis
- accelerated simulation