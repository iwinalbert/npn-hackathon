"""
M5 retail demand forecasting pipeline.

Modules
-------
config            paths, dataset constants, the backtest origin definitions
data_loader       raw CSVs -> compact wide matrices (read-only on raw_dataset/)
features          leakage-safe feature engineering, grouped A-G
backtest          fixed-origin 28-day train/validation/future frame assembly
metrics           RMSE / MAE / WAPE / bias
validation_checks correctness + empirical leakage tests
report_pdf        markdown -> PDF renderer for stage reports
"""

__version__ = "0.1.0"
