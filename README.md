# Retail Demand Forecasting

28-day demand forecasts for 30,490 Walmart store-item series (M5 dataset),
served through a FastAPI + React product with a Gemini-powered explanatory
assistant.

## Model

Frozen LightGBM Tweedie blend: 0.60 × direct (38 features) + 0.40 ×
recursive (32 features). Validation RMSE 2.0929 / MAE 1.0395 on 853,720
held-out predictions. Model binaries are hash-checked — a regression test
fails the build if either changes.

## Stack

| Layer | Tech |
|---|---|
| Data | DuckDB + parquet, 130 MB portable layer |
| Backend | FastAPI, 34 routes |
| Frontend | React + TypeScript (Vite) |
| Assistant | Gemini, read-only, grounded against verified context |
| Deploy | EC2 + Docker Compose, GitHub Actions, OIDC (no stored AWS keys, no SSH) |

## Run it

```bash
python tasks.py build-db      # once
python tasks.py docker-up
python tasks.py smoke         # verify
```

App: `localhost:8080`. API docs: `localhost:8000/docs`.

## Team

See [`TEAM.md`](TEAM.md).
