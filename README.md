# Electricity Net-Demand Forecasting in France — Statistical Learning

Forecasting the **net electricity demand** of France (Load − Wind − Solar) over the
2022–2023 *sobriety period*, using statistical / supervised-learning models in **R**.
Built for the Kaggle competition
[net-load-soberty-period](https://www.kaggle.com/competitions/net-load-soberty-period)
(M2 Mathematics & AI, Université Paris-Saclay).

This is the **statistical-learning** counterpart of the deep-learning project
[Electricity-Demand-Forecasting-in-France](https://github.com/mlahozy21/Deep-Learning-Electricity-Demand-Forecasting)
(Python, multi-output neural network).

## Problem

The target is the **net demand**, the electricity that renewable production (solar +
wind) cannot cover:

```
Net_demand = Load − (Wind_power + Solar_power)
```

Daily data spans 2013 → September 2022 (calendar, weather and lagged
production/consumption variables); the test horizon is the sobriety period
(September 2022 → September 2023). Submissions are scored with the **pinball loss at
quantile 0.8**, which penalises under-forecasting more strongly — a conservative,
grid-safe objective.

## Models

Several learners are trained and combined by online expert aggregation:

| Model | Description |
|-------|-------------|
| **GAM + Kalman filter** | Generalized Additive Model with online state-space adaptation (`viking`, EM-tuned variances). |
| **GAM Ridge** | GAM with double penalisation + term selection for a frugal, robust fit. |
| **GAM + Random Forest on residuals** | A GAM whose residuals are modelled by a Random Forest to capture leftover non-linearities. |
| **Random Forest** | Tree ensemble over the full covariate set (`mtry` tuned). |
| **QGAM** | Quantile GAMs (q = 0.2 and 0.8) to bound the predictions. |
| **Expert aggregation** | `opera::mixture` (MLpol) combines all experts online under the pinball loss at 0.8. |

Indicative validation results (RMSE / pinball@0.8):

| Model | RMSE | Pinball |
|-------|-----:|--------:|
| GAM + Kalman | 1629 | 537 |
| GAM Ridge | 1353 | 562 |
| GAM + RF (residuals) | 1754 | 716 |
| Random Forest | 2064 | 465 |

The final submission is the online expert aggregation of these models.

## Repository structure

```
.
├── README.md  LICENSE  .gitignore
├── R/
│   └── score.R                 # metrics: RMSE, MAPE, pinball loss
├── Data/
│   ├── train.csv               # 2013 – Sep 2022 (Net_demand + covariates)
│   └── test.csv                # sobriety period to predict (Kaggle format: Id, Usage)
├── Models/                      # cached pre-trained models (.RDS) loaded by the script
│   ├── KalmanEM_def.RDS  gam_ridge.RDS  randomForest.RDS  qgam08.RDS  qgam02.RDS
├── scripts/
│   ├── final_submission.R      # trains/loads models, aggregates experts, writes submission
│   └── analysis.Rmd            # exploratory data analysis and model write-up
├── Submissions/                # generated submission(s)
└── docs/
    └── report.pdf              # project report
```

## Requirements

R (≥ 4.0) and the packages:

```r
install.packages(c("tidyverse", "mgcv", "qgam", "randomForest", "forecast",
                   "viking", "opera", "magrittr"))
```

## Usage

Run from the repository root (paths are relative to it):

```r
source("scripts/final_submission.R")
# or:  Rscript scripts/final_submission.R
```

The script loads the cached models in `Models/`, builds the GAM/Kalman, ridge-GAM,
GAM+RF, Random-Forest and quantile-GAM forecasts, aggregates them online with `opera`,
and writes `Submissions/submission.csv` in the Kaggle format (`Id, Net_demand`).

To retrain the heavy models from scratch, uncomment the corresponding
`saveRDS(...)` blocks in `scripts/final_submission.R`.

## License

Released under the MIT License — see `LICENSE`.
