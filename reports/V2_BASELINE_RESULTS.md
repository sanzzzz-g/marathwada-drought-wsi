# V2 WSI Baseline Evaluation

## Objective
Evaluate whether monthly WSI can be forecast at horizons of one, two, and three months, and compare forecasting models against simple persistence.

## Evaluation setup
- Training target period: January 2004–December 2017
- Validation target period: January 2018–December 2020
- Test target period: January 2021–December 2024
- Models evaluated: Persistence, Seasonal Persistence, WSI Autoregression, Ridge Regression, and XGBoost
- Evaluation metrics: MAE and RMSE

## Main result
Persistence achieved test-set mean MAE of 0.4221 and mean RMSE of 0.5507. None of the more complex models outperformed persistence on the retrospective test set.

WSI Autoregression performed best on validation MAE, but this advantage did not generalize to the test period.

## Conclusion
Persistence remains the benchmark. The current results do not establish that complex models improve WSI forecasting.

## Pending experiments
1. Document and evaluate a revised WSI weighting scheme.
2. Obtain and validate historical reservoir storage and capacity observations.
3. Test whether reservoir predictors improve forecasting skill.
4. Validate WSI against independently observed water scarcity.
5. Preserve temporal separation and compare all models against persistence.
