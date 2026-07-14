# Multi-ticker Feature Efficacy Report

The notebook was run independently for each requested ticker using adjusted OHLCV data from 2015-01-01 through 2026-07-07. Correlations use the notebook’s Spearman methodology and future 60-day return labels.

## Summary

| ticker   |   rows |   years | top_feature_60d      |   top_corr_60d | top_stable_feature_60d   |   top_stable_median_60d |   top_stable_sign_stability |
|:---------|-------:|--------:|:---------------------|---------------:|:-------------------------|------------------------:|----------------------------:|
| GLD      |   2773 |      12 | macd_signal_strength |        -0.0998 | mr_ewma50_vol_gap        |                  0.3419 |                      0.6667 |
| XLK      |   2773 |      12 | mr_ewma50_vol_gap    |         0.1803 | mr_ewma50_vol_gap        |                  0.4541 |                      0.8333 |
| XLV      |   2773 |      12 | mr_ewma50_vol_gap    |         0.2096 | ewma50_slope             |                 -0.3465 |                      0.8333 |
| ITA      |   2773 |      12 | acc_momentum_quality |        -0.1556 | mr_ewma50_vol_gap        |                  0.3443 |                      0.9167 |
| SPY      |   2773 |      12 | mr_ewma50_vol_gap    |         0.1812 | frog_in_the_pan          |                 -0.3594 |                      0.8333 |
| QQQ      |   2773 |      12 | mr_ewma50_vol_gap    |         0.1820 | mr_ewma50_vol_gap        |                  0.4294 |                      0.8333 |
| SOXX     |   2773 |      12 | acc_klinger_signal   |        -0.1307 | acc_macd_signal          |                 -0.4367 |                      0.7500 |
| UVXY     |   2773 |      12 | acc_momentum_quality |        -0.0974 | mr_ewma50_vol_gap        |                  0.4573 |                      0.8333 |
| XLE      |   2773 |      12 | ewma50_slope         |        -0.1460 | ewma50_slope             |                 -0.3274 |                      0.7500 |

## GLD

Dataset rows: 112 correlation observations (feature-label pairs); yearly stability was computed across 12 years.

Top pooled correlations for `future_return_60d`:

| feature              |   correlation |   abs_correlation |
|:---------------------|--------------:|------------------:|
| macd_signal_strength |       -0.0998 |            0.0998 |
| cmf_days_since_cross |       -0.0824 |            0.0824 |
| bollinger_bandwidth  |        0.0533 |            0.0533 |
| acc_momentum_quality |       -0.0465 |            0.0465 |
| mr_ewma50_vol_gap    |        0.0457 |            0.0457 |

Top yearly-stable relationships for `future_return_60d`:

| feature                      |   median_corr |   mean_corr |   sign_stability |   years |
|:-----------------------------|--------------:|------------:|-----------------:|--------:|
| mr_ewma50_vol_gap            |        0.3419 |      0.2830 |           0.6667 |      12 |
| cmf                          |       -0.3195 |     -0.2000 |           0.6667 |      12 |
| acc_macd_signal              |       -0.2353 |     -0.2015 |           0.5833 |      12 |
| fip_over_bollinger_bandwidth |       -0.2143 |     -0.1276 |           0.5833 |      12 |
| acc_momentum_quality         |       -0.1705 |     -0.1680 |           0.5833 |      12 |

## XLK

Dataset rows: 112 correlation observations (feature-label pairs); yearly stability was computed across 12 years.

Top pooled correlations for `future_return_60d`:

| feature                      |   correlation |   abs_correlation |
|:-----------------------------|--------------:|------------------:|
| mr_ewma50_vol_gap            |        0.1803 |            0.1803 |
| acc_macd_signal              |       -0.1722 |            0.1722 |
| fip_over_bollinger_bandwidth |       -0.1677 |            0.1677 |
| acc_momentum_quality         |       -0.1639 |            0.1639 |
| frog_in_the_pan              |       -0.1400 |            0.1400 |

Top yearly-stable relationships for `future_return_60d`:

| feature              |   median_corr |   mean_corr |   sign_stability |   years |
|:---------------------|--------------:|------------:|-----------------:|--------:|
| mr_ewma50_vol_gap    |        0.4541 |      0.3852 |           0.8333 |      12 |
| acc_momentum_quality |       -0.4091 |     -0.3421 |           0.7500 |      12 |
| acc_macd_signal      |       -0.4066 |     -0.2745 |           0.7500 |      12 |
| ewma50_slope         |       -0.3318 |     -0.3385 |           0.9167 |      12 |
| frog_in_the_pan      |       -0.3164 |     -0.3142 |           0.9167 |      12 |

## XLV

Dataset rows: 112 correlation observations (feature-label pairs); yearly stability was computed across 12 years.

Top pooled correlations for `future_return_60d`:

| feature              |   correlation |   abs_correlation |
|:---------------------|--------------:|------------------:|
| mr_ewma50_vol_gap    |        0.2096 |            0.2096 |
| acc_macd_signal      |       -0.2061 |            0.2061 |
| ewma50_slope         |       -0.1930 |            0.1930 |
| acc_momentum_quality |       -0.1919 |            0.1919 |
| cmf_cross_signal     |        0.1739 |            0.1739 |

Top yearly-stable relationships for `future_return_60d`:

| feature                 |   median_corr |   mean_corr |   sign_stability |   years |
|:------------------------|--------------:|------------:|-----------------:|--------:|
| ewma50_slope            |       -0.3465 |     -0.2552 |           0.8333 |      12 |
| mr_ewma50_vol_gap       |        0.3089 |      0.3535 |           0.9167 |      12 |
| acc_macd_signal         |       -0.2898 |     -0.2436 |           0.8333 |      12 |
| acc_momentum_quality    |       -0.2604 |     -0.2741 |           0.8333 |      12 |
| klinger_signal_strength |        0.1704 |      0.1652 |           0.6667 |      12 |

## ITA

Dataset rows: 112 correlation observations (feature-label pairs); yearly stability was computed across 12 years.

Top pooled correlations for `future_return_60d`:

| feature                      |   correlation |   abs_correlation |
|:-----------------------------|--------------:|------------------:|
| acc_momentum_quality         |       -0.1556 |            0.1556 |
| frog_in_the_pan              |       -0.1461 |            0.1461 |
| fip_over_bollinger_bandwidth |       -0.1439 |            0.1439 |
| cmf                          |       -0.1269 |            0.1269 |
| acc_macd_signal              |       -0.1248 |            0.1248 |

Top yearly-stable relationships for `future_return_60d`:

| feature              |   median_corr |   mean_corr |   sign_stability |   years |
|:---------------------|--------------:|------------:|-----------------:|--------:|
| mr_ewma50_vol_gap    |        0.3443 |      0.3020 |           0.9167 |      12 |
| acc_momentum_quality |       -0.3224 |     -0.3105 |           0.9167 |      12 |
| frog_in_the_pan      |       -0.2565 |     -0.2063 |           0.7500 |      12 |
| cmf_days_since_cross |       -0.2240 |     -0.1510 |           0.7500 |      12 |
| acc_macd_signal      |       -0.2102 |     -0.2411 |           0.8333 |      12 |

## SPY

Dataset rows: 112 correlation observations (feature-label pairs); yearly stability was computed across 12 years.

Top pooled correlations for `future_return_60d`:

| feature              |   correlation |   abs_correlation |
|:---------------------|--------------:|------------------:|
| mr_ewma50_vol_gap    |        0.1812 |            0.1812 |
| acc_momentum_quality |       -0.1719 |            0.1719 |
| ewma50_slope         |       -0.1519 |            0.1519 |
| acc_macd_signal      |       -0.1456 |            0.1456 |
| acc_klinger_signal   |       -0.1230 |            0.1230 |

Top yearly-stable relationships for `future_return_60d`:

| feature                      |   median_corr |   mean_corr |   sign_stability |   years |
|:-----------------------------|--------------:|------------:|-----------------:|--------:|
| frog_in_the_pan              |       -0.3594 |     -0.3044 |           0.8333 |      12 |
| mr_ewma50_vol_gap            |        0.3546 |      0.2849 |           0.8333 |      12 |
| fip_over_bollinger_bandwidth |       -0.3375 |     -0.3324 |           0.8333 |      12 |
| acc_momentum_quality         |       -0.3032 |     -0.2962 |           0.8333 |      12 |
| acc_macd_signal              |       -0.2713 |     -0.1815 |           0.8333 |      12 |

## QQQ

Dataset rows: 112 correlation observations (feature-label pairs); yearly stability was computed across 12 years.

Top pooled correlations for `future_return_60d`:

| feature              |   correlation |   abs_correlation |
|:---------------------|--------------:|------------------:|
| mr_ewma50_vol_gap    |        0.1820 |            0.1820 |
| acc_momentum_quality |       -0.1724 |            0.1724 |
| cmf_days_since_cross |       -0.1692 |            0.1692 |
| acc_macd_signal      |       -0.1577 |            0.1577 |
| cmf                  |       -0.1509 |            0.1509 |

Top yearly-stable relationships for `future_return_60d`:

| feature                      |   median_corr |   mean_corr |   sign_stability |   years |
|:-----------------------------|--------------:|------------:|-----------------:|--------:|
| mr_ewma50_vol_gap            |        0.4294 |      0.2955 |           0.8333 |      12 |
| frog_in_the_pan              |       -0.4168 |     -0.2584 |           0.7500 |      12 |
| acc_momentum_quality         |       -0.3848 |     -0.3025 |           0.7500 |      12 |
| fip_over_bollinger_bandwidth |       -0.3704 |     -0.2253 |           0.6667 |      12 |
| acc_macd_signal              |       -0.3491 |     -0.2381 |           0.8333 |      12 |

## SOXX

Dataset rows: 112 correlation observations (feature-label pairs); yearly stability was computed across 12 years.

Top pooled correlations for `future_return_60d`:

| feature                      |   correlation |   abs_correlation |
|:-----------------------------|--------------:|------------------:|
| acc_klinger_signal           |       -0.1307 |            0.1307 |
| fip_over_bollinger_bandwidth |       -0.1060 |            0.1060 |
| frog_in_the_pan              |       -0.0758 |            0.0758 |
| cmf_slope                    |       -0.0748 |            0.0748 |
| macd_signal_strength         |       -0.0630 |            0.0630 |

Top yearly-stable relationships for `future_return_60d`:

| feature                      |   median_corr |   mean_corr |   sign_stability |   years |
|:-----------------------------|--------------:|------------:|-----------------:|--------:|
| acc_macd_signal              |       -0.4367 |     -0.2395 |           0.7500 |      12 |
| mr_ewma50_vol_gap            |        0.3846 |      0.3090 |           0.8333 |      12 |
| acc_momentum_quality         |       -0.3818 |     -0.2722 |           0.6667 |      12 |
| fip_over_bollinger_bandwidth |       -0.3240 |     -0.2752 |           0.9167 |      12 |
| frog_in_the_pan              |       -0.3152 |     -0.2834 |           0.9167 |      12 |

## UVXY

Dataset rows: 112 correlation observations (feature-label pairs); yearly stability was computed across 12 years.

Top pooled correlations for `future_return_60d`:

| feature                 |   correlation |   abs_correlation |
|:------------------------|--------------:|------------------:|
| acc_momentum_quality    |       -0.0974 |            0.0974 |
| klinger_signal_strength |       -0.0844 |            0.0844 |
| mr_ewma50_vol_gap       |        0.0720 |            0.0720 |
| macd_signal_strength    |        0.0573 |            0.0573 |
| bollinger_bandwidth     |       -0.0565 |            0.0565 |

Top yearly-stable relationships for `future_return_60d`:

| feature              |   median_corr |   mean_corr |   sign_stability |   years |
|:---------------------|--------------:|------------:|-----------------:|--------:|
| mr_ewma50_vol_gap    |        0.4573 |      0.3360 |           0.8333 |      12 |
| acc_momentum_quality |       -0.3089 |     -0.2741 |           0.7500 |      12 |
| acc_macd_signal      |       -0.2602 |     -0.1703 |           0.6667 |      12 |
| frog_in_the_pan      |       -0.2098 |     -0.1845 |           0.6667 |      12 |
| cmf                  |       -0.1963 |     -0.2350 |           0.8333 |      12 |

## XLE

Dataset rows: 112 correlation observations (feature-label pairs); yearly stability was computed across 12 years.

Top pooled correlations for `future_return_60d`:

| feature            |   correlation |   abs_correlation |
|:-------------------|--------------:|------------------:|
| ewma50_slope       |       -0.1460 |            0.1460 |
| frog_in_the_pan    |       -0.1140 |            0.1140 |
| acc_macd_signal    |       -0.1092 |            0.1092 |
| acc_klinger_signal |       -0.0935 |            0.0935 |
| mr_ewma50_vol_gap  |        0.0811 |            0.0811 |

Top yearly-stable relationships for `future_return_60d`:

| feature            |   median_corr |   mean_corr |   sign_stability |   years |
|:-------------------|--------------:|------------:|-----------------:|--------:|
| ewma50_slope       |       -0.3274 |     -0.2182 |           0.7500 |      12 |
| acc_macd_signal    |       -0.3205 |     -0.2498 |           0.9167 |      12 |
| mr_ewma50_vol_gap  |        0.2497 |      0.2518 |           0.8333 |      12 |
| acc_klinger_signal |       -0.1731 |     -0.1569 |           0.6667 |      12 |
| frog_in_the_pan    |       -0.1568 |     -0.1003 |           0.8333 |      12 |
