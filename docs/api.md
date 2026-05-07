# API Reference

::: src.persistent_homology
    options:
      show_source: false
      members:
        - PersistentHomologyAnalyzer
        - compute_features_for_windows
        - run_persistence_pipeline

::: src.advanced_features
    options:
      show_source: false
      members:
        - build_advanced_features
        - garman_klass_volatility
        - parkinson_volatility
        - get_tda_features
        - get_ml_features

::: src.binance_data
    options:
      show_source: false
      members:
        - HighFreqFetcher
        - run_hf_pipeline

::: src.multi_asset_pipeline
    options:
      show_source: false
      members:
        - fetch_multi_asset_pool
        - get_combined_feature_cols

::: src.ml_signals
    options:
      show_source: false
      members:
        - MLSignalGenerator
        - grid_search_ml
        - evaluate_ml_kfold

::: src.validation_v2
    options:
      show_source: false
      members:
        - wilson_interval
        - time_series_kfold
        - bootstrap_metric
        - run_validation_v2

::: src.regime_filter
    options:
      show_source: false
      members:
        - apply_regime_filter
        - vol_regime_mask

::: src.backtester
    options:
      show_source: false
      members:
        - Backtester
        - format_metrics
        - run_backtest

!!! note
    The API reference is auto-generated from docstrings in source files. Run `mkdocs build` to regenerate.
