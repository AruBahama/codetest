# Pairs Trading Research Framework

## About
This project implements a CPU-friendly pipeline for developing and back-testing a machine-learning-driven pairs-trading strategy. It collects daily OHLCV data for the current S&P 500, compresses 60-day feature windows with a convolutional autoencoder, clusters the resulting fingerprints, selects the most stationary pair from each cluster, trains a PPO reinforcement-learning agent on every chosen pair, and evaluates those policies in an out-of-sample back-test starting with a USD 1,000 stake. The entire process is offline so results are fully reproducible.

pairs_trading_system/
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── snp.csv              # your custom subset of S&P-500 tickers
│   ├── raw/                 # auto-filled by 01_data_collection.ipynb
│   └── processed/           # auto-filled by 02_preprocessing.ipynb
│
├── notebooks/
│   01_data_collection.ipynb
│   02_preprocessing.ipynb
│   03_autoencoder_training.ipynb
│   04_clustering.ipynb
│   05_rl_training.ipynb
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   │
│   ├── data/
│   │   ├── downloader.py
│   │   ├── feature_engineer.py
│   │   ├── scaler.py
│   │   └── window_builder.py
│   │
│   ├── autoencoder/
│   │   ├── cae_model.py
│   │   ├── train_cae.py
│   │   └── utils.py
│   │
│   ├── clustering/
│   │   ├── cluster_utils.py
│   │   └── select_pairs.py
│   │
│   ├── rl/
│   │   ├── envs.py
│   │   ├── train_agent.py
│   │   ├── agent_utils.py
│   │   └── hyperparams.py
│   │
│   ├── backtest/
│   │   ├── backtester.py
│   │   └── metrics.py
│   │
│   └── utils/
│       └── general.py
│
├── scripts/
│   ├── run_full_pipeline.py
│   ├── evaluate_backtest.py
│   └── launch_rl_training.sh
│
└── logs/                     # auto-generated at runtime

