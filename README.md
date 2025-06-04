# Pairs Trading Research Framework

## About
This project implements a CPU-friendly pipeline for developing and back-testing a machine-learning-driven pairs-trading strategy. It collects daily OHLCV data for the current S&P 500, compresses 60-day feature windows with a convolutional autoencoder, clusters the resulting fingerprints, selects the most stationary pair from each cluster, trains a PPO reinforcement-learning agent on every chosen pair, and evaluates those policies in an out-of-sample back-test starting with a USD 1,000 stake. The entire process is offline so results are fully reproducible.

pairs_trading_system/
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/                 # downloaded OHLCV CSVs (one file per ticker)
│   └── processed/           # cleaned, scaled, and windowed feature arrays
│
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_autoencoder_training.ipynb
│   ├── 04_clustering.ipynb
│   └── 05_rl_training.ipynb
│
├── src/
│   ├── __init__.py
│   │
│   ├── config.py            # global settings (tickers list, date ranges, hyperparameters)
│   │
│   ├── data/
│   │   ├── downloader.py     # functions to fetch OHLCV via yfinance
│   │   ├── feature_engineer.py  # compute technical indicators & pull fundamentals
│   │   ├── scaler.py         # fit/transform StandardScaler on training set
│   │   └── window_builder.py # create rolling windows (e.g., 60-day sequences)
│   │
│   ├── autoencoder/
│   │   ├── cae_model.py      # definition of convolutional autoencoder (PyTorch/Keras)
│   │   ├── train_cae.py      # script to train CAE & save encoder weights
│   │   └── utils.py          # utilities for CAE (dataset loader, loss tracking, checkpoints)
│   │
│   ├── clustering/
│   │   ├── cluster_utils.py  # functions to load latent vectors, run agglomerative clustering
│   │   └── select_pairs.py   # pick closest‐distance pairs within each cluster
│   │
│   ├── rl/                   
│   │   ├── envs.py           # custom Gym‐style environment for one pair
│   │   ├── train_agent.py    # code to loop over M selected pairs and train PPO
│   │   ├── agent_utils.py    # helper functions (experience replay if needed, logging)
│   │   └── hyperparams.py    # default RL hyperparameters (learning rate, gamma, etc.)
│   │
│   ├── backtest/
│   │   ├── backtester.py     # backtesting framework to simulate strategy on hold‐out data
│   │   └── metrics.py        # compute Sharpe ratio, max drawdown, CAGR, etc.
│   │
│   └── utils/
│       └── general.py        # file‐I/O helpers, plotting functions, logging setup
│
├── scripts/
│   ├── run_full_pipeline.py  # sequentially calls all steps 1→4 end-to-end (overwrites data each run)
│   ├── evaluate_backtest.py  # run backtests over trained agents and generate performance report
│   └── launch_rl_training.sh # shell script (or slurm job) to launch RL on GPU/cluster
│
└── logs/                     # training/validation loss logs, clustering outputs, RL checkpoints
