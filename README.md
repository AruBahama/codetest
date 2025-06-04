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

