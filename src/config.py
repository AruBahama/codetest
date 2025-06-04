"""
Global configuration for the entire pipeline.
Modify here—everything else imports from this file.
"""

from pathlib import Path

# ---------- Paths ----------
ROOT_DIR        = Path(__file__).resolve().parents[1]
DATA_DIR        = ROOT_DIR / "data"
RAW_DIR         = DATA_DIR / "raw"
PROC_DIR        = DATA_DIR / "processed"
LOG_DIR         = ROOT_DIR / "logs"

# ---------- Universe ----------
TICKER_FILE     = DATA_DIR / "snp.csv"        # one symbol per line
START_DATE      = "2015-01-01"
END_DATE        = "2024-12-31"

# ---------- Windowing ----------
WINDOW_LENGTH   = 60          # rolling window length (days)
PREDICTION_GAP  = 1           # unused for now, placeholder

# ---------- Auto-encoder ----------
LATENT_DIM      = 10
CAE_EPOCHS      = 500
CAE_BATCH_SIZE  = 256

# ---------- Clustering ----------
N_CLUSTERS      = 10
PAIRS_PER_CLUST = 15          # top-15 closest pairs by Euclidean

# ---------- Reinforcement Learning ----------
INIT_CAPITAL    = 1_000       # $1 000 per pair
TOTAL_EPISODES  = 1_000
RL_ALGO         = "PPO"       # Ray RLlib ID
RL_CONFIG       = {}          # default—overridden in src/rl/hyperparams.py

# ---------- Backtest ----------
TRADE_FREQUENCY = "1D"        # once per day
METRICS         = ["total_pnl", "annual_return", "beta", "alpha",
                   "max_drawdown", "sharpe", "sortino", "calmar"]

# ---------- Misc ----------
SEED            = 42
NUM_WORKERS     =  max(1, os.cpu_count()-1)
