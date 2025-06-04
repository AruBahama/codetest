"""
CLI one-liner: python scripts/run_full_pipeline.py
Downloads data, engineers features, trains CAE, clusters, selects pairs,
trains PPO agents, and runs backtests.  Overwrites intermediate files.
"""
from src.data.downloader      import batch_download
from src.data.feature_engineer import batch_engineer
from src.autoencoder.train_cae import train_cae          # noqa
from src.clustering.cluster_utils import cluster_latents # noqa
from src.clustering.select_pairs  import select_pairs    # noqa
from src.rl.train_agent           import train_all_pairs # noqa
from src.backtest.backtester      import run_backtests   # noqa

def main() -> None:
    batch_download()        # Step 1
    batch_engineer()        # Step 2
    train_cae()             # Step 3
    cluster_latents()       # Step 4
    select_pairs()          # Step 4b
    train_all_pairs()       # Step 5
    run_backtests()         # Step 6

if __name__ == "__main__":
    main()
