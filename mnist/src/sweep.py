# sweep.py
# multiple runs across hyperparamters

from itertools import product
from src.train import TrainConfig, run
from src.data import Logger

def main():
    """
    hidden_options = [
        [32, 32],
        [64, 64],
        [128, 64],
    ]
    learning_rates = [0.01, 0.03, 0.1]
    batch_sizes = [32, 64]
    epochs = [10]
    """
    n = 4
    hidden_options = [
        [32, 32],
    ]
    learning_rates = [0.001]
    lr_decays = [1.00, 0.95, 0.90, 0.85, 0.8]
    batch_sizes = [32]
    epochs = [50]


    results = []
    for hidden_layers, lr, decay, batch_size, max_epochs in product(
        hidden_options,
        learning_rates,
        lr_decays,
        batch_sizes,
        epochs,
    ):
        cfg = TrainConfig(
            hidden_layers=hidden_layers,
            learning_rate=lr,
            learning_rate_decay=decay,
            batch_size=batch_size,
            max_epochs=max_epochs,
            live_plot=False,
        )

        results.append(run(cfg, showLossPlot=False, showPCA=False, quiet=True))
    
    log = Logger()
    sweep_csv = log.get_CSV_path(subdir="sweeps", filename=f"sweep_{log.runID}.csv")
    for r in results:
        log.append_run_csv(r, csv_path=sweep_csv)

if __name__ == "__main__":
    main()