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
    hidden_options = [
        [32, 32],
        [64, 64],
        [128, 128],
        [256, 256],
    ]
    learning_rates = [0.1]
    batch_sizes = [32]
    epochs = [30]


    results = []
    for hidden_layers, lr, batch_size, max_epochs in product(
        hidden_options,
        learning_rates,
        batch_sizes,
        epochs,
    ):
        cfg = TrainConfig(
            hidden_layers=hidden_layers,
            learning_rate=lr,
            batch_size=batch_size,
            max_epochs=max_epochs,
            live_plot=False,
        )

        results.append(run(cfg, showPlot=False, quiet=True))
    
    log = Logger()
    sweep_csv = log.get_CSV_path(subdir="sweeps", filename=f"sweep_{log.runID}.csv")
    for r in results:
        log.append_run_csv(r, csv_path=sweep_csv)

if __name__ == "__main__":
    main()