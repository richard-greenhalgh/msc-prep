# train.py
import time
from dataclasses import dataclass

import numpy as np

import src.NNN as MyNN
from src.data import preprocess, get_dataset
from src.data import Logger
from src.vis import make_live_plot_callback, final_plot, plot_last_hidden_pca

VERBOSE = True
DEBUG = False

@dataclass
class TrainConfig:
    hidden_layers: list[int]
    seed: int = 42
    max_epochs: int = 30
    batch_size: int = 32
    optimizer: str = MyNN.OPTIMIZER_ADAM
    learning_rate: float = 0.001
    learning_rate_decay: float = 0.9
    live_plot: bool = False
    live_update_freq: int = 100   # redraw chart every X batches

def run(cfg: TrainConfig = None, showLossPlot=True, showPCA=True, quiet=False):
    # !!! SET HIDDEN LAYERS HERE !!!
    if cfg is None:
        cfg = TrainConfig([32, 32])
    x_train, y_train, x_test, y_test = get_dataset()
    x_train_new, y_train_new = preprocess(x_train, y_train)
    x_test_new, y_test_new = preprocess(x_test, y_test)

    n_inputs = x_train_new.shape[1]
    n_outputs = y_train_new.shape[1]
    model = MyNN.Model(n_inputs, cfg.hidden_layers+[n_outputs], seed=cfg.seed, optimizer=cfg.optimizer)

    callback, finish = (None, None)
    if cfg.live_plot:
        callback, finish = make_live_plot_callback(cfg.live_update_freq)
    
    # train the model...
    t0 = time.perf_counter()
    results = model.fit(
        x_train_new,
        y_train_new,
        max_epochs=cfg.max_epochs,
        batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        learning_rate_decay=cfg.learning_rate_decay,
        callback_func=callback,
        validation_split=0.2,
        early_stop=True,
        early_patience=5,
        restore_best_weights=True
    )
    elapsed = time.perf_counter() - t0
    acc_train = model.accuracy(x_train_new, y_train_new) * 100  # accuracy as %
    elapsed2 = time.perf_counter() - t0 - elapsed
    acc_test = model.accuracy(x_test_new, y_test_new) * 100     # accuracy as %
    infer_rate = 1000 * elapsed2 / len(x_train_new) # time per 1000 samples
    
    loss_train = model.calcLoss(x_train_new, y_train_new)
    loss_test = model.calcLoss(x_test_new, y_test_new)

    # convergence metrics:
    epoch_loss = results["LOSS_CURVE_EPOCH"].tolist()
    if len(epoch_loss) >= 2:
        last_delta = epoch_loss[-2] - epoch_loss[-1]
        rel_delta = last_delta / epoch_loss[-2]
    else:
        last_delta = None
        rel_delta = None

    K = min(5, len(epoch_loss))
    recent = epoch_loss[-K:]
    conv_rate = (recent[0] - recent[-1]) / K

    # collate results
    log = Logger()

    run_summary = {
        "run_id": log.runID,
        "timestamp": log.timestamp,
        "code_fingerprint": log.code_fingerprint,
        "seed": cfg.seed,
        "hidden_layers": str(cfg.hidden_layers),
        "n_inputs": int(n_inputs),
        "n_outputs": int(n_outputs),
        "n_param": int(results["NPARAM"]),
        "loss_method": results["LOSS_METHOD"],
        "epochs": cfg.max_epochs,
        "batch_size": cfg.batch_size,
        "optimizer": cfg.optimizer,
        "learning_rate": cfg.learning_rate,
        "learning_rate_decay": cfg.learning_rate_decay,
        "train_accuracy": float(acc_train),
        "test_accuracy": float(acc_test),
        "generalisation_gap_pp": float(acc_train - acc_test),
        "train_loss": float(loss_train),
        "test_loss": float(loss_test),
        "training_seconds": elapsed,
        "seconds_per_epoch": elapsed / cfg.max_epochs,
        "batch_loss": results["LOSS_CURVE_BATCH"].tolist(),
        "epoch_loss": results["LOSS_CURVE_EPOCH"].tolist(),
        "conv_last_delta": float(last_delta) if last_delta is not None else None,
        "conv_rel_delta": float(rel_delta) if rel_delta is not None else None,
        "conv_rate": float(conv_rate),
        "epochs_run": int(results["EPOCHS_RUN"]),
        "best_val_loss": float(results["BEST_VAL_LOSS"]) if np.isfinite(results["BEST_VAL_LOSS"]) else None,
        "best_epoch": int(results["BEST_EPOCH"]) if results["BEST_EPOCH"] >= 0 else None,
        "val_loss": float(results["BEST_VAL_LOSS"]) if np.isfinite(results["BEST_VAL_LOSS"]) else None,
        "val_accuracy": float(results["VAL_ACC_CURVE_EPOCH"][results["BEST_EPOCH"]] * 100)
            if results["BEST_EPOCH"] >= 0 else None,
        "val_loss_curve": results["VAL_LOSS_CURVE_EPOCH"].tolist(),
        "val_acc_curve": results["VAL_ACC_CURVE_EPOCH"].tolist(),
    }

    if quiet:
        print(f"Test accuracy: {acc_test:.3f}%  ... completed {elapsed:.3f}s")
    else:
        print("=" * 80)
        print(f"Model architecture (layers):", f"inputs({n_inputs}),", f"hidden{cfg.hidden_layers},", f"outputs({n_outputs})")
        print(f"Model parameter count      : {results['NPARAM']}")
        print(f"Training loss method       : {results['LOSS_METHOD']}")
        print("=" * 80)
        print(f"RNG seed                  : {cfg.seed}")
        print(f"Number of training samples: {len(x_train_new)}")
        print(f"Number of epochs          : {results["EPOCHS_RUN"]}")
        print(f"Samples per batch         : {cfg.batch_size}")
        print("=" * 80)
        print(f"Accuracy on in-sample training data: {acc_train:.3f}%" )
        print(f"Accuracy on out-of-sample test data: {acc_test:.3f}%" )
        print(f"Generalisation gap                 : {acc_train - acc_test:.3f} pp")
        print("    (high positive value may indicate overfitting)")
        print("=" * 80)
        print(f"Mean loss for in-sample training data: {loss_train:.6f}")
        print(f"Mean loss for out-of-sample test data: {loss_test:.6f}")
        print("=" * 80)
        print(f"Model training and inference time:")
        print(f"    Training            : {elapsed:.3f}s  ({(elapsed/cfg.max_epochs):.3f}s per epoch)")
        print(f"    Infer (training set): {elapsed2:.3f}s  ({infer_rate:.3f}s per 1000 samples)")
        print("=" * 80)

    # display resulting accuracy, draw final plot?
    if cfg.live_plot:
        if finish is not None: finish()
    elif showLossPlot:
        final_plot(run_summary)
    
    # display 2D PCA plot?
    if showPCA:
        plot_last_hidden_pca(model, x_train_new, y_train_new, run_summary, n_samples=5000, save_path=None, show=True)

    if showLossPlot or showPCA:
        import matplotlib.pyplot as plt
        plt.show()

    # log and results
    log.save_run_artifacts(
        run_summary,
        model,
        train_data=(x_train_new, y_train_new),
        test_data=(x_test_new, y_test_new),
    )
    log.append_run_csv(run_summary)
    return run_summary

#==============================================================================

if __name__ == "__main__":
    run()

