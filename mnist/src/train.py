# train.py
import time
from dataclasses import dataclass

import src.NNN as MyNN
from src.data import preprocess, get_dataset
from src.data import Logger
from src.vis import make_live_plot_callback, final_plot

VERBOSE = True
DEBUG = False

@dataclass
class TrainConfig:
    hidden_layers: list[int]
    seed: int = 42
    max_epochs: int = 3
    batch_size: int = 32
    learning_rate: float = 0.1
    live_plot: bool = False
    live_update_freq: int = 100   # redraw chart every X batches

def run(cfg: TrainConfig = None):
    # !!! SET HIDDEN LAYERS HERE !!!
    if cfg is None:
        cfg = TrainConfig([32, 32])
    x_train, y_train, x_test, y_test = get_dataset()
    x_train_new, y_train_new = preprocess(x_train, y_train)
    x_test_new, y_test_new = preprocess(x_test, y_test)

    n_inputs = x_train_new.shape[1]
    n_outputs = y_train_new.shape[1]
    model = MyNN.Model(n_inputs, cfg.hidden_layers+[n_outputs], seed=cfg.seed)

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
        callback_func=callback
    )
    elapsed = time.perf_counter() - t0
    acc_train = model.accuracy(x_train_new, y_train_new) * 100  # accuracy as %
    elapsed2 = time.perf_counter() - t0 - elapsed
    acc_test = model.accuracy(x_test_new, y_test_new) * 100     # accuracy as %
    infer_rate = 1000 * elapsed2 / len(x_train_new) # time per 1000 samples
    
    loss_train = model.calcLoss(x_train_new, y_train_new)
    loss_test = model.calcLoss(x_test_new, y_test_new)

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
        "learning_rate": cfg.learning_rate,
        "train_accuracy": float(acc_train),
        "test_accuracy": float(acc_test),
        "generalisation_gap_pp": float(acc_train - acc_test),
        "train_loss": float(loss_train),
        "test_loss": float(loss_test),
        "training_seconds": elapsed,
        "seconds_per_epoch": elapsed / cfg.max_epochs,
        "batch_loss": results["LOSS_CURVE_BATCH"].tolist(),
        "epoch_loss": results["LOSS_CURVE_EPOCH"].tolist(),
    }

    print("=" * 80)
    print(f"Model architecture (layers):", f"inputs({n_inputs}),", f"hidden{cfg.hidden_layers},", f"outputs({n_outputs})")
    print(f"Model parameter count      : {results['NPARAM']}")
    print(f"Training loss method       : {results['LOSS_METHOD']}")
    print("=" * 80)
    print(f"RNG seed                  : {cfg.seed}")
    print(f"Number of training samples: {len(x_train_new)}")
    print(f"Number of epochs          : {cfg.max_epochs}")
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
    else:
        final_plot(run_summary)
    
    # log and results
    log.save_run_artifacts(run_summary)
    log.append_run_csv(run_summary)
    return run_summary

#==============================================================================

if __name__ == "__main__":
    run()

