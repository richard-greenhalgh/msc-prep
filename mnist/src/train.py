# train.py
from dataclasses import dataclass
import time, os, csv, json, hashlib
from datetime import datetime
import src.NNN as MyNN
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

VERBOSE = True
DEBUG = False
JSON_BLACKLIST = {'batch_loss', 'epoch_loss'}

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
    run_id = make_run_id()
    timestamp = datetime.now().isoformat(timespec="seconds")

    run_summary = {
        "run_id": run_id,
        "timestamp": timestamp,
        "code_fingerprint": get_code_fingerprint(),
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
    save_run_artifacts(run_summary)
    append_run_csv(run_summary)
    return run_summary

def make_live_plot_callback(update_every=10, ma_window=20):
    x_data, y_data = [], []

    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))
    line_raw, = ax.plot([], [], label="Batch Loss", alpha=0.4)
    line_ma, = ax.plot([], [], label="Moving Avg", linewidth=2)
    ax.set_xlabel("Global batch")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    ax.set_title("Live training loss")
    ax.legend()

    plt.show(block=False)

    def callback(data: dict):
        x = int(data["BATCH"])
        y = float(data["BATCH_LOSS"])

        x_data.append(x)
        y_data.append(y)

        if x % update_every == 0:
            line_raw.set_data(x_data, y_data)
            if len(y_data) >= ma_window:
                y_ma = np.convolve(y_data, np.ones(ma_window)/ma_window, mode='valid')
                x_ma = x_data[ma_window - 1:]  # align lengths
                line_ma.set_data(x_ma, y_ma)
            ax.relim()
            ax.autoscale_view()
            fig.canvas.draw()
            fig.canvas.flush_events()

    def final_live_plot():
        if len(x_data) == 0:
            return

        line_raw.set_data(x_data, y_data)
        if len(y_data) >= ma_window:
                y_ma = np.convolve(y_data, np.ones(ma_window)/ma_window, mode='valid')
                x_ma = x_data[ma_window - 1:]  # align lengths
                line_ma.set_data(x_ma, y_ma)
        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.ioff()
        plt.show()

    return callback, final_live_plot

def final_plot(summary: dict, curve_name: str = 'batch_loss', save_path=None, show=True):
    is_batch = curve_name.lower().find('batch') != -1
    is_epoch = not is_batch
    y_data = list(summary[curve_name])
    x_data = list(range(len(y_data)))
    ma_window = min(len(y_data), max(3, int(len(y_data)/50.0)))
    fig, ax = plt.subplots(figsize=(10, 6))
    label = 'Batch Loss' if is_batch else 'Epoch Loss' if is_epoch else '***ERROR***'
    ax.plot(x_data, y_data, label=label, alpha=0.4)
    
    if len(y_data) >= ma_window and is_batch:
        y_ma = np.convolve(y_data, np.ones(ma_window)/ma_window, mode='valid')
        x_ma = x_data[ma_window - 1:]  # align lengths
        ax.plot(x_ma, y_ma, label=f"Moving Avg ({ma_window})", linewidth=2)
    
    ax.set_xlabel(label.split()[0])
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    curve_type = 'batch' if is_batch else 'epoch' if is_epoch else '???'
    title = (
        f"MNIST loss by {curve_type} | hidden={summary['hidden_layers']} | "
        f"loss={summary['loss_method']} | "
        f"train_acc={summary['train_accuracy']:.2f}% | "
        f"test_acc={summary['test_accuracy']:.2f}%"
    )
    ax.set_title(title)
    ax.legend()

    ax.relim()
    ax.autoscale_view()

    plt.tight_layout()
    
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

def preprocess(x, y):
    x = x.astype(np.float32) / 255.0
    x = x.reshape(x.shape[0], -1) # keep the first dimension, unroll the rest
    y = np.eye(10, dtype=np.float32)[y] # the y'th row of a 10x10 identity matrix
    return x, y

def get_dataset():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(base_dir, "data")
    data_path = os.path.join(data_dir, "mnist.npz")

    # load data if already in-situ
    if os.path.exists(data_path):
        with np.load(data_path) as f:
            x_train = f["x_train"]
            y_train = f["y_train"]
            x_test = f["x_test"]
            y_test = f["y_test"]
    else:
        # if no data, use keras.datasets.mnist
        os.makedirs(data_dir, exist_ok=True)

        from keras.datasets import mnist

        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        np.savez_compressed(
            data_path,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
        )

        if VERBOSE:
            print(f"Saved MNIST dataset to {data_path}")

    if DEBUG:
        print("x_train.shape:", x_train.shape, "    y_train.shape:", y_train.shape)
        print("x_test.shape :", x_test.shape, "     y_test.shape:", y_test.shape)
        print("x_train.dtype:", x_train.dtype, "    y_train.dtype:", y_train.dtype)

    return x_train, y_train, x_test, y_test

def get_log_dir():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    log_dir = os.path.join(base_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    return log_dir

def make_run_id():
    return datetime.now().strftime("run_%Y%m%d_%H%M%S")

def append_run_csv(summary: dict):
    log_dir = get_log_dir()
    csv_path = os.path.join(log_dir, "runs_summary.csv")

    row = {
        "timestamp": summary["timestamp"],
        "code_fingerprint": summary["code_fingerprint"],
        "seed": summary["seed"],
        "hidden_layers": summary["hidden_layers"],
        "n_inputs": summary["n_inputs"],
        "n_outputs": summary["n_outputs"],
        "n_param": summary["n_param"],
        "loss_method": summary["loss_method"],
        "epochs": summary["epochs"],
        "batch_size": summary["batch_size"],
        "learning_rate": summary["learning_rate"],
        "train_accuracy": summary["train_accuracy"],
        "test_accuracy": summary["test_accuracy"],
        "generalisation_gap_pp": summary["generalisation_gap_pp"],
        "train_loss": summary["train_loss"],
        "test_loss": summary["test_loss"],
        "training_seconds": summary["training_seconds"],
        "seconds_per_epoch": summary["seconds_per_epoch"],
    }

    # File doesn't exist yet -> simple create
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)
        return

    # Read existing rows/header
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        existing_rows = list(reader)
        existing_fields = reader.fieldnames or []

    new_fields = list(row.keys())

    # Preserve old order, append any genuinely new columns at the end
    merged_fields = existing_fields + [k for k in new_fields if k not in existing_fields]

    # If schema changed, rewrite whole file with expanded header
    if merged_fields != existing_fields:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=merged_fields)
            writer.writeheader()

            for old_row in existing_rows:
                padded_row = {field: old_row.get(field, "") for field in merged_fields}
                writer.writerow(padded_row)

            padded_new_row = {field: row.get(field, "") for field in merged_fields}
            writer.writerow(padded_new_row)
    else:
        # No schema change -> normal append
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=existing_fields)
            writer.writerow(row)

def save_run_artifacts(summary: dict):
    log_dir = get_log_dir()

    runs_dir = os.path.join(log_dir, "runs")
    os.makedirs(runs_dir, exist_ok=True)
    run_dir = os.path.join(runs_dir, summary["run_id"])
    os.makedirs(run_dir, exist_ok=True)

    json_path = os.path.join(run_dir, "summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {k: v for k, v in summary.items() if k not in JSON_BLACKLIST},
            f,
            indent=2
        )

    # Batch plot
    plot_path = os.path.join(run_dir, "loss_plot_batch.png")
    final_plot(summary, "batch_loss", save_path=plot_path, show=False)

    # Epoch plot
    plot_path = os.path.join(run_dir, "loss_plot_epoch.png")
    final_plot(summary, "epoch_loss", save_path=plot_path, show=False)

def get_code_fingerprint():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    src_dir = os.path.join(base_dir, "src")

    files = [
        os.path.join(src_dir, "NNN.py"),
        os.path.join(src_dir, "train.py"),
    ]

    h = hashlib.sha256()
    for path in files:
        if os.path.exists(path):
            h.update(path.encode("utf-8"))
            with open(path, "rb") as f:
                h.update(f.read())

    return h.hexdigest()[:12]

#==============================================================================

if __name__ == "__main__":
    run()

