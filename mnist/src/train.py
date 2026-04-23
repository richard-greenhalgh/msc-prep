# train.py
#import sys, os
#sys.path.append(os.path.abspath(".."))
from dataclasses import dataclass
import time, os
import src.NNN as MyNN
#from keras.datasets import mnist
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

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
        final_plot(results)
    
    return results

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

def final_plot(data: dict):
    y_data = list(data['LOSS_CURVE_BATCH'])
    x_data = list(range(len(y_data)))
    ma_window = min(len(y_data), max(3, int(len(y_data)/50.0)))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x_data, y_data, label="Batch Loss", alpha=0.4)
    
    if len(y_data) >= ma_window:
        y_ma = np.convolve(y_data, np.ones(ma_window)/ma_window, mode='valid')
        x_ma = x_data[ma_window - 1:]  # align lengths
        ax.plot(x_ma, y_ma, label=f"Moving Avg ({ma_window})", linewidth=2)
    
    ax.set_xlabel("Global batch")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    ax.set_title("Training loss by batch")
    ax.legend()

    ax.relim()
    ax.autoscale_view()

    plt.tight_layout()
    plt.show()

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

#==============================================================================

if __name__ == "__main__":
    run()

