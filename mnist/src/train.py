# train.py
#import sys, os
#sys.path.append(os.path.abspath(".."))
from dataclasses import dataclass
import time
import src.NNN as MyNN
from keras.datasets import mnist
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

VERBOSE = True

@dataclass
class TrainConfig:
    hidden_layers: list[int]
    max_epochs: int = 2
    batch_size: int = 256
    learning_rate: float = 0.1
    live_plot: bool = False
    live_update_freq: int = 10   # redraw chart every X batches

def run(cfg: TrainConfig = None):
    # !!! SET HIDDEN LAYERS HERE !!!
    if cfg is None:
        cfg = TrainConfig([32, 32])
    x_train, y_train, x_test, y_test = get_dataset()
    x_train_new, y_train_new = preprocess(x_train, y_train)
    x_test_new, y_test_new = preprocess(x_test, y_test)

    n_inputs = x_train_new.shape[1]
    n_outputs = y_train_new.shape[1]
    model = MyNN.Model(n_inputs, cfg.hidden_layers+[n_outputs])

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
    
    mse_train = model.calcMSE(x_train_new, y_train_new)
    mse_test = model.calcMSE(x_test_new, y_test_new)

    # display resulting accuracy, draw final plot?
    if cfg.live_plot and finish is not None: finish()
    print(" --- ")
    print(f"Model architecture (layers)        :", f"inputs({n_inputs}),", f"hidden{cfg.hidden_layers},", f"outputs({n_outputs})")
    print(f"Model parameter count              : {results['NPARAM']}")
    print(" --- ")
    print(f"Accuracy on in-sample training data: {acc_train:.3f}%" )
    print(f"Accuracy on out-of-sample test data: {acc_test:.3f}%" )
    print(" --- ")
    print(f"Generalisation gap                 : {acc_train - acc_test:.3f} pp  (high positive value may indicate overfitting)")
    print(" --- ")
    print(f"MSE for in-sample training data    : {mse_train:.6f}")
    print(f"MSE for out-of-sample test data    : {mse_test:.6f}")
    print(" --- ")
    print(f"Model training and inference time:")
    print(f"    Training            : {elapsed:.3f}s  ({(elapsed/cfg.max_epochs):.3f}s per epoch)")
    print(f"    Infer (training set): {elapsed2:.3f}s  ({infer_rate:.3f}s per 1000 samples)")
    return results

def make_live_plot_callback(update_every=10):
    x_data, y_data = [], []

    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))
    line, = ax.plot([], [], label="Batch Loss")
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
            line.set_data(x_data, y_data)
            ax.relim()
            ax.autoscale_view()
            fig.canvas.draw()
            fig.canvas.flush_events()

    def final_plot():
        if len(x_data) == 0:
            return

        line.set_data(x_data, y_data)
        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.ioff()
        plt.show()

    return callback, final_plot

def preprocess(x, y):
    x = x.astype(np.float32) / 255.0
    x = x.reshape(x.shape[0], -1) # keep the first dimension, unroll the rest
    y = np.eye(10, dtype=np.float32)[y] # the y'th row of a 10x10 identity matrix
    return x, y

def get_dataset():
    # load data using keras
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if VERBOSE:
        # check the shape of arrays
        print('x_train.shape:', x_train.shape, '    y_train.shape:', y_train.shape)
        print('x_test.shape :', x_test.shape, '     y_test.shape:', y_test.shape)
        print('x_train.dtype:', x_train.dtype, '    y_train.dtype:', y_train.dtype)
    
    return x_train, y_train, x_test, y_test

#==============================================================================

if __name__ == "__main__":
    run()

