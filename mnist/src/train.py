# train.py
import sys, os
#sys.path.append(os.path.abspath(".."))
import src.NNN as MyNN
from keras.datasets import mnist
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

VERBOSE = True

def run():
    x_train, y_train, x_test, y_test = get_dataset()
    x_train_new, y_train_new = preprocess(x_train, y_train)
    x_test_new, y_test_new = preprocess(x_test, y_test)

    n_inputs = x_train_new.shape[1]
    n_outputs = y_train_new.shape[1]

    # --- plotting setup ---
    x_data, y_data = [], []

    plt.ion()  # interactive mode ON
    fig, ax = plt.subplots(figsize=(10, 6))
    line, = ax.plot([], [], label="Batch MSE")
    ax.set_xlabel("Global batch")
    ax.set_ylabel("MSE")
    ax.set_yscale("log")
    ax.set_title("Live training loss")
    ax.legend()

    plt.show(block=False)  # open the window and continue

    # define callback INSIDE run so it can see x_data/y_data/fig/ax/line
    def update_plot_loss_batch(data: dict):
        x = int(data["BATCH"])

        # Preferred: if your fit() passes current batch loss directly
        if "BATCH_MSE" in data:
            y = float(data["BATCH_MSE"])
        else:
            # Fallback: use whole stored curve
            y = float(data["LOSS_CURVE_BATCH"][x])

        x_data.append(x)
        y_data.append(y)

        line.set_data(x_data, y_data)
        ax.relim()
        ax.autoscale_view()

        # only redraw every 10 batches to reduce slowdown
        if x % 10 == 0:
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.001)

    model_live = MyNN.Model(n_inputs, [32, 32, n_outputs])

    results = model_live.fit(
        x_train_new,
        y_train_new,
        max_epochs=2,
        batch_size=256,
        learning_rate=0.1,
        callback_func=update_plot_loss_batch,
    )

    # final refresh
    fig.canvas.draw()
    fig.canvas.flush_events()

    plt.ioff()
    plt.show()   # keep final figure open when training ends

    return results

def preprocess(x, y):
    x = x.astype(np.float32) / 255.0
    x = x.reshape(x.shape[0], -1) # keep the first dimension, unroll the rest
    y = np.eye(10)[y] # the y'th row of a 10x10 identity matrix
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

