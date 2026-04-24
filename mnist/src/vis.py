# vis.py
# visualisation / plotting utilities

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

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
        f"optimizer={summary['optimizer']} | "
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