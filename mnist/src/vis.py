# vis.py
# visualisation / plotting utilities

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

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

def final_plot(summary: dict, curve_name: str = 'epoch_loss', save_path=None, show=True):
    is_batch = curve_name.lower().find('batch') != -1
    is_epoch = not is_batch
    
    y_data = list(summary[curve_name])
    x_data = list(range(len(y_data)))
    ma_window = min(len(y_data), max(3, int(len(y_data)/50.0)))
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if is_batch:
        ax.plot(x_data, y_data, label="Batch Loss", alpha=0.4)
    
        if len(y_data) >= ma_window and is_batch:
            y_ma = np.convolve(y_data, np.ones(ma_window)/ma_window, mode='valid')
            x_ma = x_data[ma_window - 1:]  # align lengths
            ax.plot(x_ma, y_ma, label=f"Moving Avg ({ma_window})", linewidth=2)
    
    else:
        # training loss
        ax.plot(x_data, y_data, label="Train Loss", linewidth=2)

        # validation loss (if applicable)
        val_curve = summary.get("val_loss_curve", None)
        if val_curve is not None:
            ax.plot(x_data, val_curve, label="Val Loss", linewidth=2)

        # mark best epoch
        best_epoch = summary.get("best_epoch", None)
        if best_epoch is not None:
            ax.axvline(best_epoch, linestyle='--', alpha=0.5, label=f"Best epoch ({best_epoch})")
        
        # secondary axis for accuracy
        val_acc_curve = summary.get("val_acc_curve", None)
        if val_acc_curve is not None:
            ax2 = ax.twinx()

            val_acc_pct = [v * 100 for v in val_acc_curve]

            ax2.plot(
                x_data,
                val_acc_pct,
                linestyle="--",
                linewidth=2,
                color="green",
                label="Val Accuracy (%)",
            )

            ax2.set_ylabel("Accuracy (%)")
            
            ymin = min(val_acc_pct)
            ymax = max(val_acc_pct)
            margin = (ymax - ymin) * 0.2
            ax2.set_ylim(max(0, ymin - margin), min(100, ymax + margin))

            # combine legends from both axes
            lines_1, labels_1 = ax.get_legend_handles_labels()
            lines_2, labels_2 = ax2.get_legend_handles_labels()
            ax.legend(lines_1 + lines_2, labels_1 + labels_2)
    
    ax.set_xlabel("Epoch" if is_epoch else "Batch")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    
    title = (
        f"MNIST loss | hidden={summary['hidden_layers']} | "
        f"loss={summary['loss_method']} | "
        f"optimizer={summary['optimizer']} | "
        f"train_acc={summary['train_accuracy']:.2f}% | "
        f"test_acc={summary['test_accuracy']:.2f}%"
    )
    ax.set_title(title)

    ax.relim()
    ax.autoscale_view()

    plt.tight_layout()
    
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show(block=False)
    else:
        plt.close(fig)

def plot_last_hidden_pca(model, x, y, summary, n_samples=5000, save_path=None, show=True):
    # See longer comment at end of file for more information on Principal Component Analysis
    perm = model.rng.permutation(len(x))[:n_samples]
    x = x[perm]
    y = y[perm]

    A = [] # store sample of activations in final hidden layer
    labels = [] # store true digit for each sample
    preds = [] # store model predictions

    # collect activations from the final hidden layer
    for i in range(len(x)):
        preds.append( np.argmax(model.forward_pass(x[i])) )
        A.append(model.layers[-2].vec_activations.copy())
        labels.append( np.argmax(y[i]) )
    A = np.array(A)
    labels = np.array(labels)
    preds = np.array(preds)
    correct = preds == labels
    incorrect = ~correct
    A_2d = PCA(n_components=2).fit_transform(A)

    x_data = A_2d[:,0]
    y_data = A_2d[:,1]

    fig, ax = plt.subplots(figsize=(10, 6))

    # correct predictions
    scatter = ax.scatter(
        x_data[correct],
        y_data[correct],
        c=labels[correct],
        cmap='tab10',
        alpha=0.6,
        s=10
    )

    # incorrect predictions (highlighted)
    ax.scatter(
        x_data[incorrect],
        y_data[incorrect],
        c=labels[incorrect],
        cmap='tab10',
        edgecolors='black',
        linewidths=0.8,
        s=30,
        label='Incorrect'
    )
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Digit")
    
    ax.set_xlabel("PCA[1]")
    ax.set_ylabel("PCA[2]")
    #ax.set_yscale("log")

    title = (
        f"PCA (2D) | hidden={summary['hidden_layers']} | "
        f"loss={summary['loss_method']} | "
        f"optimizer={summary['optimizer']} | "
        f"train_acc={summary['train_accuracy']:.2f}% | "
        f"test_acc={summary['test_accuracy']:.2f}%"
    )
    ax.set_title(title)
    plt.tight_layout()
    
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show(block=False)
    else:
        plt.close(fig)

"""
# === PCA via eigen-decomposition example calc ===
# X: (n_samples, n_features)
# rows = data points (e.g. MNIST activations)
# cols = features (neurons)

# 1. Mean-center the data
X_centered = X - X.mean(axis=0)

# 2. Compute covariance matrix (or X^T X equivalent)
C = X_centered.T @ X_centered   # shape: (n_features, n_features)

# 3. Eigen-decomposition
eigvals, eigvecs = np.linalg.eigh(C)

# 4. Sort by descending eigenvalue (largest variance first)
idx = np.argsort(eigvals)[::-1]
eigvals = eigvals[idx]
eigvecs = eigvecs[:, idx]

# 5. Select top k principal components
W = eigvecs[:, :k]   # projection matrix (n_features × k)

# 6. Project data into lower dimension
Z = X_centered @ W   # (n_samples × k)

# Z is your PCA-reduced data (e.g. 2D for plotting)

# Notes:
# - eigvecs = principal directions (columns)
# - eigvals = variance explained along each direction
# - sorted so eigvals[0] is the most important direction
# - Z gives coordinates in the new feature space

# In practice:
# sklearn / numpy PCA uses SVD (X = U Σ V^T) for numerical stability,
# but produces the same principal components as this method.



# === PCA via SVD (practical / numerically stable) ===
# X: (n_samples, n_features)

# 1. Mean-center the data
X_centered = X - X.mean(axis=0)

# 2. Compute SVD
# X = U Σ V^T
U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

# 3. Principal components (directions)
# rows of Vt = principal directions
# so columns of V = Vt.T
V = Vt.T

# 4. Select top k components
W = V[:, :k]   # (n_features × k)

# 5. Project data
Z = X_centered @ W   # (n_samples × k)

"""