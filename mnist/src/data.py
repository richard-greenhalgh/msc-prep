# data.py
# prepare dataset for training, manage logging results
import os, csv, json, hashlib
from datetime import datetime
import numpy as np

from src.vis import final_plot, plot_last_hidden_pca

DEBUG = False
# don't include these (vector) elements in json log
JSON_BLACKLIST = {"batch_loss", "epoch_loss", "val_loss_curve", "val_acc_curve"}

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

        print(f"Saved MNIST dataset to {data_path}")

    if DEBUG:
        print("x_train.shape:", x_train.shape, "    y_train.shape:", y_train.shape)
        print("x_test.shape :", x_test.shape, "     y_test.shape:", y_test.shape)
        print("x_train.dtype:", x_train.dtype, "    y_train.dtype:", y_train.dtype)

    return x_train, y_train, x_test, y_test

class Logger:
    def __init__(self):
        self.dir = self.get_log_dir()
        self.runID = self.make_run_id()
        self.timestamp = datetime.now().isoformat(timespec="seconds")
        self.code_fingerprint = self.get_code_fingerprint()
        self.CSV_COLS = [
            "timestamp", "code_fingerprint", "seed", "hidden_layers",
            "n_inputs", "n_outputs", "n_param",
            "loss_method", "epochs", "epochs_run",
            "batch_size", "optimizer", "learning_rate", "learning_rate_decay",

            "train_accuracy", "val_accuracy", "test_accuracy",
            "generalisation_gap_pp",

            "train_loss", "val_loss", "test_loss",
            "best_val_loss", "best_epoch",

            "training_seconds", "seconds_per_epoch",
            "conv_last_delta", "conv_rel_delta", "conv_rate",
        ]

    def get_log_dir(self):
        base_dir = os.path.dirname(os.path.dirname(__file__))
        log_dir = os.path.join(base_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        return log_dir

    def make_run_id(self):
        return datetime.now().strftime("run_%Y%m%d_%H%M%S")

    def get_CSV_path(self, subdir:str=None, filename:str=None):
        if filename is None: filename = "runs_summary.csv"
        if subdir is None:
            return os.path.join(self.dir, filename)
        else:
            csvdir = os.path.join(self.dir, subdir)
            os.makedirs(csvdir, exist_ok=True)
            return os.path.join(csvdir, filename)

    def append_run_csv(self, summary: dict, csv_path=None):
        if csv_path is None: csv_path = self.get_CSV_path()
        
        row = { k:summary.get(k, "") for k in self.CSV_COLS }

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

    def save_run_artifacts(self, summary: dict, model, train_data=None, test_data=None):
        log_dir = self.dir

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

        # Epoch plot (training data + validation)
        plot_path = os.path.join(run_dir, "loss_plot_epoch.png")
        final_plot(summary, "epoch_loss", save_path=plot_path, show=False)

        # PCA 2D (training data)
        if train_data is not None:
            x, y = train_data
            plot_path = os.path.join(run_dir, "PCA2d_plot_train.png")
            plot_last_hidden_pca(model, x, y, summary, n_samples=5000, save_path=plot_path, show=False)

        # PCA 2D (test data)
        if test_data is not None:
            x, y = test_data
            plot_path = os.path.join(run_dir, "PCA2d_plot_test.png")
            plot_last_hidden_pca(model, x, y, summary, n_samples=5000, save_path=plot_path, show=False)

    def get_code_fingerprint(self):
        base_dir = os.path.dirname(os.path.dirname(__file__))
        src_dir = os.path.join(base_dir, "src")

        files = [
            os.path.join(src_dir, "NNN.py"),
            os.path.join(src_dir, "train.py"),
            os.path.join(src_dir, "data.py"),
            os.path.join(src_dir, "vis.py"),
            os.path.join(src_dir, "sweep.py"),
        ]

        h = hashlib.sha256()
        for path in files:
            if os.path.exists(path):
                h.update(path.encode("utf-8"))
                with open(path, "rb") as f:
                    h.update(f.read())

        return h.hexdigest()[:12]