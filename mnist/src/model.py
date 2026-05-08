# NNN.py
# Neural Network (in) Numpy
# from-scratch NN implementation for learning purposes

# TODO:
# create separate doc explaining the maths?
# - solidify understanding of Layer.backprop()
#
# add callback capability to allow dynamic charts
# - Model.fit(..., callback=callback_function):
# -     ...
# -     # each batch:
# -     callback({"epoch": epoch, "batch": batch_index, ...})
#
# infer learning rate automatically, then decay over time?
# - run one epoch, increase learning rate each batc, plot loss as function of learning rate
#
# softmax / crossentropy loss?
# - implement stable softmax for output logits --> DONE
# - replace MSE with cross-entropy classification loss --> DONE
# - use simplified output gradient softmax(logits) - target --> DONE
# - compare training speed and validation accuracy vs current MSE version
# 
# add logging capability to keep track of experimentation
# - CSV file with columns (???)

import numpy as np

from math import ceil
from src.layer import Layer, LayerDense, LayerConv2D, LayerFlatten
from src.layer import LAYER_DENSE, LAYER_CONV2D, LAYER_FLATTEN
from src.layer import ACTIVATION_ReLU, ACTIVATION_None

LOSS_MSE = 'MSE'
LOSS_CROSS_ENTROPY = 'CROSS_ENTROPY'
OPTIMIZER_SGD = 'SGD'
OPTIMIZER_ADAM = 'ADAM'

class Model:
    
    def __init__(self, input_shape, arch: list[Layer], output_activation=ACTIVATION_None, 
                 loss=LOSS_CROSS_ENTROPY, seed=42, optimizer=OPTIMIZER_SGD):
        assert len(input_shape) > 0
        self.rng = np.random.default_rng(seed=seed)
        self.optimizer = optimizer
        self.optimizer_step = 0
        if self.optimizer == OPTIMIZER_ADAM: self.setAdamParameters()
        self.input_shape = input_shape
        self.layers = []

        # pass layer arch(itecture) data structure and add layers accordingly
        for L in arch: self._add_layer(L)

        # set up variables/buffers for further use
        self.epoch_index = None
        self.global_batch = None
        self.loss_calc = loss
        self.loss_curve_epoch = None      # loss curve for training data
        self.val_loss_curve_epoch = None  # loss curve for validation data
        self.val_acc_curve_epoch = None   # acc. curve for validation data
        self.BPE = None # batches per epoch
        self.loss_curve_batch = None # loss by batch
        self.best_params = None
        self.best_adam_m = None
        self.best_adam_v = None
        self.best_optimizer_step = None

    def _add_layer(self, L: Layer):
        isFirst = len(self.layers) == 0
        parent = None if isFirst else self.layers[-1]

        if L.type == LAYER_DENSE:
            if not isFirst and len(parent.output_shape) != 1:
                raise Exception("LAYER_DENSE expects 1D input, use LAYER_FLATTEN if required.")
            
            inputs = self.input_shape[0] if isFirst else parent.output_shape[0]
            self.layers.append( LayerDense(inputs, L.outputs, L.activation, parent, rng=self.rng) )
        elif L.type == LAYER_FLATTEN:
            assert parent is not None
            self.layers.append( LayerFlatten(input_shape=parent.output_shape, parentLayer=parent) )
        elif L.type == LAYER_CONV2D:
            self.layers.append( LayerConv2D(
                input_shape     = self.input_shape if isFirst else parent.output_shape,
                n_filters       = L.outputs,
                kernel_size     = L.CNN_kernel_size,
                stride          = L.CNN_stride,
                mode            = L.CNN_mode,
                activation_func = L.activation,
                parentLayer     = parent,
                childLayer      = None,
                rng             = self.rng
            ) )
        else:
            raise Exception("Unknown layer type: " + L.type)
        
        if not isFirst:
            # update child of previous layer, set most recent as output layer
            parent.childLayer = self.layers[-1]
            self.outputLayer = self.layers[-1]
        else:
            # isFirst = True, set as inputLayer
            self.inputLayer = self.layers[0]
    
    def forward_pass(self, x: np.ndarray):
        return self.inputLayer(x) # Layer.__call__() recurses through layers

    def backward_pass(self, deriv0):
        # back prop from last layer
        self.outputLayer.backprop(deriv0)

    # epoch(): a training pass through all training data
    # - split into batches via batch_size
    def epoch(self, x_in: np.ndarray, y_target: np.ndarray, learning_rate=0.01, batch_size=256, 
              callback_func=None, shuffle=True):
        assert x_in.shape[0] == y_target.shape[0] # check number of X matches Y
        
        if shuffle:
            perm = self.rng.permutation(len(x_in))
            x_in = x_in[perm]
            y_target = y_target[perm]
        
        y_pred = np.zeros(y_target.shape, dtype=np.float32)
        y_softmax = np.zeros(y_target.shape, dtype=np.float32)
        self.loss_sum, self.loss_avg = 0.0, 0.0

        # reset all derivs
        for L in self.layers: L.reset_derivs()
        batch_sample = 0  # the counter for samples in this batch
        batch_count = 0   # the counter for no. of batches
        batch_loss_sum = 0.0    # the loss for this batch

        # loop through samples in training data
        for i in range(len(x_in)):
            y_pred[i] = self.forward_pass(x_in[i]) # raw logits
            
            # Handle different loss methodology
            if self.loss_calc == LOSS_CROSS_ENTROPY:
                # softmax reminder: exp(out) / sum( exp(all) )
                # stable version: deduct highest logit from all logits
                np.exp(y_pred[i]-np.max(y_pred[i]), out=y_softmax[i])
                y_softmax[i] /= np.sum(y_softmax[i])

                # use CE loss: -log(p_correct)
                # !!! NOTE THIS IS ONLY TRUE FOR ONE-HOT / CLASSIFICATION !!!
                p_correct = y_softmax[i][ np.argmax(y_target[i]) ]
                sample_loss = -np.log( np.clip(p_correct, 1e-7, 1.0 - 1e-7) )
                self.loss_sum += sample_loss # running total of CE loss
                batch_loss_sum += sample_loss
                self.backward_pass(y_softmax[i] - y_target[i]) # probs - y_target

            elif self.loss_calc == LOSS_MSE:
                resid = y_pred[i] - y_target[i] # vector of residuals
                sample_loss = np.sum(resid**2) # sum of squared residuals
                self.loss_sum += sample_loss # running total of SSE
                batch_loss_sum += sample_loss
                self.backward_pass(2.0 * resid / y_target.shape[1]) # deriv of mean squared residuals
            else:
                raise Exception("Bad loss method: " + self.loss_calc)
            
            batch_sample += 1
            if batch_sample == batch_size: # update params for this batch   
                # calculate AVERAGE gradients (after acumulating sum of gradients)
                for L in self.layers: L.scale_derivs(batch_sample)
                self.update_params(learning_rate)
                # track loss by batch
                batch_loss_avg = batch_loss_sum / batch_sample
                if self.loss_calc == LOSS_MSE: batch_loss_avg /= y_target[0].size
                self._store_loss_batch(batch_loss_avg, batch_count)
                # callback for live updates?
                if callback_func is not None:
                    callback_func(self._make_fit_results(y_pred, self.global_batch, batch_loss_avg))
                # reset batch info, start a new batch
                for L in self.layers: L.reset_derivs()
                batch_sample, batch_loss_sum = 0, 0.0
                batch_count += 1
                self.global_batch += 1
                
        self.loss_avg = self.loss_sum / len(y_target) # average loss for epoch
        if self.loss_calc == LOSS_MSE: self.loss_avg /= y_target[0].size

        if batch_sample > 0: # partial batch remaining
            for L in self.layers: L.scale_derivs(batch_sample)
            self.update_params(learning_rate)
            batch_loss_avg = batch_loss_sum / batch_sample
            if self.loss_calc == LOSS_MSE: batch_loss_avg /= y_target[0].size
            self._store_loss_batch(batch_loss_avg, batch_count)

        return self._make_fit_results(y_pred, batch_count, batch_loss_avg)
    
    def _make_fit_results(self, y_pred, batch, loss_avg):
        return {
            'Y_PRED': y_pred,
            'BATCH': batch,
            'BATCH_LOSS': loss_avg,
            'EPOCH': self.epoch_index,
            'LOSS_CURVE_BATCH': self.loss_curve_batch,
            'LOSS_CURVE_EPOCH': self.loss_curve_epoch,
            'VAL_LOSS_CURVE_EPOCH': self.val_loss_curve_epoch,
            'VAL_ACC_CURVE_EPOCH': self.val_acc_curve_epoch,
            'BEST_VAL_LOSS': None, # final result, set in Model.fit()
            'BEST_EPOCH': None,    # final result, set in Model.fit()
            'EPOCHS_RUN': None,    # final result, set in Model.fit()
            'NPARAM': self.countParameters(),
            'LOSS_METHOD': self.loss_calc
        }

    def _store_loss_batch(self, loss_avg, batch_index):
        self.loss_curve_batch[self.epoch_index * self.BPE + batch_index] = loss_avg

    def _store_best_params(self):
        self.best_params = [L.params.copy() for L in self.layers]

        if self.optimizer == OPTIMIZER_ADAM:
            self.best_adam_m = [L._adam_m.copy() for L in self.layers]
            self.best_adam_v = [L._adam_v.copy() for L in self.layers]
            self.best_optimizer_step = self.optimizer_step


    def _restore_best_params(self):
        for i, L in enumerate(self.layers):
            L.params[:] = self.best_params[i]

            if self.optimizer == OPTIMIZER_ADAM:
                L._adam_m[:] = self.best_adam_m[i]
                L._adam_v[:] = self.best_adam_v[i]
                self.optimizer_step = self.best_optimizer_step

    # fit(): entry point for fitting parameters
    # x_in: training data
    # y_target: actuals corresponding to training data
    # max_epochs: maximum number of passes through the training data
    # learning_rate: initial learning rate
    # batch_size: split each epoch into batches with X samples
    # show_progress: True/Fale
    def fit(self, x_in: np.ndarray, y_target: np.ndarray, max_epochs=100, learning_rate=0.01, learning_rate_decay=1.00,
            batch_size=256, callback_func=None, validation_split=0.0, early_stop=False, early_patience=5,
            restore_best_weights=True):
        # create validation data split?
        if validation_split > 0.0:
            n = len(x_in)
            perm = self.rng.permutation(n)
            split = int((1.0 - validation_split) * n)

            train_idx = perm[:split]
            val_idx = perm[split:]

            x_val = x_in[val_idx]
            y_val = y_target[val_idx]

            x_train = x_in[train_idx]
            y_train = y_target[train_idx]
        else:
            x_train = x_in
            y_train = y_target
            x_val = None
            y_val = None
        
        self.loss_curve_epoch = np.full(max_epochs, np.nan)
        self.val_loss_curve_epoch = np.full(max_epochs, np.nan)
        self.val_acc_curve_epoch = np.full(max_epochs, np.nan)
        self.BPE = ceil(len(x_train)/batch_size) # batches per epoch
        self.loss_curve_batch = np.full(max_epochs * self.BPE, np.nan)
        self.global_batch = 0

        best_val_loss = np.inf
        best_epoch = -1
        epochs_without_improvement = 0

        for i in range(max_epochs):
            self.epoch_index = i
            result = self.epoch(x_train, y_train, learning_rate, batch_size, callback_func)
            self.loss_curve_epoch[i] = self.loss_avg

            # use validation data, check for early stop
            if x_val is not None:
                val_loss = self.calcLoss(x_val, y_val)

                self.val_loss_curve_epoch[i] = val_loss
                self.val_acc_curve_epoch[i] = self.accuracy(x_val, y_val)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = i
                    epochs_without_improvement = 0
                    if restore_best_weights:
                        self._store_best_params()
                else:
                    epochs_without_improvement += 1

                if early_stop and epochs_without_improvement >= early_patience:
                    break
            
            learning_rate *= learning_rate_decay
        # <-- end of loop through epochs

        # restore best weights based on validation set (if applicable)
        if restore_best_weights and x_val is not None and best_epoch >= 0:
            self._restore_best_params()
        
        # truncate curves if needed
        epochs_run = i + 1
        self.loss_curve_epoch = self.loss_curve_epoch[:epochs_run]
        self.val_loss_curve_epoch = self.val_loss_curve_epoch[:epochs_run]
        self.val_acc_curve_epoch = self.val_acc_curve_epoch[:epochs_run]
        self.loss_curve_batch = self.loss_curve_batch[:epochs_run * self.BPE]
        
        # capture final results
        result['LOSS_CURVE_BATCH'] = self.loss_curve_batch
        result['LOSS_CURVE_EPOCH'] = self.loss_curve_epoch
        result['VAL_LOSS_CURVE_EPOCH'] = self.val_loss_curve_epoch
        result['VAL_ACC_CURVE_EPOCH'] = self.val_acc_curve_epoch
        result['BEST_VAL_LOSS'] = best_val_loss
        result['BEST_EPOCH'] = best_epoch
        result['EPOCHS_RUN'] = epochs_run

        return result
    
    # apply(): use the model for inference (run the model to get predicted y)
    # rawLogit=False: return softmax probabilities
    # rawLogit=True : return raw logits
    def apply(self, x_in: np.ndarray, rawLogit=False):
        y_pred = np.zeros((x_in.shape[0], self.outputLayer.vec_activations.shape[0]), dtype=np.float32)
        for i in range(len(x_in)):
            y_pred[i] = self.forward_pass(x_in[i])
            if not rawLogit:
                np.exp(y_pred[i]-np.max(y_pred[i]), out=y_pred[i])
                y_pred[i] /= np.sum(y_pred[i])
        return y_pred
    
    def calcMSE(self, x_in: np.ndarray, y_true: np.ndarray) -> float:
        y_pred = self.apply(x_in, rawLogit=True)
        return np.mean((y_pred - y_true) ** 2)
    
    def calcCELoss(self, x_in: np.ndarray, y_true: np.ndarray) -> float:
        y_pred = self.apply(x_in) # softmax probabilities
        class_idx = np.argmax(y_true, axis=1) # vector of true classes
        rows = np.arange(len(y_pred)) # rows for paired indexing
        p_correct = y_pred[rows, class_idx] # vector of the model prob. for true classes
        sample_loss = -np.log(np.clip(p_correct, 1e-7, 1.0 - 1e-7))
        return np.mean(sample_loss)

    def calcLoss(self, x_in: np.ndarray, y_true: np.ndarray) -> float:
        if self.loss_calc == LOSS_MSE:
            return self.calcMSE(x_in, y_true)
        elif self.loss_calc == LOSS_CROSS_ENTROPY:
            return self.calcCELoss(x_in, y_true)
        else:
            raise Exception("Bad loss method: " + self.loss_calc)

    def accuracy(self, x_in: np.ndarray, y_true: np.ndarray) -> float:
        pred_class = np.argmax(self.apply(x_in), axis=1)
        true_class = np.argmax(y_true, axis=1)
        return np.mean(pred_class == true_class)

    # update weights/biases during model training
    def update_params(self, learning_rate=0.01):
        # update parameters
        self.optimizer_step += 1
        for L in self.layers:
            if self.optimizer == OPTIMIZER_ADAM:
                for P, dP, M, V in zip(L.params, L.derivs, L.adam_m, L.adam_v):
                    M[:] = self.ADAM_beta1 * M + (1 - self.ADAM_beta1) * dP
                    V[:] = self.ADAM_beta2 * V + (1 - self.ADAM_beta2) * (dP ** 2)

                    # bias correction
                    t = np.int32(self.optimizer_step)
                    m_hat = M / (1 - self.ADAM_beta1**t)
                    v_hat = V / (1 - self.ADAM_beta2**t)

                    P -= learning_rate * m_hat / (np.sqrt(v_hat) + self.ADAM_EPS)
            elif self.optimizer == OPTIMIZER_SGD:
                for P, dP in zip(L.params, L.derivs):
                    P -= learning_rate * dP
            else:
                for P, dP in zip(L.params, L.derivs):
                    P -= learning_rate * dP
    
    def countParameters(self):
        return sum(L.params.size for L in self.layers)
    
    def setAdamParameters(self, beta1=0.9, beta2=0.999, eps=1e-8):
        self.ADAM_beta1 = np.float32(beta1)
        self.ADAM_beta2 = np.float32(beta2)
        self.ADAM_EPS = np.float32(eps)
