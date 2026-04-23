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

RNG = None
ACTIVATION_ReLU = 'ReLU'
ACTIVATION_None = None
LOSS_MSE = 'MSE'
LOSS_CROSS_ENTROPY = 'CROSS_ENTROPY'
class Model:
    def __init__(self, n_input, n_outputs: list, output_activation=ACTIVATION_None, loss=LOSS_CROSS_ENTROPY, seed=42):
        assert len(n_outputs) > 0
        self.rng = np.random.default_rng(seed=seed)
        self.layers = []
        in_out = [n_input]
        in_out.extend(n_outputs) # list with (#in, #out1, #out2, ...)
        for i in range(len(n_outputs)):
            # parent of layers[i] is layers[i-1]
            parent = None if i == 0 else self.layers[i - 1]
            activation = output_activation if (i+1)==len(n_outputs) else ACTIVATION_ReLU
            # create the Layer with required no. of inputs/outputs, activation function
            self.layers.append( Layer(in_out[i], in_out[i+1], activation, parent) )
        for i in range(len(self.layers)-1):
            # child of layers[i] is layers[i+1]
            self.layers[i].childLayer = self.layers[i+1]
        self.inputLayer = self.layers[0]
        self.outputLayer = self.layers[-1]
        self.epoch_index = None
        self.global_batch = None
        self.loss_calc = loss
        self.loss_curve_epoch = None
        self.BPE = None # batches per epoch
        self.loss_curve_batch = None # loss by batch

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
        for L in self.layers: L.derivs.fill(0.0)
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
                for L in self.layers: L.derivs /= batch_sample
                self.update_params(learning_rate)
                # track loss by batch
                batch_loss_avg = batch_loss_sum / batch_sample
                if self.loss_calc == LOSS_MSE: batch_loss_avg /= y_target[0].size
                self._store_loss_batch(batch_loss_avg, batch_count)
                # callback for live updates?
                if callback_func is not None:
                    callback_func(self._make_fit_results(y_pred, self.global_batch, batch_loss_avg))
                # reset batch info, start a new batch
                for L in self.layers: L.derivs.fill(0.0)
                batch_sample, batch_loss_sum = 0, 0.0
                batch_count += 1
                self.global_batch += 1
                
        self.loss_avg = self.loss_sum / len(y_target) # average loss for epoch
        if self.loss_calc == LOSS_MSE: self.loss_avg /= y_target[0].size

        if batch_sample > 0: # partial batch remaining
            for L in self.layers: L.derivs /= batch_sample
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
            'NPARAM': self.countParameters(),
            'LOSS_METHOD': self.loss_calc
        }

    def _store_loss_batch(self, loss_avg, batch_index):
        self.loss_curve_batch[self.epoch_index * self.BPE + batch_index] = loss_avg

    # fit(): entry point for fitting parameters
    # x_in: training data
    # y_target: actuals corresponding to training data
    # max_epochs: maximum number of passes through the training data
    # learning_rate: initial learning rate
    # batch_size: split each epoch into batches with X samples
    # show_progress: True/Fale
    def fit(self, x_in: np.ndarray, y_target: np.ndarray, max_epochs=100, learning_rate=0.01, batch_size=256, show_progress=False, callback_func=None):
        self.loss_curve_epoch = np.zeros(max_epochs)
        self.BPE = ceil(len(x_in)/batch_size) # batches per epoch
        self.loss_curve_batch = np.zeros(max_epochs * self.BPE)
        self.global_batch = 0
        for i in range(max_epochs):
            self.epoch_index = i
            result = self.epoch(x_in, y_target, learning_rate, batch_size, callback_func)
            ypred = result['Y_PRED']
            self.loss_curve_epoch[i] = self.loss_avg
            if show_progress and i % 10 == 0:
                print(f"Epoch: {i:05d} --- ", f"Loss: {self.loss_avg:.5f} --- ", [f"{y[0]:.3f}" for y in ypred])
        # get ypred based on final parameters
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
        for L in self.layers:
            L.params -= learning_rate * L.derivs
    
    def countParameters(self):
        return sum(L.params.size for L in self.layers)

class Layer:
    """ a layer of neurons """

    def __init__(self, n_input, n_output, activation_func=ACTIVATION_ReLU, parentLayer=None, childLayer=None, rng=None):
        self.rng = rng if rng is not None else np.random.default_rng()
        self.params = self.params = np.zeros((n_output, n_input + 1), dtype=np.float32)
        # initialize weights [W<-- b]
        self.params[:, :-1] = self.rng.standard_normal((n_output, n_input), dtype=np.float32)
        # initialize biases as zero [W b<--]
        self.params[:, -1] = 0.0
        # scale weights according to activation type and size of layer
        if activation_func == ACTIVATION_ReLU:
            # He initialization
            # - want activations to have ~constant variance across layers
            # - scale by 1/n_input, but ReLU zeros out ~50%, so use 2/n_input
            scale = np.sqrt(2.0 / n_input)
        else:
            scale = np.sqrt(1.0 / n_input)
        self.params[:, :-1] *= scale
        
        self.derivs = np.zeros((n_output, n_input+1), dtype=np.float32)
        self.mx_weights = self.params[:, :-1]
        self.vec_biases = self.params[:, -1]
        self.vec_input = np.zeros(n_input, dtype=np.float32) # store last used input to __call__()
        self.vec_pre_activations = np.zeros(n_output, dtype=np.float32)
        self.vec_activations = np.zeros(n_output, dtype=np.float32)
        self.AF = activation_func
        self.parentLayer = parentLayer
        self.childLayer = childLayer

        # allocate buffers/views for use in backprop()
        self._local_derivs = np.zeros_like(self.derivs)
        self._local_derivs_dX_dw = self._local_derivs[:, :-1]
        self._local_derivs_dX_db = self._local_derivs[:, -1]

        self._final_derivs_dL_dp = np.zeros_like(self.derivs)
        self._final_derivs_dL_dw = self._final_derivs_dL_dp[:, :-1]
        self._final_derivs_dL_db = self._final_derivs_dL_dp[:, -1]
        
        self._upstream_dL_dz = np.zeros(n_output, dtype=np.float32)
        self._parent_deriv = np.zeros(n_input, dtype=np.float32)

    def __call__(self, vec_x):
        # - take array of inputs
        # - apply weights/biases/activation function
        # - return array of outputs
        self.vec_input[:] = vec_x
        
        np.dot(self.mx_weights, vec_x, out=self.vec_pre_activations) # z = Wx + b
        self.vec_pre_activations += self.vec_biases
        if self.AF == ACTIVATION_ReLU:
            np.maximum(self.vec_pre_activations, 0, out=self.vec_activations)
        else:
            # no activation function
            self.vec_activations[:] = self.vec_pre_activations

        if self.childLayer is None:
            return self.vec_activations
        else:
            return self.childLayer(self.vec_activations)
    
    def backprop(self, upstream_deriv):
        p = self.parentLayer
        # upstream_deriv: 
        # - is a vector, length equal to no. of outputs of THIS layer
        # - (equal to the number of ROWS in the weights matrix for THIS layer)
        # - contains derivatives of the loss wrt THIS layer's outputs (activations)
        # - (dL/da_1, dL/da_2, ..., dL/da_n)

        # (oversimplified!) 1-neuron-per-layer example looks like:
        # upstream derivative = upstream(layer) = W(layer+1) * W(layer+2) * ... * W(layer n)
        # dL/dW(layer) = [local deriv = X(layer)] * upstream(layer)
        # dL/db(layer) = [local deriv = 1.0] * upstream(layer)

        # step 0. transform upstream_derivs from dL/da to dL/dz (wrt pre-activations)
        # - dL/dz = dL/da * da/dz
        # - da/dz (ReLU): 0 (if z <= 0), 1 (otherwise)
        # - da/dz (None): 1
        if self.AF == ACTIVATION_ReLU:
            self._upstream_dL_dz[:] = upstream_deriv * (self.vec_pre_activations > 0)
        else:
            self._upstream_dL_dz[:] = upstream_deriv

        # !!! remember [W b] structure !!! (use views)
        # step 1. calculate, local_derivs for weights matrix
        # each column populated with input x: x_1, x_2, ..., x_n
        # (No longer needed, just need _final_derivs..., leaving for posterity)
        self._local_derivs_dX_dw[:] = self.vec_input

        # step 2. calculate, local_derivs for bias vector (final column)
        # (No longer needed, just need _final_derivs..., leaving for posterity)
        self._local_derivs_dX_db.fill(1.0)

        # step 3. "multiply" by the upstream derivs to get final derivs wrt loss
        # - we need to scale each row of the local derivs by the corresponding element of the upstream derivs
        # - this operation is the outer product
        # - for biases, the local deriv is 1, so the final column equals the upstream derivs
        self._final_derivs_dL_dw[:] = np.outer(self._upstream_dL_dz, self.vec_input)
        self._final_derivs_dL_db[:] = self._upstream_dL_dz

        # step 4. ACCUMULATE derivs from step 3 into self.derivs
        self.derivs += self._final_derivs_dL_dp

        # step 5. prepare parent_deriv to continue backprop to parent
        # - becomes the new upstream_deriv for the next layer in backpropagation
        # - is a vector, length equal to no. of inputs of THIS layer, OR
        # - is a vector, length equal to no. of outputs of PARENT layer
        # - contains derivatives of the loss wrt PARENT layer's outputs (activations), OR
        # - contains derivatives of the loss wrt THIS layer's inputs
        # - (dL/da_1, dL/da_2, ..., dL/da_n)
        if p is not None: 
            self._parent_deriv.fill(0)
            # calculate the weighted sum over the rows of the weights matrix, with each row scaled by dL/dz
            # ( dL/da(in) = dz(out)/da(in) * dL/dz(out) )
            # dz(out)/da(in): derivatives wrt inputs leaves the weights (z = a1w1 + a2w2 + ... + b)
            # dL/dz(out): computed above in self._upstream_dL_dz
            """ (explicit version for reference)
            rows, cols = self.mx_weights.shape
            for i in range(rows):
                for j in range(cols):
                self._parent_deriv[j] += self.mx_weights[i][j] * self._upstream_dL_dz[i]
            """
            self._parent_deriv[:] = self.mx_weights.T @ self._upstream_dL_dz
            p.backprop(self._parent_deriv) # if has parent, keep backprop'ing


def test_Layer():
    l = Layer(3, 2)
    x = np.array([1.0, 2.0, 3.0])
    l(x)
    print(l.mx_weights)
    print(l.vec_biases)
    print(l.vec_activations)

def test_Model():
    M = Model(3, [4, 4, 1])

    xs = np.array([
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0]
    ])

    ys = np.array([[1.0], [-1.0], [-1.0], [1.0]])

    #for i in range(100):
    #    ypred = M.epoch(xs, ys, learning_rate=0.094)
    #    if i % 10 == 0:
    #        print(f"{i:05d} --- ", f"MSE: {M.MSE:.5f} --- ", [f"{y[0]:.3f}" for y in ypred])

    M.fit(xs, ys, learning_rate=0.094)