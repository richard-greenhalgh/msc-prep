# CNN.py
# Explore implementation of a Convolutional NN

# TODO LIST:
# -- amend preprocessing to handle CNN (i.e. do not flatten)
# -- implement single 3x3 filter and apply to a MNIST image, view result
# -- build out a CNN layer object / forward pass
# -- build out backprop for CNN
# -- consider padding/stride?

from math import floor
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from src.data import get_dataset, preprocess
from src.vis import draw_grid, draw_1
from src.NNN import ACTIVATION_None, ACTIVATION_ReLU

def test_shapes():
    x_raw, y_raw, x_raw_test, y_raw_test = get_dataset() # get train/test data
    x_train, y_train = preprocess(x_raw, y_raw, flatten=False)

    L = LayerConv2D((1, 28, 28), n_filters=8, kernel_size=(3, 3), stride=1)
    out = L(x_train[0])
    print(out.shape)
    L = LayerConv2D((1, 28, 28), n_filters=8, kernel_size=(3, 3), stride=2)
    out = L(x_train[0])
    print(out.shape)

def test_filter():
    x_raw, y_raw, x_raw_test, y_raw_test = get_dataset() # get train/test data
    x_train, y_train = preprocess(x_raw, y_raw, flatten=False)

    
    # create a "vertical line" filter as example
    F_vertical = np.array([
        [-1.0, 0.0, 1.0],
        [-1.0, 0.0, 1.0],
        [-1.0, 0.0, 1.0]
    ], dtype=np.float32)
    # create a "horizontal line" filter as example
    F_horizontal = np.array([
        [-1.0, -1.0, -1.0],
        [+0.0, +0.0, +0.0],
        [+1.0, +1.0, +1.0]
    ], dtype=np.float32)

    # apply filter to first image in training set, ignoring first dim. ("channel")
    def conf2Dtest(image, F):
        Fh, Fw = F.shape # filter height and width
        gap = int(Fh / 2.0) # "size" of the filter
        img = image[gap:-gap, gap:-gap] # the pixels where we can apply the filter
        height, width = image.shape
        result = np.zeros_like(img)
        for h in range(gap, height-gap, 1):
            for w in range(gap, width-gap, 1):
                window = image[h-gap:h+gap+1,w-gap:w+gap+1]
                # apply the filter for this (h,w) location
                result[h-gap,w-gap] = np.sum( window * F )
        return result
    r1 = conf2Dtest(x_train[0][0], F_vertical)
    r2 = conf2Dtest(x_train[0][0], F_horizontal)

    # [before, after]
    gap = 1
    img = x_train[0][0][gap:-gap, gap:-gap]
    #draw_grid(np.array([img, r1, r2]), np.array([y_raw[0]]*3), np.array([0,1,2]), (1,3), scale=3)
    #draw_1(np.array([img, result]), [y_raw[0], y_raw[0]], 0, scale=3)


class LayerConv2D:
    """ a layer of 2D convolution filters """

    def __init__(
        self,
        input_shape,
        n_filters,
        kernel_size=(3, 3),
        stride=1,
        mode="valid",
        activation_func=ACTIVATION_ReLU,
        parentLayer=None,
        childLayer=None,
        rng=None,
    ):
        assert len(input_shape) == 3 # (channel, height, width)

        self.rng = rng if rng is not None else np.random.default_rng()

        self.input_shape = input_shape
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.K_h, self.K_w = kernel_size
        self.stride = stride
        self.mode = mode
        self.AF = activation_func
        self.parentLayer = parentLayer
        self.childLayer = childLayer

        # logic assumes odd sized kernel dims
        assert self.K_h % 2 == 1
        assert self.K_w % 2 == 1

        C_in, H_in, W_in = input_shape

        def outsize(_in, _k):
            return floor((_in - _k) / stride) + 1
        H_out = outsize(H_in, self.K_h)
        W_out = outsize(W_in, self.K_w)
        self.output_shape = (n_filters, H_out, W_out)

        self.filters = np.zeros((n_filters, C_in, self.K_h, self.K_w), dtype=np.float32)
        self.biases = np.zeros(n_filters, dtype=np.float32)

        self.filters[:] = self.rng.standard_normal(self.filters.shape, dtype=np.float32)
        self.biases[:] = 0.0

        # scale parameters according to activation type and size of kernel
        fan_in = C_in * self.K_h * self.K_w
        if activation_func == ACTIVATION_ReLU:
            # He initialization
            # - want activations to have ~constant variance across layers
            # - scale by 1/n_input, but ReLU zeros out ~50%, so use 2/n_input
            scale = np.sqrt(2.0 / fan_in)
        else:
            scale = np.sqrt(1.0 / fan_in)
        self.filters *= scale
        
        
        self.derivs_f = np.zeros_like(self.filters, dtype=np.float32)
        self.derivs_b = np.zeros_like(self.biases, dtype=np.float32)

        self.params = [self.filters, self.biases]
        self.derivs = [self.derivs_f, self.derivs_b]

        
        self.last_input = np.zeros(self.input_shape, dtype=np.float32) # store last used input to __call__()
        self.pre_activations = np.zeros(self.output_shape, dtype=np.float32)
        self.activations = np.zeros(self.output_shape, dtype=np.float32)
        self.AF = activation_func
        self.parentLayer = parentLayer
        self.childLayer = childLayer

        # allocate buffers/views for use in backprop()
        self._local_derivs_dX_df = np.zeros_like(self.filters)
        self._local_derivs_dX_db = np.zeros_like(self.biases)

        self._final_derivs_dL_df = np.zeros_like(self.filters)
        self._final_derivs_dL_db = np.zeros_like(self.biases)
        
        self._upstream_dL_dz = np.zeros(self.output_shape, dtype=np.float32)
        self._parent_deriv = np.zeros(self.input_shape, dtype=np.float32)

        # buffers for OPTIMIZER_ADAM
        self._adam_m_f = np.zeros_like(self.filters)
        self._adam_m_b = np.zeros_like(self.biases)
        self._adam_v_f = np.zeros_like(self.filters)
        self._adam_v_b = np.zeros_like(self.biases)

    def _apply_filters(self, img):
        gap_h = self.K_h // 2 # number of inaccessible pixels, due to size of filter
        gap_w = self.K_w // 2 # number of inaccessible pixels, due to size of filter
        channels, height, width = img.shape
        for f in range(self.n_filters):
            out = self.pre_activations[f]
            out.fill(self.biases[f])
            for c in range(channels):
                for h in range(gap_h, height-gap_h, self.stride):
                    for w in range(gap_w, width-gap_w, self.stride):
                        window = img[c, h-gap_h:h+gap_h+1, w-gap_w:w+gap_w+1]
                        # apply the filter for this (h,w) location
                        out_i = (h - gap_h) // self.stride
                        out_j = (w - gap_w) // self.stride
                        out[out_i, out_j] += np.sum(window * self.filters[f, c])

    def __call__(self, img):
        # - take input image
        # - apply each filter to get feature maps
        # - apply activation function (e.g. ReLU)
        # - return activations by filter
        assert img.shape == self.input_shape

        # save input for most recent forward pass
        self.last_input[:] = img
        
        # get feature maps, add biases
        self._apply_filters(img)

        # apply activation function
        if self.AF == ACTIVATION_ReLU:
            np.maximum(self.pre_activations, 0, out=self.activations)
        else:
            # no activation function
            self.activations[:] = self.pre_activations

        if self.childLayer is None:
            return self.activations
        else:
            return self.childLayer(self.activations)
    
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

class LayerFlatten:
    def __init__(self, input_shape, parentLayer=None, childLayer=None):
        self.input_shape = input_shape
        self.output_shape = (np.prod(input_shape),)
        self.parentLayer = parentLayer
        self.childLayer = childLayer
        self.last_input_shape = None
        self.out = np.zeros(self.output_shape, np.float32)

    def __call__(self, x: np.ndarray):
        assert x.shape == self.input_shape
        self.last_input_shape = x.shape
        self.out[:] = x.reshape(-1) # collapse input to 1D vector

        if self.childLayer is None:
            return self.out
        else:
            return self.childLayer(self.out)

#==============================================================================

if __name__ == "__main__":
    #test_filter()
    test_shapes()
