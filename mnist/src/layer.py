# layer.py
# LayerDense / LayerConv2D / LayerFlatten

# =============================================================================
# TODO (Next Steps for CNN Work)
# =============================================================================

# --- Core Experiments ---
# [ ] Compare architectures:
#     - Dense(32,32) baseline
#     - Conv → Flat → Dense
#     - Conv → Conv → Flat → Dense
#     → track accuracy, convergence speed, generalisation gap

# --- Visualisation ---
# [ ] Visualise learned filters (plot each 3x3 kernel)
# [ ] Visualise feature maps for a sample image (per filter output)

# --- Model Scaling ---
# [ ] Increase n_filters (8 → 16 → 32) and observe impact
# [ ] Try multiple Conv2D layers in succession

# --- CNN Features ---
# [ ] Add padding support ("same" mode)
# [ ] Implement LayerMaxPool2D (2x2, stride 2)
# [ ] Try architectures: Conv → Pool → Conv → Pool → Dense

# --- Alternative Architectures ---
# [ ] Try Global Average Pooling instead of Flatten
#     (Conv → avg spatial dims → Dense)

# --- Training Experiments ---
# [ ] Tune learning rate (e.g. 1e-3, 1e-2, 3e-4)
# [ ] Experiment with batch size (32 / 64 / 128)
# [ ] Overfitting test on small dataset (~1k samples)

# --- Performance / Engineering ---
# [ ] Profile runtime (forward vs backward cost)
# [ ] Optionally extend Numba usage further if needed
# [ ] Consider reducing duplicate shape calculations (minor cleanup)

# --- Stretch / Advanced ---
# [ ] Try Conv-only model (no Dense layers)
#     (Conv → Conv → GlobalAvgPool → Softmax)
# [ ] Build a CNN that consistently beats MLP baseline

# =============================================================================

from dataclasses import dataclass
from math import floor
from numba import njit
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from src.data import get_dataset, preprocess
from src.vis import draw_grid, draw_1

USE_NUMBA_CORE = True

LAYER_DENSE = 'LAYER_DENSE'
LAYER_CONV2D = 'LAYER_CONV2D'
LAYER_FLATTEN = 'LAYER_FLATTEN'

ACTIVATION_ReLU = 'ReLU'
ACTIVATION_None = None

@dataclass
class Layer:
    type: str
    outputs: object = None
    activation: str = ACTIVATION_ReLU
    CNN_kernel_size: tuple[int] = (3, 3)
    CNN_stride: int = 1
    CNN_mode: str = "valid"

class LayerDense:
    """ a layer of neurons """

    def __init__(self, n_input, n_output, activation_func=ACTIVATION_ReLU, parentLayer=None, childLayer=None, rng=None):
        self.rng = rng if rng is not None else np.random.default_rng()
        self.output_shape = (n_output,)

        # __params packs Wb together, params gives list of [weights, biases]
        self.__params = np.zeros((n_output, n_input + 1), dtype=np.float32)
        self.mx_weights = self.__params[:, :-1]
        self.vec_biases = self.__params[:, -1]
        self.params = [self.mx_weights, self.vec_biases] # [W, b]

        # initialize weights [W<-- b]
        self.__params[:, :-1] = self.rng.standard_normal((n_output, n_input), dtype=np.float32)
        # initialize biases as zero [W b<--]
        self.__params[:, -1] = 0.0

        # scale weights according to activation type and size of layer
        if activation_func == ACTIVATION_ReLU:
            # He initialization
            # - want activations to have ~constant variance across layers
            # - scale by 1/n_input, but ReLU zeros out ~50%, so use 2/n_input
            scale = np.sqrt(2.0 / n_input)
        else:
            scale = np.sqrt(1.0 / n_input)
        self.__params[:, :-1] *= scale
        
        # __derivs packs Wb together, derivs gives list of [dW, db]
        self.__derivs = np.zeros((n_output, n_input+1), dtype=np.float32)
        self.derivs_w = self.__derivs[:, :-1]
        self.derivs_b = self.__derivs[:, -1]
        self.derivs = [self.derivs_w, self.derivs_b] # [dW, db]
        
        self.vec_input = np.zeros(n_input, dtype=np.float32) # store last used input to __call__()
        self.vec_pre_activations = np.zeros(n_output, dtype=np.float32)
        self.vec_activations = np.zeros(n_output, dtype=np.float32)
        self.AF = activation_func
        self.parentLayer = parentLayer
        self.childLayer = childLayer

        # allocate buffers/views for use in backprop()
        self._local_derivs = np.zeros_like(self.__derivs)
        self._local_derivs_dX_dw = self._local_derivs[:, :-1]
        self._local_derivs_dX_db = self._local_derivs[:, -1]

        self._final_derivs_dL_dp = np.zeros_like(self.__derivs)
        self._final_derivs_dL_dw = self._final_derivs_dL_dp[:, :-1]
        self._final_derivs_dL_db = self._final_derivs_dL_dp[:, -1]
        
        self._upstream_dL_dz = np.zeros(n_output, dtype=np.float32)
        self._parent_deriv = np.zeros(n_input, dtype=np.float32)

        # buffers for OPTIMIZER_ADAM
        # need [Wb] packed in array, and list [W, b] of views
        self._adam_m = np.zeros_like(self.__params)
        self.adam_m = [self._adam_m[:, :-1], self._adam_m[:, -1]]
        self._adam_v = np.zeros_like(self.__params)
        self.adam_v = [self._adam_v[:, :-1], self._adam_v[:, -1]]

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
    
    def reset_derivs(self):
        for d in self.derivs: d.fill(0.0)

    def scale_derivs(self, n):
        for d in self.derivs: d /= n

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
        self.__derivs += self._final_derivs_dL_dp

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
        self.adam_m = [self._adam_m_f, self._adam_m_b]
        self.adam_v = [self._adam_v_f, self._adam_v_b]

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
        if USE_NUMBA_CORE:
            self.pre_activations[:] = conv2d_forward_core(
                img,
                self.filters,
                self.biases,
                self.stride,
                self.K_h,
                self.K_w,
            )
        else:
            self._apply_filters(img) # non-numba version

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
    
    def reset_derivs(self):
        for d in self.derivs: d.fill(0.0)

    def scale_derivs(self, n):
        for d in self.derivs: d /= n
    
    def backprop(self, upstream_deriv):
        p = self.parentLayer
        # upstream_deriv: 
        # - is an ndarray, same shape as outputs of THIS layer
        # - contains derivatives of the loss wrt THIS layer's outputs (activations)
        # - (dL/da_1, dL/da_2, ..., dL/da_n)

        # step 1. transform upstream_derivs from dL/da to dL/dz (wrt pre-activations)
        # - dL/dz = dL/da * da/dz
        # - da/dz (ReLU): 0 (if z <= 0), 1 (otherwise)
        # - da/dz (None): 1
        if self.AF == ACTIVATION_ReLU:
            self._upstream_dL_dz[:] = upstream_deriv * (self.pre_activations > 0)
        else:
            self._upstream_dL_dz[:] = upstream_deriv

        # step 2. mirror the forward pass loops, accumulating into derivs_f/derivs_b
        # each output value z[f, out_i, out_j] was produced from one input window
        # and one filter slice filters[f, c]
        self._parent_deriv.fill(0.0)

        # use numba backprop?
        if USE_NUMBA_CORE:
            conv2d_backward_core(
                self.last_input,
                self.filters,
                self._upstream_dL_dz,
                self.derivs_f,
                self.derivs_b,
                self._parent_deriv,
                self.stride,
                self.K_h,
                self.K_w,
            )
        else:
            gap_h = self.K_h // 2 # number of inaccessible pixels, due to size of filter
            gap_w = self.K_w // 2 # number of inaccessible pixels, due to size of filter
            channels, height, width = self.last_input.shape
            for f in range(self.n_filters):
                for h in range(gap_h, height-gap_h, self.stride):
                    for w in range(gap_w, width-gap_w, self.stride):
                        out_i = (h - gap_h) // self.stride
                        out_j = (w - gap_w) // self.stride

                        # dz is the scalar "blame" assigned to this particular output pixel.
                        # We use it to update:
                        #   1. filter gradients: how much each filter weight contributed
                        #   2. bias gradients: bias contributed directly to this output
                        #   3. parent/input gradients: how much loss should be passed back to each input pixel
                        dz = self._upstream_dL_dz[f, out_i, out_j]

                        self.derivs_b[f] += dz # note bias is applied AFTER summing over channels, not per channel
                        for c in range(channels):
                            window = self.last_input[c, h-gap_h:h+gap_h+1, w-gap_w:w+gap_w+1]
                            # z included window[u, v] * filter[f, c, u, v],
                            # so dL/d_filter = window * dL/dz
                            self.derivs_f[f, c] += window * dz

                            # the same output z also depended on the input window,
                            # so scatter the filter weights, scaled by dz, back into the input-gradient image
                            self._parent_deriv[c, h-gap_h:h+gap_h+1, w-gap_w:w+gap_w+1] += self.filters[f, c] * dz

        if p is not None: 
            p.backprop(self._parent_deriv) # if has parent, keep backprop'ing

@njit
def conv2d_forward_core(x, filters, biases, stride, K_h, K_w):
    C, H, W = x.shape
    F = filters.shape[0]

    gap_h = K_h // 2
    gap_w = K_w // 2

    H_out = (H - K_h) // stride + 1
    W_out = (W - K_w) // stride + 1

    out = np.zeros((F, H_out, W_out), dtype=np.float32)

    for f in range(F):
        for i in range(H_out):
            for j in range(W_out):
                acc = biases[f]

                h = i * stride + gap_h
                w = j * stride + gap_w

                for c in range(C):
                    for u in range(K_h):
                        for v in range(K_w):
                            acc += (
                                x[c, h - gap_h + u, w - gap_w + v]
                                * filters[f, c, u, v]
                            )

                out[f, i, j] = acc

    return out

@njit
def conv2d_backward_core(
    x,
    filters,
    dZ,
    derivs_f,
    derivs_b,
    parent_deriv,
    stride,
    K_h,
    K_w,
):
    C, H, W = x.shape
    F, _, _, _ = filters.shape

    gap_h = K_h // 2
    gap_w = K_w // 2

    H_out = dZ.shape[1]
    W_out = dZ.shape[2]

    for f in range(F):
        for i in range(H_out):
            for j in range(W_out):

                dz = dZ[f, i, j]
                derivs_b[f] += dz

                h = i * stride + gap_h
                w = j * stride + gap_w

                for c in range(C):
                    for u in range(K_h):
                        for v in range(K_w):

                            x_val = x[c, h - gap_h + u, w - gap_w + v]
                            w_val = filters[f, c, u, v]

                            # dL/dW
                            derivs_f[f, c, u, v] += x_val * dz

                            # dL/dX
                            parent_deriv[c, h - gap_h + u, w - gap_w + v] += w_val * dz
class LayerFlatten:
    def __init__(self, input_shape, parentLayer=None, childLayer=None):
        self.input_shape = input_shape
        self.output_shape = (int(np.prod(input_shape)),)
        self.parentLayer = parentLayer
        self.childLayer = childLayer
        self.out = np.zeros(self.output_shape, np.float32)
        # buffer for backprop()
        self._parent_deriv = np.zeros(self.input_shape, np.float32)
        # empty lists for Model.update_params()
        self.params = []
        self.derivs = []
        self.adam_m = []
        self.adam_v = []

    def __call__(self, x: np.ndarray):
        assert x.shape == self.input_shape
        self.out[:] = x.reshape(-1) # collapse input to 1D vector
        if self.childLayer is None:
            return self.out
        else:
            return self.childLayer(self.out)
    
    def reset_derivs(self):
        pass # N/A for LayerFlatten

    def scale_derivs(self, n):
        pass # N/A for LayerFlatten

    def backprop(self, upstream_deriv: np.ndarray):
        assert upstream_deriv.shape == self.output_shape
        p = self.parentLayer
        assert p is not None # not valid for flatten to be the first layer, must have parent
        # take the upstream_deriv, reshape to self.input_shape into self._parent_deriv
        self._parent_deriv[:] = upstream_deriv.reshape(self.input_shape)
        p.backprop(self._parent_deriv) # if has parent, keep backprop'ing


def test_Conv2D_shapes():
    x_raw, y_raw, x_raw_test, y_raw_test = get_dataset() # get train/test data
    x_train, y_train = preprocess(x_raw, y_raw, flatten=False)

    L = LayerConv2D((1, 28, 28), n_filters=8, kernel_size=(3, 3), stride=1)
    out = L(x_train[0])
    print(out.shape)
    L = LayerConv2D((1, 28, 28), n_filters=8, kernel_size=(3, 3), stride=2)
    out = L(x_train[0])
    print(out.shape)

def test_Conv2D_filter():
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

def test_Conv2D_Flatten():
    # test the "plumbing" of LayerConv2D and LayerFlatten
    from src.train import TrainConfig, run

    x_raw, y_raw, x_raw_test, y_raw_test = get_dataset() # get train/test data
    _x_train, y_train = preprocess(x_raw, y_raw, flatten=False)
    _x_test, y_test = preprocess(x_raw_test, y_raw_test, flatten=False)

    # use less data...
    n_train = 5000
    n_test = 1000

    _x_train = _x_train[:n_train]
    y_train = y_train[:n_train]
    _x_test = _x_test[:n_test]
    y_test = y_test[:n_test]

    # create the Conv2D and Flatten layers, "wire" them together with the parent/child
    Lconv = LayerConv2D((1, 28, 28), n_filters=8, kernel_size=(3, 3), stride=2)
    Lflat = LayerFlatten(Lconv.output_shape, parentLayer=Lconv)
    Lconv.childLayer = Lflat

    # create a new dataset by running the forward pass of these two layers via Lconv.__call__()
    print("transforming [training] data...")
    x_train = np.zeros((len(_x_train),) + Lflat.output_shape, dtype=np.float32)
    for i in range(len(_x_train)): x_train[i] = Lconv(_x_train[i])
    print("transforming [test] data...")
    x_test = np.zeros((len(_x_test),) + Lflat.output_shape, dtype=np.float32)
    for i in range(len(_x_test)): x_test[i] = Lconv(_x_test[i])

    # train the MLP model on this new dataset
    dataset = (x_train, y_train, x_test, y_test)
    cfg = TrainConfig([32, 32], max_epochs=20)

    run(cfg, dataset, showLossPlot=True, showPCA=True, quiet=False)

#==============================================================================

if __name__ == "__main__":
    #test_Conv2D_filter()
    #test_Conv2D_shapes()
    test_Conv2D_Flatten()
