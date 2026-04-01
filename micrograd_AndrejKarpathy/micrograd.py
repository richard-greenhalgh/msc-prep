# "micrograd" module, created by following along with Andrej Karpathy's lecture
# Andrej's video: https://www.youtube.com/watch?v=VMj-3S1tku0
# Andrej's github: https://github.com/karpathy/micrograd

import math
import numpy as np
import random

class Value:

    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children) # set() unordered collection of unique elements
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data,
                    (self, other), # keep record of children
                    '+'            # keep track of the operator that created the Value object
        )

        def _backward():
            # for addition we "copy" the parent grad to the child
            # A      + B      = C
            # parent + parent = child
            # 
            # both local derivs are 1.0: dC/dA = 1, dC/dB = 1
            # propagate the child deriv back to parent(s) using chain rule
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        
        # note setting member as the function _backward()
        # so the output Value() "out" has access to local derivative of its inputs
        out._backward = _backward

        return out
    
    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data,
                    (self, other), # keep record of children
                    '*'            # keep track of the operator that created the Value object
        )

        def _backward():
            # for multiplication we mutliply by the child
            # A      * B      = C
            # parent * parent = child
            # 
            # local derivs are: dC/dA = B, dC/dB = A
            # propagate the child deriv back to parent(s) using chain rule
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        # note setting member as the function _backward()
        # so the output Value() "out" has access to local derivative of its inputs
        out._backward = _backward

        return out
    
    def __pow__(self, other): # self**other
        assert isinstance(other, (int, float)), "only int/float powers supported"
        out = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            # for multiplication we mutliply by the child
            # A      ^ B      = C
            # parent ^ parent = child
            # 
            # local derivs are: dC/dA = B*A**(B-1), dC/dB = logA A**B
            # propagate the child deriv back to parent(s) using chain rule
            self.grad += (other * self.data**(other-1.0)) * out.grad
            #other.grad += self.data * out.grad

        # note setting member as the function _backward()
        # so the output Value() "out" has access to local derivative of its inputs
        out._backward = _backward

        return out
    
    def __rmul__(self, other):
        return self * other
    
    def __truediv__(self, other):
        return self * other**-1.0
    
    def __neg__(self): # -self
        return self * -1
    
    def __sub__(self, other): # self - other
        return self + (-other)

    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(t, (self,), 'tanh')

        def _backward():
            # for tanh
            # B     = tanh(A)
            # child = tanh(parent)
            # 
            # child deriv wrt to input(s): dB/dA = 1 - tanh(A)**2 = 1 - B**2
            # propagate the child deriv back to parent(s) using chain rule
            self.grad += (1 - t**2) * out.grad
            #other.grad = ...  no "other" for this operator

        # note setting member as the function _backward()
        # so the output Value() "out" has access to local derivative of its inputs
        out._backward = _backward

        return out
    
    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self,), 'exp')

        def _backward():
            # for exp
            # B     = exp(A)
            # child = exp(parent)
            # 
            # child deriv wrt to input(s): dB/dA = exp(A) = B
            # propagate the child deriv back to parent(s) using chain rule
            self.grad += out.data * out.grad
            #other.grad = ...  no "other" for this operator

        # note setting member as the function _backward()
        # so the output Value() "out" has access to local derivative of its inputs
        out._backward = _backward

        return out

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        topo

        self.grad = 1.0
        for node in reversed(topo): # note to start at the output node o, we need to reverse topo
            # print('_backward() for ', node.label)
            node._backward()

class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1,1))

    def __call__(self, x):
        # w * x + b
        #print(list(wi*xi for wi, xi in zip(self.w, x)))
        #print(self.b)
        activation = sum( (wi*xi for wi, xi in zip(self.w, x)), self.b )
        out = activation.tanh()
        return out
    
    def parameters(self):
        return self.w + [self.b]

class Layer:
    def __init__(self, nin, nout):
        # a list of <nout> Neurons each having <nin> inputs
        self.neurons = [Neuron(nin) for _ in range(nout)]
    
    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]
        #ps = []
        #for neuron in self.neurons:
        #    ps.extend( neuron.parameters() )
        #return ps

class MLP:
    def __init__(self, nin, nouts):
        # a multi layer perceptron
        #   <nin> inputs, connected to:
        #   nouts[0] neurons, conneced to:
        #   nouts[1] neurons, ...
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

def test_micrograd():
    random.seed(42)
    
    # raw data, 4 observations
    xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0]
    ]

    # target y values: "a binary classifier", either +1 or -1
    ys = [1.0, -1.0, -1.0, 1.0]

    # create the neural net
    n = MLP(3, [4, 4, 1])

    def forward_pass():
        ypred = [n(x) for x in xs]
        loss = sum([(yout-ygt)**2 for ygt, yout in zip(ys, ypred)])
        return ypred, loss

    def summary(iter, loss, ypred):
        print(f"{iter:04d}", f"-- loss: {loss.data:010.6f}", '-- pred:', [f"{v.data:010.6f}" for v in ypred])

    # iterate until loss < 0.001?
    ypred, loss = forward_pass()
    iter = 0
    summary(iter, loss, ypred)
    while loss.data >= 0.001 and iter < 1000:
        
        # forward pass
        ypred, loss = forward_pass()
        
        # reset all gradients to zero, backward pass
        for p in n.parameters():
            p.grad = 0.0
        loss.backward()

        # parameter update
        for p in n.parameters():
            p.data += -0.05 * p.grad
        
        iter += 1
        if iter % 25 == 0: summary(iter, loss, ypred)
    # done
    summary(iter, loss, ypred)

if __name__ == '__main__':
    test_micrograd()