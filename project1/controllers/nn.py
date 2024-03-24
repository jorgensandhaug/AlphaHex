from controllers.controller import Controller
import numpy as np
import jax.numpy as jnp
from jax._src.util import safe_zip
import jax

class NeuralNetwork:
    def __init__(self, activation_functions):
        self.activation_functions = activation_functions

    def forward(self, params, x):
        # params is array of "layers", where each layer is a dictionary of weights and biases
        for activation_function, layer in safe_zip(self.activation_functions, params):
            x = jnp.dot(x, layer['weights']) + layer.get("biases", jnp.zeros(layer['weights'].shape[1]))
            x = activation_function(x) 
        return x
        
    def update_parameters(self, params, grad, learning_rate):
        return jax.tree_map(lambda p, g: p - learning_rate * g, params, grad)
    
class NNController(Controller):
    def __init__(self, activation_functions, limit_output=None, limit_params=None):
        self.nn = NeuralNetwork(activation_functions)
        self.limit_output = limit_output
        self.limit_params = limit_params

    def init_params(self, key, layer_shapes, nn_weight_range=(-1, 1), use_biases=True):
        if len(layer_shapes) != len(self.nn.activation_functions):
            raise Exception("Length of layer_shapes must be the same as length of activation_functions")

        keys = jax.random.split(key, len(layer_shapes))
        params = []
        for i, shape in enumerate(layer_shapes):
            weights_key, biases_key = jax.random.split(keys[i])
            weights = jax.random.uniform(weights_key, shape, minval=nn_weight_range[0], maxval=nn_weight_range[1])
            biases = jax.random.uniform(biases_key, (shape[1],), minval=nn_weight_range[0], maxval=nn_weight_range[1])

            if use_biases:
                params.append({'weights': weights, 'biases': biases})
            else:
                params.append({'weights': weights})

        return params


    def forward(self, params, error_history, i):
        error = error_history[i]
        derivative = jax.lax.cond(i == 0, lambda error_history, i: error_history[i], lambda error_history, i: error_history[i] - error_history[i-1], error_history, i)
        integral = jnp.sum(error_history)

        x = jnp.array([error, integral, derivative]) # shape (3,)

        output = self.nn.forward(params, x)[0]
        #print(f"Output: {output}")
        return output

    def update_parameters(self, params, grad, learning_rate):
        params = self.nn.update_parameters(params, grad, learning_rate)
        if self.limit_params is not None:
            params = jax.tree_map(lambda p: jnp.clip(p, a_min=-self.limit_params[0], a_max=self.limit_params[1]), params)

        return params
