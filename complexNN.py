import tensorflow as tf
from tensorflow import keras
#from tensorflow.python.framework import dtypes
from tensorflow.python.keras import activations
#from tensorflow.python.ops import gradients_util
#from tensorflow.python.ops import array_ops
#from tensorflow.python.ops import tensor_array_ops
#from tensorflow.python.ops import control_flow_ops
import numpy as np
import tensorflow.python.keras.backend as K

class ComplexDense(keras.layers.Layer):
    def __init__(self, input_dim, units, activation=None, trainable=True):
        super(ComplexDense, self).__init__()
        w_init = tf.random_normal_initializer()
        if trainable is True: 
            self.w = tf.Variable(
                initial_value=tf.cast(w_init(shape=(input_dim, units), dtype='float32'), dtype=tf.complex64),
                trainable=True,
            )
        else:
            self.w = tf.Variable(
                initial_value=tf.cast(calculate_first_layer(), dtype=tf.complex64),
                trainable=False,
            )
        self.activation =  activations.get(activation)
        
    def call(self, inputs):
        return self.activation(tf.matmul(inputs, self.w))

class ComplexG(keras.layers.Layer):
    def __init__(self, input_dim, activation=None):
        super(ComplexG, self).__init__()
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(
            initial_value=tf.convert_to_tensor(np.array(np.identity(input_dim), dtype=np.complex64)),
            #initial_value=tf.convert_to_tensor(np.array(np.identity(input_dim) + np.random.rand(input_dim, input_dim), dtype=np.complex64)),
            #initial_value=tf.ones([input_dim, input_dim], tf.complex64),
            trainable=True,
        )
        self.activation =  activations.get(activation)
        
    def call(self, inputs):
        return self.activation(tf.matmul(inputs, self.w))

def calculate_first_layer():
    w = np.zeros((5, 15))
    k = 0
    for i in range(5):
        for j in range(i, 5): 
            if j == i:
                w[i][k] = 1
            else:
                w[i][k] = 1 / np.sqrt(2)   
                w[j][k] = 1 / np.sqrt(2)   
            k = k + 1    
    return w 

class LinearTrans(keras.layers.Layer):
    def __init__(self, input_dim, units, activation=None):
        super(LinearTrans, self).__init__()
        self.w = tf.Variable(
            initial_value=tf.cast(calculate_transform_layer(), dtype=tf.complex64),
            trainable=False,
            )
        self.activation =  activations.get(activation)

    def call(self, inputs):
        return self.activation(tf.matmul(inputs, self.w))

def calculate_transform_layer():
    w = np.identity(15, dtype=float)
    k = 0
    for i in range(5):
        for j in range(i, 5):
            if j == i:
                continue
            else:
                # Mapping 0 ~ 4 to the corresponding column 0, 5, 9, 12, 1 
                x = int(15 - (6 - i) * (5 - i) / 2) 
                y = int(15 - (6 - j) * (5 - j) / 2)

                w[x][k] = -0.5
                w[y][k] = -0.5 

            k = k + 1  
    return w

class Biholomorphic(keras.layers.Layer):
    '''A layer transform zi to zi*zjbar'''
    def __init__(self):
        super(Biholomorphic, self).__init__()
        
    def call(self, inputs):
        zzbar = tf.einsum('ai,aj->aij', inputs, tf.math.conj(inputs))
        zzbar = tf.reshape(zzbar, [len(zzbar),-1])
        return tf.concat([tf.math.real(zzbar), tf.math.imag(zzbar)], axis=1)

def gradients_z(func, x):
    dx_real = tf.gradients(tf.math.real(func), x)
    dx_imag = tf.gradients(tf.math.imag(func), x)
    return (tf.math.conj(dx_real) + tf.math.conj(dx_imag)*tf.constant(1j, dtype=x.dtype)) / 2

def gradients_zbar(func, x):
    dx_real = tf.gradients(tf.math.real(func), x)
    dx_imag = tf.gradients(tf.math.imag(func), x)
    return (dx_real + dx_imag*tf.constant(1j, dtype=x.dtype)) / 2


def complex_hessian(func, x):
    # Take a real function and calculate dzdzbar(f)
    #grad = gradients_z(func, x)
    grad = tf.math.conj(tf.gradients(func, x))
    hessian = tf.stack([gradients_zbar(tmp[0], x)[0]
                        for tmp in tf.unstack(grad, axis=2)],
                       axis = 1) / 2.0
 
    return hessian 

def generate_dataset(HS):
    dataset = None
    for patch in HS.patches:
        for subpatch in patch.patches:
            new_dataset = dataset_on_patch(subpatch)
            if dataset is None:
                dataset = new_dataset
            else:
                dataset = dataset.concatenate(new_dataset)
    return dataset

def dataset_on_patch(patch):

    # So that you don't need to invoke set_k()
    patch.s_tf_1, patch.J_tf_1 = patch.num_s_J_tf(k=1)
    patch.omega_omegabar = patch.get_omega_omegabar(lambdify=True)
    patch.restriction = patch.get_restriction(lambdify=True)
    patch.r_tf = patch.num_restriction_tf()

    x = tf.convert_to_tensor(np.array(patch.points, dtype=np.complex64))
    y = tf.cast(patch.num_Omega_Omegabar_tf(), dtype=tf.complex64)

    mass = y / tf.cast(patch.num_FS_volume_form_tf('identity', k=1), dtype=tf.complex64)

    # The Kahler metric calculated by complex_hessian will include the derivative of the norm_coordinate, 
    # here we transform the restriction so that the corresponding column and row will be ignored in the hessian
    trans_mat = np.delete(np.identity(patch.dimensions), patch.norm_coordinate, axis=0)
    trans_tensor = tf.convert_to_tensor(np.array(trans_mat, dtype=np.complex64))
    restriction = tf.matmul(patch.r_tf, trans_tensor) 

    dataset = tf.data.Dataset.from_tensor_slices((x, y, mass, restriction))

    return dataset

def weighted_MAPE(y_true, y_pred, mass):
    weights = mass / K.sum(mass)
    return K.sum(tf.cast(K.abs(y_true - y_pred), dtype=tf.complex64) / y_true * weights)

'''
def complex_hessians(ys,
                     xs,
                     gate_gradients=False,
                     aggregation_method=None,
                     name="hessians"):

    xs = gradients_util._AsList(xs)  # pylint: disable=protected-access
    kwargs = {
        "gate_gradients": gate_gradients,
        "aggregation_method": aggregation_method
    }

    hessians = []
    _gradients = tf.gradients(ys, xs, **kwargs) 
    
    for gradient, x in zip(_gradients, xs):
        # change shape to one-dimension without graph branching
        gradient = array_ops.reshape(gradient, [-1])

        # Declare an iterator and tensor array loop variables for the gradients.
        n = array_ops.size(x)
        loop_vars = [
            array_ops.constant(0, dtypes.int32),
            tensor_array_ops.TensorArray(x.dtype, n)
        ]
        # Iterate over all elements of the gradient and compute second order
        # derivatives.
       
        _, hessian = control_flow_ops.while_loop(
            lambda j, _: j < n,
            lambda j, result: (j + 1,
                               result.write(j, gradients_z(gradient[j], x)[0] / 2)),
            loop_vars
        )

        _shape = array_ops.shape(x)
        _reshaped_hessian = array_ops.reshape(hessian.stack(),
                                              array_ops.concat((_shape, _shape), 0))
        hessians.append(_reshaped_hessian)
    
    return hessians
'''
