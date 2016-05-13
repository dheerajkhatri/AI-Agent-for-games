import numpy as np

import theano
import theano.tensor as T

from theano import shared, function, config
from collections import OrderedDict

from lasagne.layers import InputLayer, DenseLayer, ReshapeLayer, Conv2DLayer
import lasagne.layers
import lasagne.nonlinearities
import lasagne.updates
import lasagne.objectives
import lasagne.init

from ntm.updates import graves_rmsprop

CONV_LAYER = 1
FULL_CONN_LAYER = 2

class DQN:
    def __init__(self, batch_size, num_in_fmap, ext_in_shape, filter_type, filter_shape, filter_stride, nonlinearities, clip_err):
        '''
        batch_size: Number examples in a batch
        num_in_fmap: Number of input feature maps in an example
        ext_in_shape: (height, width of single input feature map
        filter_type: list of types of filters. (CONV_LAYER/FULL_CONN_LAYER)
        filter_shape: list of shapes of filters. 4d for CONV_LAYER and 2d for FULL_CONN_LAYER
        filter_stride: list of strides of filters. 2d for CONV_LAYER and None for FULL_CONN_LAYER
        alpha: Parameter for relu activation
        '''
        self.batch_size = batch_size
        self.ext_in_shape = ext_in_shape
        self.num_in_fmap = num_in_fmap
        self.filter_shape = filter_shape
        self.filter_type = filter_type
        self.filter_stride = filter_stride
        self.nonlinearities = nonlinearities
        self.clip_err = clip_err
        
        self.__theano_build__()
        
    def __theano_build__(self):
        X = T.tensor4('X')
        Y = T.matrix('Y')
        filter_target = T.matrix('filter_target')
        
        batch_size = self.batch_size
        ext_in_shape = self.ext_in_shape
        num_in_fmap = self.num_in_fmap
        filter_shape = self.filter_shape
        filter_type = self.filter_type
        filter_stride = self.filter_stride
        nonlinearities = self.nonlinearities
        clip_err = self.clip_err
        
        num_filter = len(filter_type)
        # num layers including input and output
        num_layer = num_filter+1
        
        input_layer_shape = (batch_size, num_in_fmap, ext_in_shape[0], ext_in_shape[1])
        network = InputLayer(shape=input_layer_shape, input_var=X)

        for i in range(0, num_filter):
            if filter_type[i] == CONV_LAYER:
                network = Conv2DLayer(network, num_filters=filter_shape[i][0], 
                            filter_size=(filter_shape[i][2], filter_shape[i][3]),
                            stride = (filter_stride[i][0], filter_stride[i][1]),
                            nonlinearity=nonlinearities[i],
                            W=lasagne.init.GlorotUniform(),
                            b=lasagne.init.Constant(.1))
            elif filter_type[i] == FULL_CONN_LAYER:
                network = DenseLayer(network, num_units=filter_shape[i][1],
                             nonlinearity=nonlinearities[i],
                             W=lasagne.init.GlorotUniform(),
                             b=lasagne.init.Constant(.1))
        
        self.network = network
        pred = lasagne.layers.get_output(network)
        err = Y-pred*filter_target
        #err = Y-pred
        if clip_err > 0:
            q_p = T.minimum(abs(err), clip_err)
            l_p = abs(err)-q_p
            loss = 0.5 * q_p ** 2 + clip_err * l_p
        else:
            loss = 0.5 * err ** 2
        o_err = T.sum(loss)
        #o_err = T.mean((Y-pred*filter_target)**2)
        #o_err = T.clip(T.mean((Y-pred*filter_target)**2), -1, 1)
        
        self.tparams = lasagne.layers.get_all_params(network, trainable=True)        
        learning_rate = T.scalar('learning_rate')
        momentum = T.scalar('momentum')
        epsilon = T.scalar('epsilon')
        alpha_rmsprop = T.scalar('alpha_rmsprop')
        rmsprop_updates = lasagne.updates.rmsprop(o_err, self.tparams, learning_rate = 0.00025, rho = 0.95, epsilon = 0.01)
        self.rmsprop_step = function([X, Y, filter_target, learning_rate, momentum, alpha_rmsprop, epsilon],
                                     [o_err], updates=rmsprop_updates, on_unused_input='ignore')
        self.prediction = function([X], pred)
        
    def get_model_params_to_save(self):
        model_params = OrderedDict()
        mykey = 0
        for value in self.tparams:
            model_params[str(mykey)] = value.get_value()
            mykey += 1
        return model_params
    
    def get_model_params(self):
        model_params = lasagne.layers.helper.get_all_param_values(self.network)
        return model_params
    
    def load_model_params(self, model_params):
        lasagne.layers.helper.set_all_param_values(self.network, model_params)
        
    def load_model_params_from_file(self, model_params):
        for i in range(len(self.tparams)):
            self.tparams[i].set_value(model_params[str(i)])

# Need to test DQN class with Hand written Digit recognition