import numpy as np
import os,re
from keras.datasets import mnist
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Input
import time
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ReduceLROnPlateau,EarlyStopping

import tensorflow_probability as tfp
tfb = tfp.bijectors
tfd = tfp.distributions
import fjord_regularization

class MLP_ODE(keras.Model):
    """Multi-layer NN ode_fn."""
    def __init__(self, num_hidden, num_layers, num_output,num_cond=2,name='mlp_ode'):
        super(MLP_ODE, self).__init__()
        self._num_hidden = num_hidden
        self._num_output = num_output
        self._num_layers = num_layers
        self._num_cond = num_cond
        self._modules = []
        
        #Fully connected layers with tanh activation and linear output
        self._modules.append(Input(shape=(1+self._num_output+self._num_cond))) #time is part of the inputs
        for _ in range(self._num_layers - 1):
            self._modules.append(layers.Dense(self._num_hidden,activation='tanh'))
            
        self._modules.append(layers.Dense(self._num_output,activation=None))
        self._model = keras.Sequential(self._modules)

        if self._num_cond > 1:
            #In more dimensiona, is useful to feed the conditional distributions after passing through an independent network model
            self._cond_model = keras.Sequential(
                [
                    Input(shape=(self._num_cond)),
                    layers.Dense(self._num_hidden,activation='relu'),
                    layers.Dense(self._num_cond,activation=None),
                ])
        
    @tf.function
    def call(self, t, data,conditional_input=None):
        if self._num_cond==1:
            #No network for a single feature
            cond_transform=tf.cast(conditional_input,dtype=tf.float32)
        else:
            cond_transform = self._cond_model(conditional_input)
            
        t = t*tf.ones([data.shape[0],1])
        inputs = tf.concat([t, data,cond_transform], -1)
        return self._model(inputs)

def make_bijector_kwargs(bijector, name_to_kwargs):
    #Hack to pass the conditional information through all the bijector layers
    if hasattr(bijector, 'bijectors'):
        return {b.name: make_bijector_kwargs(b, name_to_kwargs) for b in bijector.bijectors}
    else:
        for name_regex, kwargs in name_to_kwargs.items():
            if re.match(name_regex, bijector.name):
                return kwargs
    return {}

def save_model(model,name="ffjord",checkpoint_dir = '../checkpoints'):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    model.save_weights('{}/{}'.format(checkpoint_dir,name,save_format='tf'))

def load_model(model,name="ffjord",checkpoint_dir = '../checkpoints'):
    model.load_weights('{}/{}'.format(checkpoint_dir,name,save_format='tf')).expect_partial()
    
        
class FFJORD(keras.Model):
    def __init__(self, stacked_mlps, batch_size,num_output,trace_type='hutchinson',name='FFJORD'):
        super(FFJORD, self).__init__()
        self._num_output=num_output
        self._batch_size = batch_size 
        ode_solve_fn = tfp.math.ode.DormandPrince(atol=1e-5).solve
        #Gaussian noise to trace solver
        if trace_type=='hutchinson':
            trace_augmentation_fn = fjord_regularization.trace_jacobian_hutchinson
        elif trace_type == 'exact':
            trace_augmentation_fn = fjord_regularization.trace_jacobian_exact
        else:
            raise Exception("Invalid trace estimator")
        
        
        self.bijectors = []
        for imlp,mlp in enumerate(stacked_mlps):
            ffjord = fjord_regularization.FFJORD(
                state_time_derivative_fn=mlp,
                ode_solve_fn=ode_solve_fn,
                trace_augmentation_fn=trace_augmentation_fn,
                name='bijector{}'.format(imlp), #Bijectors need to be names to receive conditional inputs
                jacobian_factor = 1,
                kinetic_factor = 1
            )
            self.bijectors.append(ffjord)

        #Reverse the bijector order
        self.chain = tfb.Chain(list(reversed(self.bijectors)))

        self.loss_tracker = keras.metrics.Mean(name="loss")
        #Determien the base distribution
        self.base_distribution = tfp.distributions.MultivariateNormalDiag(
            loc=self._num_output*[0.0], scale_diag=self._num_output*[1.0]
        )
        
        self.flow=self.Transform()
        self._variables = self.flow.variables
        
    @property
    def metrics(self):
        """List of the model's metrics.
        We make sure the loss tracker is listed as part of `model.metrics`
        so that `fit()` and `evaluate()` are able to `reset()` the loss tracker
        at the start of each epoch and at the start of an `evaluate()` call.
        """
        return [self.loss_tracker]
    
    @tf.function
    def call(self, inputs, conditional_input=None):
        kwargs = make_bijector_kwargs(self.flow.bijector,{'bijector.': {'conditional_input':conditional_input }})
        return self.flow.bijector.forward(inputs,**kwargs)
        
            
    def Transform(self):        
        return tfd.TransformedDistribution(distribution=self.base_distribution, bijector=self.chain)

    
    @tf.function
    def log_loss(self,_x,_c):
        loss = -tf.reduce_mean(self.flow.log_prob(
            _x,
            bijector_kwargs=make_bijector_kwargs(
                self.flow.bijector, {'bijector.': {'conditional_input': _c}})                                      
        ))
        regularization_loss = tf.zeros_like(_x)
        stacked = len(self.bijectors)
        current_positions = _x
        for i in range(stacked):
            index = stacked - 1 - i
            kwargs = bijector_kwargs=make_bijector_kwargs(self.bijectors[index], {'bijector.': {'conditional_input': _c}})
            current_positions, current_loss = self.bijectors[index]._regularization_loss(current_positions, **kwargs)
            regularization_loss = regularization_loss + current_loss
        
        loss = loss + regularization_loss
        
        return loss
    
    @tf.function
    def conditional_prob(self,_x,_c):
        prob = self.flow.prob(
            _x,
            bijector_kwargs=make_bijector_kwargs(
                self.flow.bijector, {'bijector.': {'conditional_input': _c}})                                      
        )
        
        return prob
    
    
    @tf.function()
    def train_step(self, values):
        #Full shape needs to be given when using tf.dataset
        data = values[:self._batch_size,:self._num_output]
        cond = values[:self._batch_size,self._num_output:]
        data.set_shape((self._batch_size,self._num_output))
        cond.set_shape((self._batch_size,cond.shape[1]))

        with tf.GradientTape() as tape:
            loss = self.log_loss(data,cond)
            
        g = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(g, self.trainable_variables))
        self.loss_tracker.update_state(loss)

        return {"loss": self.loss_tracker.result()}
    
    @tf.function
    def test_step(self, values):
        data = values[:self._batch_size,:self._num_output]
        cond = values[:self._batch_size,self._num_output:]
        data.set_shape((self._batch_size,self._num_output))
        cond.set_shape((self._batch_size,cond.shape[1]))
        
        loss = self.log_loss(data,cond)
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}
    

        
    

if __name__ == '__main__':

    LR = 1e-3
    NUM_EPOCHS = 1000
    STACKED_FFJORDS = 1 #Number of stacked transformations
    NUM_LAYERS = 4 #Hiddden layers per bijector
    NUM_OUTPUT = 28*28 #Output dimension
    NUM_HIDDEN = 256 #Hidden layer node size
    NUM_COND = 1 #Number of conditional dimensions
    
    #Target dataset: half moon

    BATCH_SIZE = 5000 

    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    y_train = y_train.reshape((-1,1))
    X_train = X_train.reshape(-1, 784)
    X_train = X_train.astype('float32')
    X_train /= 255

    samples_train = np.concatenate([X_train,y_train],-1) #Last dimensions are the conditional values
    
    y_test = y_test.reshape((-1,1))
    X_test = X_test.reshape(-1, 784)
    X_test = X_test.astype('float32')
    X_test /= 255

    samples_test = np.concatenate([X_test,y_test],-1) #Last dimensions are the conditional values
    
    #Stack of bijectors 
    stacked_mlps = []
    for _ in range(STACKED_FFJORDS):
        mlp_model = MLP_ODE(NUM_HIDDEN, NUM_LAYERS, NUM_OUTPUT,NUM_COND)
        stacked_mlps.append(mlp_model)

    #Create the model
    model = FFJORD(stacked_mlps,BATCH_SIZE,NUM_OUTPUT)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=LR))

    callbacks = [
        ReduceLROnPlateau(patience=5, factor=0.5,
                          min_lr=1e-8,verbose=1),
        EarlyStopping(patience=10,restore_best_weights=True),
    ]

    
    history = model.fit(
        x=samples_train,
        validation_data=(samples_test,),
        batch_size=BATCH_SIZE,
        epochs=NUM_EPOCHS,
        verbose=1,
        callbacks=callbacks,
    )

    NSAMPLES = 10
    #Sample the learned distribution
    transformed = model.flow.sample(
        NSAMPLES,
        bijector_kwargs=make_bijector_kwargs(
            model.flow.bijector, {'bijector.': {'conditional_input': 4*np.ones((NSAMPLES,1))}})
    ).numpy()


    # #Plotting

    transformed = transformed*255
    transformed=transformed.reshape(-1,28,28)
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.imshow(transformed[i], cmap='gray', interpolation='none')
    
    plot_folder = '../plots'
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)
    
    plt.savefig('{}/conditional_mnist.pdf'.format(plot_folder))
