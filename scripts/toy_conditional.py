import numpy as np
import os,re
import sklearn.datasets as skd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Input
import time
import matplotlib.pyplot as plt

import tensorflow_probability as tfp
tfb = tfp.bijectors
tfd = tfp.distributions


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

        
class FFJORD(keras.Model):
    def __init__(self, stacked_mlps, batch_size,num_output,name='FFJORD'):
        super(FFJORD, self).__init__()
        self._num_output=num_output
        self._batch_size = batch_size 
        ode_solve_fn = tfp.math.ode.DormandPrince(atol=1e-5).solve
        #Gaussian noise to trace solver
        trace_augmentation_fn = tfb.ffjord.trace_jacobian_hutchinson
        #trace_augmentation_fn = tfb.ffjord.trace_jacobian_exact
        
        bijectors = []
        for imlp,mlp in enumerate(stacked_mlps):
            ffjord = tfb.FFJORD(
                state_time_derivative_fn=mlp,
                ode_solve_fn=ode_solve_fn,
                trace_augmentation_fn=trace_augmentation_fn,
                name='bijector{}'.format(imlp) #Bijectors need to be names to receive conditional inputs
            )
            bijectors.append(ffjord)

        #Reverse the bijector order
        self.chain = tfb.Chain(list(reversed(bijectors)))

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
        
        return loss
    
    
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

    LR = 1e-2
    NUM_EPOCHS = 20
    STACKED_FFJORDS = 4 #Number of stacked transformations
    NUM_LAYERS = 8 #Hiddden layers per bijector
    NUM_OUTPUT = 2 #Output dimension
    NUM_HIDDEN = 4*NUM_OUTPUT #Hidden layer node size
    NUM_COND = 1 #Number of conditional dimensions
    
    #Target dataset: half moon
    DATASET_SIZE = 1024 * 8
    BATCH_SIZE = 256 

    samples= np.concatenate(
    (
        np.random.normal(0.,0.5,(DATASET_SIZE//2,2)),
        np.random.normal(-3.,0.5,(DATASET_SIZE//2,2)),
    ),0).astype(np.float32)

    constrain = np.concatenate(
        (np.ones((DATASET_SIZE//2,1)),np.zeros((DATASET_SIZE//2,1))),0).astype(np.float32)
    

    samples = np.concatenate([samples,constrain],-1) #Last dimensions are the conditional values
    
    #Stack of bijectors 
    stacked_mlps = []
    for _ in range(STACKED_FFJORDS):
        mlp_model = MLP_ODE(NUM_HIDDEN, NUM_LAYERS, NUM_OUTPUT,NUM_COND)
        stacked_mlps.append(mlp_model)

    #Create the model
    model = FFJORD(stacked_mlps,BATCH_SIZE,NUM_OUTPUT)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=LR))
    
    history = model.fit(
        samples,
        batch_size=BATCH_SIZE,
        epochs=NUM_EPOCHS,
        verbose=1,
    )

    NSAMPLES = DATASET_SIZE
    #Sample the learned distribution
    transformed = model.flow.sample(
        NSAMPLES,
        bijector_kwargs=make_bijector_kwargs(
            model.flow.bijector, {'bijector.': {'conditional_input': constrain}})
    )

    transformed_first = model.flow.sample(
        NSAMPLES,
        bijector_kwargs=make_bijector_kwargs(
            model.flow.bijector, {'bijector.': {'conditional_input': np.ones((NSAMPLES,1),dtype=np.float32)}})
    )

    transformed_second = model.flow.sample(
        NSAMPLES,
        bijector_kwargs=make_bijector_kwargs(
            model.flow.bijector, {'bijector.': {'conditional_input': np.zeros((NSAMPLES,1),dtype=np.float32)}})
    )

    #Plotting    
    fig = plt.figure(figsize=(8, 6))
    plt.subplot(211)    
    plt.scatter(transformed[:, 0], transformed[:, 1], color="r")
    plt.subplot(212)
    plt.scatter(transformed_first[:, 0], transformed_first[:, 1], color="r")
    plt.scatter(transformed_second[:, 0], transformed_second[:, 1], color="b")
    
    plot_folder = '../plots'
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)
    
    fig.savefig('{}/conditional_gaus.pdf'.format(plot_folder))


