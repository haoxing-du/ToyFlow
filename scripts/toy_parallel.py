import numpy as np
import os,re
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Input
import time
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ReduceLROnPlateau,EarlyStopping
import horovod.tensorflow.keras as hvd
import preprocessing
import argparse


import tensorflow_probability as tfp
tfb = tfp.bijectors
tfd = tfp.distributions
#import fjord_regularization
tf.random.set_seed(1234)

class BACKBONE_ODE(keras.Model):
    """ODE backbone network implementation"""
    def __init__(self, data_shape,num_cond,config,name='mlp_ode',use_conv=False):
        super(BACKBONE_ODE, self).__init__()
        self._data_shape = data_shape
        self._num_cond = num_cond
        if config is None:
            raise ValueError("Config file not given")

        self.config = config
        #config file with ML parameters to be used during training        
        self.activation = self.config['ACT']


        #Transformation applied to conditional inputs
        inputs_time = Input((1))
        inputs_cond = Input((self._num_cond))
        # conditional=inputs_time
        conditional = tf.concat([inputs_time,inputs_cond],-1)


        #The architecture used to model the time-dependent flow are based on an encoder-decoder implementation with additional skip connections (U-NET). Here are simple implementations for each. How can you adapt the convolutional model to be used for the calorimeter dataset? Tip: The calorimeter consists of 3 layers, each with an image-like shape of 3x96, 12x12, and 12x6 
        if use_conv:
            inputs,outputs = self.ConvModel(conditional)
        else:
            inputs,outputs = self.DenseModel(conditional)

        self._model = keras.Model(inputs=[inputs,inputs_time,inputs_cond],outputs=outputs)
                        

    def call(self, t, data,conditional):
        t_reshape = t*tf.ones_like(conditional,dtype=tf.float32)
        return self._model([data,t_reshape,conditional])


    def DenseModel(self,time_embed):
        inputs = Input((self._data_shape))
        nlayers =self.config['NLAYERS']
        dense_sizes = self.config['LAYER_SIZE']
        skip_layers = []
        
        #Encoder
        layer_encoded = self.time_dense(inputs,time_embed,dense_sizes[0])
        skip_layers.append(layer_encoded)
        for ilayer in range(1,nlayers):
            layer_encoded=self.time_dense(layer_encoded,time_embed,dense_sizes[ilayer])
            skip_layers.append(layer_encoded)
        skip_layers = skip_layers[::-1] #reverse the order


        #Decoder
        layer_decoded = skip_layers[0]
        for ilayer in range(len(skip_layers)-1):
            layer_decoded = self.time_dense(layer_decoded,time_embed,dense_sizes[len(skip_layers)-2-ilayer])
            layer_decoded = (layer_decoded+ skip_layers[ilayer+1])/np.sqrt(2)
        
            
        outputs = layers.Dense(self._data_shape[0],activation=None,use_bias=True)(layer_decoded)
        return inputs, outputs

    
    def ConvModel(self,time_embed):

        # inputs = Input((int(np.prod(self._data_shape)))) #Inputs need to be flattened so we combine multiple bijectors

        # #Recover the image-like format
        # image_shape = [-1] + self._data_shape + [1]
        
        # inputs_reshaped = tf.reshape(inputs,image_shape)

        inputs = Input((self._data_shape))
        
        stride_size=self.config['STRIDE']
        kernel_size =self.config['KERNEL']
        nlayers =self.config['NLAYERS']
        conv_sizes = self.config['LAYER_SIZE']
        skip_layers = []

        
        #Encoder
        layer_encoded = self.time_conv(inputs,time_embed,conv_sizes[0],
                                       kernel_size=kernel_size,
                                       stride=1,padding='same')
        skip_layers.append(layer_encoded)
        for ilayer in range(1,nlayers):
            # layer_encoded = layers.Conv2D(conv_sizes[ilayer],activation=self.activation,
            #                               kernel_size=kernel_size,padding='same',
            #                               strides=stride_size)(layer_encoded)

            layer_encoded = self.time_conv(layer_encoded,time_embed,conv_sizes[ilayer],
                                          kernel_size=kernel_size,padding='same',
                                          stride=stride_size)

            
            
            skip_layers.append(layer_encoded)

        skip_layers = skip_layers[::-1] #reverse the order
            
        #Decoder
        layer_decoded = skip_layers[0]
        for ilayer in range(len(skip_layers)-1):


            # layer_decoded = layers.Conv2DTranspose(conv_sizes[len(skip_layers)-2-ilayer],
            #                                        kernel_size=kernel_size,padding="same",
            #                                        strides=stride_size,
            #                                        activation=self.activation)(layer_decoded)

            layer_decoded = self.time_transconv(layer_decoded,time_embed,
                                                conv_sizes[len(skip_layers)-2-ilayer],
                                                kernel_size=kernel_size,padding='same',
                                                stride=stride_size)
            
            
            layer_decoded = (layer_decoded+ skip_layers[ilayer+1])/np.sqrt(2)

            
            
        outputs = layers.Conv2D(1,kernel_size=kernel_size,padding="same",
                                strides=1,activation=None,use_bias=True)(layer_decoded)

        # outputs = tf.squeeze(outputs)
        # outputs = layers.Flatten()(outputs) #Flatten the output 
        return inputs, outputs


        
    def time_conv(self,input_layer,embed,hidden_size,stride=1,kernel_size=2,padding="same",activation=True):
        ## Incorporate information from conditional inputs
        time_layer = layers.Dense(hidden_size,activation=self.activation,use_bias=False)(embed)

        layer = layers.Conv2D(hidden_size,kernel_size=kernel_size,padding=padding,
                              strides=stride,use_bias=False,activation=self.activation)(input_layer)
        time_layer = tf.reshape(time_layer,(-1,1,1,hidden_size))
        layer=layer+time_layer
        layer = layers.Conv2D(hidden_size,kernel_size=kernel_size,padding=padding,
                              strides=1,use_bias=True,activation=None)(layer)
        if activation:            
            return self.activate(layer)
        else:
            return layer

    def time_transconv(self,input_layer,embed,hidden_size,stride=1,kernel_size=2,padding="same",activation=True):
        ## Incorporate information from conditional inputs
        time_layer = layers.Dense(hidden_size,activation=self.activation,use_bias=False)(embed)

        layer = layers.Conv2DTranspose(hidden_size,kernel_size=kernel_size,padding=padding,
                                       strides=stride,use_bias=False,
                                       activation=self.activation)(input_layer)
        
        time_layer = tf.reshape(time_layer,(-1,1,1,hidden_size))
        layer=layer+time_layer
        layer = layers.Conv2D(hidden_size,kernel_size=kernel_size,padding=padding,
                              strides=1,use_bias=True,activation=None)(layer)
        if activation:            
            return self.activate(layer)
        else:
            return layer

        

    def time_dense(self,input_layer,embed,hidden_size,activation=True):
        ## Incorporate information from conditional inputs
        time_layer = layers.Dense(hidden_size,activation=self.activation,use_bias=False)(embed)
        layer = layers.Dense(hidden_size,use_bias=False,activation=self.activation)(input_layer)
        layer+=time_layer
        layer = layers.Dense(hidden_size,use_bias=False,activation=None)(layer)
        # layer = layers.BatchNormalization()(layer)
        # layer = layers.Dropout(0.1)(layer)
        if activation:            
            return self.activate(layer)
        else:
            return layer


    def activate(self,layer):
        if self.activation == 'leaky_relu':                
            return keras.layers.LeakyReLU(-0.01)(layer)
        elif self.activation == 'tanh':
            return keras.activations.tanh(layer)
        elif self.activation == 'softplus':
            return keras.activations.softplus(layer)
        elif self.activation == 'relu':
            return keras.activations.relu(layer)
        elif self.activation == 'swish':
            return keras.activations.swish(layer)
        else:
            raise ValueError("Activation function not supported!")   


    

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
    def __init__(self, stacked_layers,num_output,num_batch,trace_type='hutchinson',name='FFJORD'):
        super(FFJORD, self).__init__()
        self._num_output=num_output
        self._num_batch = num_batch
        
        ode_solve_fn = tfp.math.ode.DormandPrince(atol=1e-5,rtol=1e-5).solve
        # ode_solve_fn = tfp.math.ode.BDF().solve
        #Gaussian noise to trace solver
        if trace_type=='hutchinson':
            trace_augmentation_fn = tfb.ffjord.trace_jacobian_hutchinson
        elif trace_type == 'exact':
            trace_augmentation_fn = tfb.ffjord.trace_jacobian_exact
        else:
            raise Exception("Invalid trace estimator")
        
        
        self.bijectors = []
        for ilayer,layer in enumerate(stacked_layers):
            ffjord = tfb.FFJORD(
                state_time_derivative_fn=layer, #Each of the CNN models we build
                ode_solve_fn=ode_solve_fn,
                trace_augmentation_fn=trace_augmentation_fn,
                name='bijector{}'.format(ilayer), #Bijectors need to be names to receive conditional inputs
                # jacobian_factor = 0.0, #Regularization strength
                # kinetic_factor = 0.0
            )
            self.bijectors.append(ffjord)

        #Reverse the bijector order
        self.chain = tfb.Chain(list(reversed(self.bijectors)))

        self.loss_tracker = keras.metrics.Mean(name="loss")
        #Determine the base distribution
        self.base_distribution = tfp.distributions.Normal(
            loc=np.zeros(self._num_output,dtype=np.float32), scale=np.ones(self._num_output,dtype=np.float32)
        )
        
        self.flow=tfd.TransformedDistribution(distribution=self.base_distribution, bijector=self.chain)
    @property
    def metrics(self):
        """List of the model's metrics.
        We make sure the loss tracker is listed as part of `model.metrics`
        so that `fit()` and `evaluate()` are able to `reset()` the loss tracker
        at the start of each epoch and at the start of an `evaluate()` call.
        """
        return [self.loss_tracker]
    

    def call(self, inputs, conditional=None):
        kwargs = make_bijector_kwargs(self.flow.bijector,{'bijector.': {'conditional':conditional }})
        return self.flow.bijector.forward(inputs,**kwargs)
        

    def log_loss(self,_x,_c):
        loss = -tf.reduce_mean(self.flow.log_prob(
            _x,
            bijector_kwargs=make_bijector_kwargs(
                self.flow.bijector, {'bijector.': {'conditional': _c}}) 
        ))
        return loss
    

    def conditional_prob(self,_x,_c):
        prob = self.flow.prob(
            _x,
            bijector_kwargs=make_bijector_kwargs(
                self.flow.bijector, {'bijector.': {'conditional': _c}})                                      
        )
        
        return prob
    
    
    @tf.function()
    def train_step(self, inputs):
        #Full shape needs to be given when using tf.dataset

        data,cond = inputs
        # data.set_shape((self._num_batch,data.shape[1]))
        # cond.set_shape((self._num_batch,cond.shape[1]))

        with tf.GradientTape() as tape:
            loss = self.log_loss(data,cond)
            
        g = tape.gradient(loss, self.trainable_variables)
        # g = [tf.clip_by_norm(grad, 1) #clip large gradients
        #      for grad in g]

        self.optimizer.apply_gradients(zip(g, self.trainable_variables))
        self.loss_tracker.update_state(loss)

        return {"loss": self.loss_tracker.result()}
    
    @tf.function
    def test_step(self, inputs):
        data,cond = inputs
        # data.set_shape((self._num_batch,data.shape[1]))
        # cond.set_shape((self._num_batch,cond.shape[1]))
        
        # data.set_shape((self._num_batch,self._num_output))
        # cond.set_shape((self._num_batch,cond.shape[1]))
        
        loss = self.log_loss(data,cond)
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}
    

        
    

if __name__ == '__main__':

    #Start horovod and pin each GPU to a different process
    hvd.init()    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')


    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_folder', default='/pscratch/sd/v/vmikuni/SGM/gamma.hdf5', help='Path to calorimeter dataset used during training')
    parser.add_argument('--model_name', default='moon', help='Name of the model to train. Options are: mnist, moon, calorimeter')
    flags = parser.parse_args()


        
    model_name = flags.model_name #Let's try the parallel training using different models and different network architectures. Possible options are [mnist,moon,calorimeter]
        
    if model_name == 'mnist':
        from tensorflow.keras.datasets import mnist, fashion_mnist
        NTEST=500
        dataset_config = preprocessing.LoadJson('config_mnist.json')
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        ntrain,samples_train =preprocessing.MNIST_prep(X_train, y_train)
        ntest,samples_test =preprocessing.MNIST_prep(X_test, y_test)
        
        use_conv = True #Use convolutional networks for the model backbone
    elif model_name == 'moon':
        import sklearn.datasets as skd
        dataset_config = preprocessing.LoadJson('config_moon.json')
        moons_train = skd.make_moons(n_samples=60000, noise=.06)[0].astype("float32")
        moons_test = skd.make_moons(n_samples=10000, noise=.06)[0].astype("float32")
        ntrain,samples_train =preprocessing.Moon_prep(moons_train)
        ntest,samples_test =preprocessing.Moon_prep(moons_test)
        print(ntest,ntrain)
        use_conv = False
    elif model_name == 'calorimeter':        
        dataset_config = preprocessing.LoadJson('config_calorimeter.json')
        file_path=flags.data_folder
        ntrain,ntest,samples_train,samples_test = preprocessing.Calo_prep(file_path)
        use_conv = False
    else:
        raise ValueError("Model not implemented!")
        
        
    LR = float(dataset_config['LR'])
    NUM_EPOCHS = dataset_config['MAXEPOCH']
    STACKED_FFJORDS = dataset_config['NSTACKED'] #Number of stacked transformations
    
    NUM_LAYERS = dataset_config['NLAYERS'] #Hiddden layers per bijector
    BATCH_SIZE = dataset_config['BATCH']

    
    #Stack of bijectors 
    stacked_convs = []
    for _ in range(STACKED_FFJORDS):
        conv_model = BACKBONE_ODE(dataset_config['SHAPE'], 1, config=dataset_config,use_conv=use_conv)
        stacked_convs.append(conv_model)

    #Create the model
    model = FFJORD(stacked_convs,dataset_config['SHAPE'],BATCH_SIZE)


    opt = tf.optimizers.Adam(LR)

    # Horovod: add Horovod DistributedOptimizer.
    opt = hvd.DistributedOptimizer(
        opt, backward_passes_per_step=1, average_aggregated_gradients=True)
    
    model.compile(optimizer=opt,experimental_run_tf_function=False)

    callbacks = [
        hvd.callbacks.BroadcastGlobalVariablesCallback(0),
        hvd.callbacks.MetricAverageCallback(),
        ReduceLROnPlateau(patience=5, factor=0.1,
                          min_lr=1e-7,verbose=hvd.rank() == 0),
        EarlyStopping(patience=dataset_config['EARLYSTOP'],restore_best_weights=True),
    ]

    
    history = model.fit(
        samples_train.batch(BATCH_SIZE),
        steps_per_epoch=int(ntrain/BATCH_SIZE),
        validation_data=samples_test.batch(BATCH_SIZE),
        validation_steps=max(1,int(ntest/BATCH_SIZE)),
        epochs=NUM_EPOCHS,
        verbose=hvd.rank() == 0,
        callbacks=callbacks,
    )

    if hvd.rank() == 0:
        plot_folder = '../plots'
        if not os.path.exists(plot_folder):
            os.makedirs(plot_folder)
            
        save_model(model,name=model_name)
        if model_name == 'moon':
            NSAMPLES = 1000
            #Sample the learned distribution
            transformed = model.flow.sample(
                NSAMPLES,
                bijector_kwargs=make_bijector_kwargs(
                    model.flow.bijector, {'bijector.': {'conditional': np.ones((NSAMPLES),dtype=np.float32)}})
            ).numpy()


            fig = plt.figure(figsize=(8, 6))
            plt.scatter(transformed[:, 0], transformed[:, 1], color="r")
            fig.savefig('{}/double_moon.pdf'.format(plot_folder))


        elif model_name == 'mnist':
            NSAMPLES = 10
            #Sample the learned distribution
            transformed = model.flow.sample(
                NSAMPLES,
                bijector_kwargs=make_bijector_kwargs(
                    model.flow.bijector, {'bijector.': {'conditional': y_test.astype('float32')[:NSAMPLES]/10.0}})
            ).numpy()


            #Plotting
            # transformed =(transformed + 1.) / 2.
            # transformed = np.clip(transformed,0,1)
            transformed=transformed.reshape(-1,28,28)
            for i in range(9):
                plt.subplot(3,3,i+1)
                plt.imshow(transformed[i], cmap='gray', interpolation='none')
        
            plt.savefig('{}/conditional_mnist.pdf'.format(plot_folder))

        elif model_name == 'calorimeter':
            #What distributions are interesting to look at?
            pass
