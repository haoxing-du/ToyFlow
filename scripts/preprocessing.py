import json, yaml
import os
import h5py as h5
import numpy as np
import tensorflow as tf


def split_data(data,nevts,frac=0.8):
    data = data.shuffle(nevts)
    train_data = data.take(int(frac*nevts)).repeat()
    test_data = data.skip(int(frac*nevts)).repeat()
    return train_data,test_data

def LoadJson(file_name):
    import json,yaml
    JSONPATH = os.path.join(file_name)
    return yaml.safe_load(open(JSONPATH))

def ReverseNorm(e,e_layer,e_voxel):
    '''Revert the transformations applied to the training set'''    
    alpha = 1e-6
    
    gen_energy = 10**(e+1)
    # layer_norm = 10**(e_layer)*gen_energy

    exp = np.exp(e_layer)    
    x = exp/(1+exp)
    u_norm = (x-alpha)/(1 - 2*alpha)
    
    layer_norm= np.zeros(u_norm.shape,dtype=np.float32)
    layer_norm[:,0] = np.squeeze(gen_energy)*u_norm[:,0]*u_norm[:,1]
    layer_norm[:,1] = np.squeeze(gen_energy)*u_norm[:,0]*u_norm[:,2]*(1-u_norm[:,1])
    layer_norm[:,2] = np.squeeze(gen_energy)*u_norm[:,0]*(1-u_norm[:,1])*(1-u_norm[:,2])

    
    exp = np.exp(e_voxel)    
    x = exp/(1+exp)
    voxel = (x-alpha)/(1 - 2*alpha)
    voxel[:,:288] = voxel[:,:288] * np.expand_dims(layer_norm[:,0],-1)/np.sum(voxel[:,:288],-1,keepdims=True)
    voxel[:,288:432] = voxel[:,288:432] * np.expand_dims(layer_norm[:,1],-1)/np.sum(voxel[:,288:432],-1,keepdims=True)
    voxel[:,432:] = voxel[:,432:] * np.expand_dims(layer_norm[:,2],-1)/np.sum(voxel[:,432:],-1,keepdims=True)
    
    return gen_energy,voxel

def DataLoader(file_name,nevts=-1):
    '''
    Inputs:
    - name of the file to load
    - number of events to use
    Outputs:
    - Generated particle energy (value to condition the flow) (nevts,1)
    - Energy deposition in each layer (nevts,3)
    - Normalized energy deposition per voxel (nevts,504)
    '''
    import horovod.tensorflow.keras as hvd
    hvd.init()
    
    with h5.File(file_name,"r") as h5f:
        if nevts <0:
            nevts = len(h5f['energy'])
        e = h5f['energy'][hvd.rank():nevts:hvd.size()].astype(np.float32)
        layer0= h5f['layer_0'][hvd.rank():nevts:hvd.size()].astype(np.float32)/1000.0
        layer1= h5f['layer_1'][hvd.rank():nevts:hvd.size()].astype(np.float32)/1000.0
        layer2= h5f['layer_2'][hvd.rank():nevts:hvd.size()].astype(np.float32)/1000.0

    def preprocessing(data):
        ''' 
        Inputs: Energy depositions in a layer
        Outputs: Total energy of the layer and normalized energy deposition
        '''
        x = data.shape[1]
        y = data.shape[2]
        data_flat = np.reshape(data,[-1,x*y])
        #add noise like caloflow does
        data_flat +=np.random.uniform(0,1e-6,size=data_flat.shape)
        energy_layer = np.sum(data_flat,-1).reshape(-1,1)
        #Some particle showers have no energy deposition at the last layer
        data_flat = np.ma.divide(data_flat,energy_layer).filled(0)

        #Log transform from caloflow paper
        alpha = 1e-6
        x = alpha + (1 - 2*alpha)*data_flat
        shower = np.ma.log(x/(1-x)).filled(0)

        
        return energy_layer,shower


    flat_energy , flat_shower = preprocessing(np.nan_to_num(layer0))    
    for il, layer in enumerate([layer1,layer2]):
        energy ,shower = preprocessing(np.nan_to_num(layer))
        flat_energy = np.concatenate((flat_energy,energy),-1)
        flat_shower = np.concatenate((flat_shower,shower),-1)

    def convert_energies(e,layer_energies):
        converted = np.zeros(layer_energies.shape,dtype=np.float32)
        #CaloFlow FLow 1
        converted[:,0] = np.sum(layer_energies,-1)/np.squeeze(e)
        converted[:,1] = layer_energies[:,0]/np.sum(layer_energies,-1)
        converted[:,2] = layer_energies[:,1]/np.sum(layer_energies[:,1:],-1)
        alpha = 1e-6
        
        x = alpha + (1 - 2*alpha)*converted
        converted = np.ma.log(x/(1-x)).filled(0)
        return converted
        
    flat_energy = convert_energies(e,flat_energy)
    return np.log10(e/10.),flat_energy,flat_shower


def Calo_prep(file_path,nevts=-1):
    energy, energy_layer, energy_voxel = DataLoader(file_path,nevts)
    # print(energy.shape,energy_layer.shape, energy_voxel.shape)
    #> (99999, 1) (99999, 3) (99999, 504)

    #Let's try to learn the normalized energy depositions (energy_voxel) and the per layer normalization (energy_layer) simultaneously by concatenating the 2 datasets.
    
    tf_data = tf.data.Dataset.from_tensor_slices(np.concatenate([energy_voxel,energy_layer],-1))
    tf_cond = tf.data.Dataset.from_tensor_slices(energy)
    samples =tf.data.Dataset.zip((tf_data, tf_cond))
    nevts=energy.shape[0]
    frac = 0.8 #Fraction of events used for training
    train,test = split_data(samples,nevts,frac)
    return int(frac*nevts), int((1-frac)*nevts), train,test

def MNIST_prep(X,y,nevts=-1):
    '''
    Splits the MNIST dataset into equally sized chunks based on 
    the number of simultaneous processes to run
        
    Inputs: Data and label used for conditioning
    Outputs: Merged dataset used for training and total number of 
    training events in the split sample
        '''
    import horovod.tensorflow.keras as hvd
    hvd.init()
    if nevts<0:
        nevts = y.shape[0]

    y = y.reshape((-1,1))[hvd.rank():nevts:hvd.size()] #Split dataset equally between GPUs
    y = y.astype('float32')/10.0
    
    X = X.reshape(-1,28*28)[hvd.rank():nevts:hvd.size()]
    X = X.astype('float32')
    X+=np.random.uniform(0,1,X.shape)
    X /= 256.0

    alpha = 1e-6
    x = alpha + (1 - 2*alpha)*X
    X = np.ma.log(x/(1-x)).filled(0)
    
    # X= 2*X-1
    
    tf_data = tf.data.Dataset.from_tensor_slices(X)
    tf_cond = tf.data.Dataset.from_tensor_slices(y)
    samples =tf.data.Dataset.zip((tf_data, tf_cond)).shuffle(X.shape[0]).repeat()
    return y.shape[0],samples

def Moon_prep(X,nevts=-1):
    '''
    Splits the Double moon dataset into equally sized chunks based on 
    the number of simultaneous processes to run
        
    Inputs: Data 
    Outputs: Merged dataset with mock conditional labels to match 
    the shape expected for training and number of training events in the split sample
    '''
    import horovod.tensorflow.keras as hvd
    if nevts<0:
        nevts = X.shape[0]

    hvd.init()
    y = np.ones(X.shape[0],dtype=np.float32)
    y = y.reshape((-1,1))[hvd.rank():nevts:hvd.size()] #Split dataset equally between GPUs
    
        
    X = X[hvd.rank():nevts:hvd.size()]

    tf_data = tf.data.Dataset.from_tensor_slices(X)
    tf_cond = tf.data.Dataset.from_tensor_slices(y)
    samples =tf.data.Dataset.zip((tf_data, tf_cond)).shuffle(X.shape[0]).repeat()
    return y.shape[0],samples

    




if __name__ == "__main__":
    file_path = '/pscratch/sd/v/vmikuni/SGM/gamma.hdf5'
    energy, energy_layer, energy_voxel = DataLoader(file_path,1000)
    print(energy.shape, energy_layer.shape, energy_voxel.shape)
