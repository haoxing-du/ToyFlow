# Continuous normalizing flows and likelihood parameter estimation

This repository contains simple scripts to run a continuous normalizing flow (FFJORD) using tensorflow probability with and without conditional inputs.

Requirements:
* Tensorflow 2.7
* Tensorflow probability 0.15 (older versions do not accept conditional inputs)

First, let's start from [FFJORD](https://arxiv.org/abs/1810.01367), in ```scripts``` run:

```bash
python toy_ffjord.py
```

The output of the acript above creates plots for the base distribution (multidimensional gaussian) and the transformed distribution that we are interested for the density estimation (double moon).

Questions:
* What is the log probability for the point (0,0.5)?
* Starting from the point (-0.1,0.6), drawn from a Normal distribution, what is the transformed coordinate in the double moon space?
* Change the code to sample events from the double moon distribution and feed those values to the inverse transformation. Does that give you back the base distribution (Normal distribution)?

# Part 2

Next step is to parametrize the density estimation for values z such that we estimate p(x|z) instead of p(x). Try to run:

```bash
python toy_conditional.py
```
The outputs of the script are plots of the target distribution (double gaussian) and the conditional target distributions.

Questions:
* Where in the base distribution are each of the double gaussians mapped to? (Hint: draw samples from each gaussian separately and run the transformation backwards)
* Imagine you started with a random point drawn from the 2-gaussian distribution. How would you determine the conditional value this point belongs to? (Hint: In the toy example, z can only assume 2 values z1, and z2, determine the probability of p(x|z1) and p(x|z2))
* Change the script such that now the conditional distribution is continuous and describes the mean of a gaussian distribution in 2D with std=1. Train the flow with this new condition and sample
* Now, imagine again that you have a data point x and want to determine the most likely value for z as p(z|x). Implement a function that maximizes lop(p(z|x)) and determine the z value. Test the algorithm by drawing different values for z and see if you get the correct answer back.

# Part 3

Now, we are going to run the scripts in parallel using the Perlmutter supercomputer! We employ data parallelization using the [Horovod](https://github.com/horovod/horovod) library to split the data used during training between multiple GPUs.

To do so we are going to use the script ```toy_parallel.py``` that implements a couple of additional features besides the parallel training. Check it out and compare with the ```toy_conditional.py``` script to see the differences.

A new feature is the implementation of convolutional neural networks as the backbone of the FFJORD implementation. Try out different examples by running the script and changing the parameter ```model_name``` to the available implementations: ```moon, mnist, calorimeter``` and running.

```bash
python train_paralle.py --model_name moon
```
Individual settings for each dataset are now defined inside dedicated ```config*.json``` files. Take a look at each of them and identify what the parameters represent within the script.

All implementaions should run out of the box and give you similar distributions as the ones we looked before. For the calorimeter implementation however there are not yet any distributions defined. Which observables are useful to compare the performance of the flow model?


Up to now we always used a single GPU during training. Let's scale this up and test how the different implementations behave as we try to use **16** GPUs at a time!

First, hop in to the Perlmutter system through an ssh connection. Once connected, move to the folder you've been using to run your experiments and request 4 iteractive nodes:

```bash
salloc -C gpu -q interactive  -t 30 -n 16 --ntasks-per-node=4  --gpus-per-task=1 -A [m3929] --gpu-bind=none
```

Change the -A flag accordingly based on the project number that you are assigned to.

After getting the allocation, you should be almost ready to go. Load any modules you've been using so far and run the script with the srun command:

```bash
module load tensorflow/2.6.0
srun python train_parallel.py --model_name moon
```

This should be enough to run using multiple processes at a time! What is the time difference for each epoch between the multi-GPU implementation and the single GPU implementation?

The last task is to investigate the calorimeter data in more detail. The preprocessing adopted mimics the one used in the [CaloFlow paper](https://arxiv.org/abs/2106.05285). As is, we use the fully-connected model as the backbone implementation for the model. However, we do know that the calorimeter can be represented as a set of 3 images, each representing a layer of the calorimeter, with dimensions: 3x96 (in layer 0), 12x12 (in layer 1), and 12x6 (in layer 2). Since the images hold geometric information of the shower shapes, we expect the density estimation based on convolutions to be better than a simple fully connected model (you can see for yourself, verify the loss value on the MNIST dataset when using a fully connected model and when using convolutional layers).

How would you implement a convolutional model for this dataset?