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
The outputs of the script are plots of the target distribution (double gaussian) and the conditional target distributions
Questions:
* Where in the base distribution are each of the double gaussians mapped to? (Hint: draw samples from each gaussian separately and run the transformation backwards)
* Imagine you started with a random point drawn from the 2-gaussian distribution. How would you determine the conditional value this point belongs to? (Hint: In the toy example, z can only assume 2 values z1, and z2, determine the probability of p(x|z1) and p(x|z2))
* Change the script such that now the conditional distribution is continuous and describes the mean of a gaussian distribution in 2D with std=1. Train the flow with this new condition and sample
* Now, imagine again that you have a data point x and want to determine the most likely value for z as p(z|x). Implement a function that maximizes lop(p(z|x)) and determine the z value. Test the algorithm by drawing different values for z and see if you get the correct answer back.
