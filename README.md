## Current status of the Project:

functional backbone/outline of the program
- we tried to use Jax for very long, however it kept showing us errors, we didn't really know how to fix -> therefore we now cut it out completely
- The loss function uses very simply lagrange multipliers in order to maximize Tz and Tx under the constraint that eta should be held constant
- We use the SepCMA optimizer to do Reinforcement Learning on this cost function

optimization works in theory
- however due to the lack of optimization yet, it is practically non feasible


## ideas for improvements we tried:

Improvement of the Tx and Tz calculation:
- we tried to use the q and p quadratures to measure the exponential decay of Z. X decay we measure just with pairity expectation value. We then defined a cost function that is quadratic in the difference between the desired ratio and Tz/Tx and scales only linearly with the magnitude of Tz and Tx. We prepare the state in a coherent state to measure the Z decay and a cat state to measure X decay.

- The way we would measure q and p in practice would be through characteristic function tomography. There are ways to do this.
Direct Characteristic-Function Tomography of Quantum States of the
Trapped-Ion Motional Oscillator”, Phys. Rev. Lett. 125, 043602 (2020).
Assuming our cat state manifold will be alligned along the x axis, we can take the derivative in the orthogonal direction
to obtain the expectation of q. 
this should decay at the same rate as \<sigmaz\> so we can use it to measure Tz by doing an exponential fit.
It however does not require a good estimate of alpha and can be obtained through sampling in the fourier space around 0

different fits:
- here we also had some issues with jax and compatibility


## next steps:

We would like to implement and exponential fit that is jax compatible so we can properly batch our const function evaluaiton.
We would also tune the hyperparameters and compare various optimizers against each other.

