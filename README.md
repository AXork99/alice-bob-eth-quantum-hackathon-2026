## Current status of the Project:

functional backbone/outline of the program
- we tried to use Jax for very long, however it kept showing us errors, we didn't really know how to fix -> therefore we now cut it out completely
- The loss function uses very simply lagrange multipliers in order to maximize Tz and Tx under the constraint that eta should be held constant

optimization works in theory
- however due to the lack of optimization yet, it is practically non feasible



## ideas for improvements we tried:

Improvement of the Tx and Tz calculation:
- we tried to use the x and p quadratures to measure the exponential decay of Z. X decay we measure just with pairity expectation value. We then defined a cost function that is quadratic in the difference between the desired ratio and Tz/Tx and scales more slowly with the magnitude of Tz and Tx.

different fits:
- here we also had some issues with jax and compatibility


## next steps: