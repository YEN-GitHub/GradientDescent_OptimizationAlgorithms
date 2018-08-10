# Gradient Descent Optimization Algorithms
## Introduction: 
 Save my gradient descent optimization algorithms code. <br>
 
 SG improved in three ways: <br>
 > Direction1-Noise Reduction Method <br>
 > Direction2-Variance Reduction Method (SG have residual, can't use fixed stepsize) <br>
 > Direction3-Second Order Method (escaping saddle point) <br>

## Algorithm list：
### Basic Algorithm Implementation
>Basic-GradientDescentVariants:<br>
>> {Batch Gradient Descent } Algorithm: [BatchGradientDescent.py](https://github.com/YEN-GitHub/StochasticGradient_Algorithm/tree/master/Basic-GradientDescentVariants/BatchGradientDescent.py) <br>
>> {Stochastic Gradient Descent } Algorithm: [StochasticGradientDescent.py](https://github.com/YEN-GitHub/StochasticGradient_Algorithm/tree/master/Basic-GradientDescentVariants/StochasticGradientDescent.py) <br>
>> {Mini-Batch Gradient Descent } Algorithm: [Mini-Batch_GradientDescent.py](https://github.com/YEN-GitHub/StochasticGradient_Algorithm/tree/master/Basic-GradientDescentVariants/Mini-Batch_GradientDescent.py) <br>

> Direction1-Noise Reduction Method: <br>
>> { ISTA } Algorithm: [ISTA.py](https://github.com/YEN-GitHub/GradientDescent_OptimizationAlgorithms/tree/master/Direction1-NoiseReductionMethod/ISTA.py) <br>
>> { FISTA } Algorithm: [FISTA.py](https://github.com/YEN-GitHub/GradientDescent_OptimizationAlgorithms/tree/master/Direction1-NoiseReductionMethod/FISTA.py) <br>

> Direction2-Variance Reduction Method:<br>
>> {  momentum stochastic gradient descent } Algorithm: [MomentumStochasticGradientDescent.py](https://github.com/YEN-GitHub/GradientDescent_OptimizationAlgorithms/tree/master/Direction2-VarianceReductionMethod/MomentumStochasticGradientDescent.py) <br>
>> { Nesterov Accelerated Gradient } Algorithm: [NesterovAcceleratedGradient.py](https://github.com/YEN-GitHub/GradientDescent_OptimizationAlgorithms/tree/master/Direction2-VarianceReductionMethod/NesterovAcceleratedGradient.py) <br>
>> { Adagrad } Algorithm: [Adagrad.py](https://github.com/YEN-GitHub/GradientDescent_OptimizationAlgorithms/tree/master/Direction2-VarianceReductionMethod/Adagrad.py) <br>
>> { Adadelta } Algorithm: [Adadelta.py](https://github.com/YEN-GitHub/GradientDescent_OptimizationAlgorithms/tree/master/Direction2-VarianceReductionMethod/Adadelta.py) <br>
>> { RMSprop } Algorithm: [RMSprop.py](https://github.com/YEN-GitHub/GradientDescent_OptimizationAlgorithms/tree/master/Direction2-VarianceReductionMethod/RMSprop.py) <br>
>> { Adam } Algorithm: [Adam.py](https://github.com/YEN-GitHub/GradientDescent_OptimizationAlgorithms/tree/master/Direction2-VarianceReductionMethod/Adam.py) <br>
>> { AdaMax } Algorithm: [AdaMax.py](https://github.com/YEN-GitHub/GradientDescent_OptimizationAlgorithms/tree/master/Direction2-VarianceReductionMethod/AdaMax.py) <br>

> Direction3-Second Order Method:<br>

### Used Tensorflow optimizer 
>> { Optimizer method compare} Algorithm: [Optimizer_Comparison.py](https://github.com/YEN-GitHub/GradientDescent_OptimizationAlgorithms/tree/master/Opt_BaseOnTensorflow/Optimizer_Comparison.py) <br>

> Basic-GradientDescentVariants:<br>
>> { Batch Gradient Descent } Algorithm: [BatchGradientDescent.py](https://github.com/YEN-GitHub/StochasticGradient_Algorithm/tree/master/Opt_BaseOnTensorflow/Basic-GradientDescentVariants/BatchGradientDescent.py) <br>
>> { Stochastic Gradient Descent } Algorithm: [StochasticGradientDescent.py](https://github.com/YEN-GitHub/StochasticGradient_Algorithm/tree/master/Opt_BaseOnTensorflow/Basic-GradientDescentVariants/StochasticGradientDescent.py) <br>
>> { Mini-Batch Gradient Descent } Algorithm: [Mini-Batch_GradientDescent.py](https://github.com/YEN-GitHub/StochasticGradient_Algorithm/tree/master/Opt_BaseOnTensorflow/Basic-GradientDescentVariants/Mini-Batch_GradientDescent.py) <br>

> Direction 2: Variance Reduction Method Based on Tesorflow
>> { momentum stochastic gradient descent } Algorithm: [Momentum.py](https://github.com/YEN-GitHub/GradientDescent_OptimizationAlgorithms/tree/master/Opt_BaseOnTensorflow/Direction2-VarianceReductionMethod/Momentum.py) <br>
>> { Nesterov Accelerated Gradient } Algorithm: [Nesterov.py](https://github.com/YEN-GitHub/GradientDescent_OptimizationAlgorithms/tree/master/Opt_BaseOnTensorflow/Direction2-VarianceReductionMethod/Nesterov.py) <br>
>> { Adadelta } Algorithm: [Adadelta.py](https://github.com/YEN-GitHub/GradientDescent_OptimizationAlgorithms/tree/master/Opt_BaseOnTensorflow/Direction2-VarianceReductionMethod/Adadelta.py) <br>
>> { Adagrad } Algorithm: [Adagrad.py](https://github.com/YEN-GitHub/GradientDescent_OptimizationAlgorithms/tree/master/Opt_BaseOnTensorflow/Direction2-VarianceReductionMethod/Adagrad.py) <br>
>> { RMSprop } Algorithm: [RMSprop.py](https://github.com/YEN-GitHub/GradientDescent_OptimizationAlgorithms/tree/master/Opt_BaseOnTensorflow/Direction2-VarianceReductionMethod/RMSprop.py) <br>
>> { Adam } Algorithm: [Adam.py](https://github.com/YEN-GitHub/GradientDescent_OptimizationAlgorithms/tree/master/Opt_BaseOnTensorflow/Direction2-VarianceReductionMethod/Adam.py) <br>
>> { AdaMax } Algorithm: [AdaMax.py](https://github.com/YEN-GitHub/GradientDescent_OptimizationAlgorithms/tree/master/Opt_BaseOnTensorflow/Direction2-VarianceReductionMethod/AdaMax.py) <br>

### Optimizer Implementation Based on Tesorflow
>> { PowerSign } Optimizer: [PowerSign.py](https://github.com/YEN-GitHub/StochasticGradient_Algorithm/tree/master/Opt_BaseOnTensorflow/OptimizerImplementation/PowerSign.py) <br>
>> { AddSign } Optimizer: [AddSign.py](https://github.com/YEN-GitHub/StochasticGradient_Algorithm/tree/master/Opt_BaseOnTensorflow/OptimizerImplementation/AddSign.py) <br>
>> { Test Optimizer } Algorithm: [TestOptimizer.py](https://github.com/YEN-GitHub/StochasticGradient_Algorithm/tree/master/Opt_BaseOnTensorflow/OptimizerImplementation/TestOptimizer.py) <br>

## References
> Paper:  `Optimization Methods for Large-Scale Machine Learning`  <br>
> Paper:  `An Overview of Gradient Descent Optimization Algorithms`  <br>
> Blog: `Benoit Descamps:Custom Optimizer in TensorFlow`  <br>
> Github user: `tsycnh/mlbasic`  <br>