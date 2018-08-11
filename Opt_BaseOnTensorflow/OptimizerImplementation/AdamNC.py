# Date: 2018-08-11 11:20
# Author: Enneng Yang
# Abstract：AdamNc

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.training import optimizer
from tensorflow.python.training import training_ops
from tensorflow.python.util.tf_export import tf_export
from tensorflow.python.ops import variable_scope

class AdamNCOptimizer(optimizer.Optimizer):

    '''
    implement AdamNC. ref:https://openreview.net/pdf?id=ryQu7f-RZ
                          https://github.com/andrehuang/Generalized-AdamNC/blob/master/adamnc.py
    nitialization:
           m_0 <- 0 (1st moment vector)
           v_0 <- 0 (2st moment vector)
           t   <- 0   (timeStep)

    update rule:
           t        <- t + 1
           lr_t     <- learning_rate * sqrt(1 - beta2^t) / (1 - beta1^t)
           m_t      <- beta1 * m_{t-1} + (1 - beta1) * g
           v_t      <- beta2 * v_{t-1} + (1 - beta2) * g * g
           variable <- variable - lr_t * m_t / (sqrt(v_t) + epsilon)
       '''

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.99, epsilon=1e-8,
                 use_locking=False, name="AdamNC"):
        super(AdamNCOptimizer,self).__init__(use_locking=use_locking,name=name)
        self._lr = learning_rate
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon

        # Tensor versions of the constructor arguments, created in _prepare().
        self._lr_t = None
        self._beta1_t = None
        self._beta2_t = None
        self._epsilon_t = None

    def _create_slots(self, var_list):
        first_var = min(var_list, key= lambda x: x.name)

        create_new = self._beta1_power is None
        if not create_new and context.in_eager_mode():
            create_new = (self._beta1_power.graph is not first_var.graph)

        if create_new:
            with ops.colocate_with(first_var):
                self._beta1_power = variable_scope.variable(self._beta1, name="beta1_power", trainable=False)
                self._beta2_power = variable_scope.variable(self._beta2, name="beta2_power", trainable=False)

        for v in var_list:
            self._zeros_slot(v, "m", self._name)
            self._zeros_slot(v, "v", self._name)

    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")
        self._beta1_t = ops.convert_to_tensor(self._beta1, name="beta1")
        self._beta2_t = ops.convert_to_tensor(self._beta2, name="beta2")
        self._epsilon_t = ops.convert_to_tensor(self._epsilon, name="epsilon")

    def _apply_dense(self, grad, var):
        beta1_power = math_ops.cast(self._beta1_power, var.dtype.base_dtype)
        beta2_power = math_ops.cast(self._beta2_power, var.dtype.base_dtype)
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
        beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
        epsilon_t = math_ops.cast(self._epsilon_t, var.dtype.base_dtype)

        # lr = (lr_t * math_ops.sqrt(1 - beta2_power) / (1 - beta1_power))   # Adam
        lr = lr_t #AdanNC

        # m_t = beta1 * m + (1 - beta1) * g_t
        m = self.get_slot(var, "m")
        m_scaled_g_values = grad * (1 - beta1_t)
        m_t = state_ops.assign(m, m * beta1_t, use_locking=self._use_locking)

        # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
        v = self.get_slot(var, "v")
        v_scaled_g_values = (grad * grad) * (1 - beta2_t)
        v_t = state_ops.assign(v, v * beta2_t, use_locking=self._use_locking)

        v_sqrt = math_ops.sqrt(v_t)
        var_update = state_ops.assign_sub(var, lr*m_t/(v_sqrt+epsilon_t),use_locking=self._use_locking)

        return control_flow_ops.group(*[var_update, m_t, v_t])



