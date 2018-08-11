# Date: 2018-08-10 20:49
# Author: Enneng Yang
# Abstract： AMSgrad ref: https://openreview.net/pdf?id=ryQu7f-RZ
#                         https://github.com/taki0112/AMSGrad-Tensorflow

from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.training import optimizer

class AMSGradOptimizer(optimizer.Optimizer):

    '''
    Implementation of AMSgrad ref: https://openreview.net/pdf?id=ryQu7f-RZ
    '''
    def __init__(self, learning_rate=0.01, beta1=0.9, beta2=0.99, epsilon=1e-8, use_locking=False, name="AMSGrad"):
        super(AMSGradOptimizer, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon

        #  Tensor versions of the constructor arguments, created in _prepare().
        self._lr_t = None
        self._beta1_t = None
        self._beta2_t = None
        self._epsilon_t = None

        self._beta1_power = None
        self._beta2_power = None


    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr)
        self._beta1_t = ops.convert_to_tensor(self._beta1)
        self._beta2_t = ops.convert_to_tensor(self._beta2)
        self._epsilon_t = ops.convert_to_tensor(self._epsilon)


    def _create_slots(self, var_list):
        first_var = min(var_list, key=lambda x: x.name)

        create_new = self._beta1_power is None
        if not create_new and context.in_eager_mode():
            create_new = (self._beta1_power.graph is not first_var.graph)

        if create_new:
            with ops.colocate_with(first_var):
                self._beta1_power = variable_scope.variable(self._beta1, name="beta1_power", trainable=False)
                self._beta2_power = variable_scope.variable(self._beta2, name="beta2_power", trainable=False)

        # Create slots for the first and second moments
        for v in var_list:
            self._zeros_slot(v, "m", self._name)
            self._zeros_slot(v, "v", self._name)
            self._zeros_slot(v, "vhat", self._name)


    def _apply_dense(self, grad, var):
        beta1_power = math_ops.cast(self._beta1_power, var.dtype.base_dtype)
        beta2_power = math_ops.cast(self._beta2_power, var.dtype.base_dtype)
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
        beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
        epsilon_t = math_ops.cast(self._epsilon_t, var.dtype.base_dtype)

        lr = (lr_t * math_ops.sqrt(1-beta2_power) / (1-beta1_power))

        # m_t = beta1 * m + (1-beta1) * g_t
        m = self.get_slot(var, "m")
        m_scaled_g_value = grad * (1-beta1_t)
        m_t = state_ops.assign(m, beta1_t * m + m_scaled_g_value, use_locking=self._use_locking)

        # v_t = beta2 *v + (1-beta2)*(g_t*g_t)
        v = self.get_slot(var, "v")
        v_scaled_g_value = (grad * grad)* (1-beta2_t)
        v_t = state_ops.assign(v, beta2_t * v + v_scaled_g_value)

        #amsgrad
        vhat = self.get_slot(var, "vhat")
        vhat_t = state_ops.assign(vhat, math_ops.maximum(v_t, vhat))
        v_sqart = math_ops.sqrt(vhat_t)

        var_update = state_ops.assign_sub(var, lr * m_t / (v_sqart + epsilon_t), use_locking=self._use_locking)

        return control_flow_ops.group(*[var_update, m_t, v_t, vhat_t])

    def _apply_sparse(self, grad, var):
        return self._apply_dense(grad, var)
