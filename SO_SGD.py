# -*- coding: utf-8 -*-

# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""SGD optimizer implementation."""
# pylint: disable=g-classes-have-attributes
#rom __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.training import training_ops
from tensorflow.python.util.tf_export import keras_export


#@keras_export("keras.optimizers.SGD")
class SO_SGD(optimizer_v2.OptimizerV2):
  r"""Approximate Stabilized Hessian Gradient descent optimizer.
  Update rule for parameter `w` with gradient `g` when `momentum` is 0:
  ```python
  Undefined
  ```
  Update rule when `momentum` is larger than 0:
  ```python
  velocity = momentum * velocity - learning_rate * D_gamma_epsilon^-1 * g
  w = w * velocity
  ```
  
  Args:
    alpha: A `Tensor`, floating point value,  Defaults to 0.01.
    momentum: float hyperparameter >= 0 that accelerates gradient descent
      in the relevant
      direction and dampens oscillations. Must be between(0,1), with 0.99 as 
      default.
    gamma: A floating point value greater than epsilon. It is the maximum
      Learning rate * alpha.
    epsilon: A floating point value greater than zero. It is the minimum 
      learning rate times alpha.
    name: Optional name prefix for the operations created when applying
      gradients.  Defaults to `"SGD"`.
    **kwargs: Keyword arguments. Allowed to be one of
      `"clipnorm"` or `"clipvalue"`.
      `"clipnorm"` (float) clips gradients by norm; `"clipvalue"` (float) clips
      gradients by value.
  Usage:
      See example training script
  Reference:
      - ASHgrad paper, with formal reference added when publication accepted.
  """

  _HAS_AGGREGATE_GRAD = True

  def __init__(self,
               learning_rate=0.001,
               momentum=0.99,
               gamma=0.25/.001,
               epsilon=0.0001/0.001,
               name="SO_SGD",
               **kwargs):
    super(SO_SGD, self).__init__(name, **kwargs)
    self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
    self._set_hyper("gamma", gamma)
    self._set_hyper("momentum", momentum)
    self._set_hyper("epsilon", epsilon)
    self.nesterov = False
    self._is_first=True

  #Updated
  def _create_slots(self, var_list):
       
    for var in var_list:
        self.add_slot(var, "g_mom")
    for var in var_list:
        self.add_slot(var,'dh')
    for var in var_list:
        self.add_slot(var,'update')

  def _prepare_local(self, var_device, var_dtype, apply_state):
    super(SO_SGD, self)._prepare_local(var_device, var_dtype, apply_state)
    apply_state[(var_device, var_dtype)]["momentum"] = array_ops.identity(
        self._get_hyper("momentum", var_dtype))
    apply_state[(var_device, var_dtype)]["gamma"] = array_ops.identity(
        self._get_hyper("gamma", var_dtype))
    apply_state[(var_device, var_dtype)]["epsilon"] = array_ops.identity(
        self._get_hyper("epsilon", var_dtype))

  def _resource_apply_dense(self, grad, var, apply_state=None):
    var_device, var_dtype = var.device, var.dtype.base_dtype
    coefficients = ((apply_state or {}).get((var_device, var_dtype))
                    or self._fallback_apply_state(var_device, var_dtype))
    var_dtype = var.dtype.base_dtype
    alpha=self._get_hyper('learning_rate',var_dtype)
    
    if self._is_first:
        momentum=grad
        H_inv=tf.ones(var.shape,var_dtype)
        self.get_slot(var,'dh').assign(H_inv)
        self._is_first=False
    else:        
        momentum=self.get_slot(var,'g_mom')
        momentum_new=self._get_hyper('momentum',var_dtype)*momentum+(1-self._get_hyper('momentum',var_dtype))*grad
        momentum.assign(momentum_new)
        
        dh=self.get_slot(var,'dh')
        ones=tf.ones(var.shape)
        H_inv=tf.maximum(tf.math.reciprocal_no_nan(dh),tf.multiply(ones,self._get_hyper('epsilon',var_dtype)/alpha))
        H_inv=tf.minimum(H_inv,tf.multiply(ones,self._get_hyper('gamma',var_dtype)/alpha))            
    update=alpha*H_inv*momentum
    self.get_slot(var,'update').assign(update)
    
    return training_ops.resource_apply_gradient_descent(
          var.handle, 1.0, update, use_locking=self._use_locking)


  def update(self,grad,grad_post,dh,dh_post,var_set):
      """
      This function updates the hessian approximation:
        grad: the gradient estimate for the batch before backprop
        grad_post: the gradient estimate for the batch after backprop
        df: the diagonal estimate of the second derivatives before
          backprop
        df_post: the diagonal estimate of the second derivatives after 
          backprop
        var_set: The parameter set          
      """
      
      for i in range(0,len(var_set)):
          H=self.get_slot(var_set[i],'dh')
          update=self.get_slot(var_set[i],"update")
          tmp=tf.math.multiply(tf.subtract(grad[i],grad_post[i]),tf.math.reciprocal_no_nan(update))
          tmp2=tf.add(tf.add(dh[i],dh_post[i]),tmp)
          adj=tf.multiply(1.0/3.0,tmp2)
          H_new=self._get_hyper('momentum')*H+(1-self._get_hyper('momentum'))*adj
          H.assign(H_new)
          
          momentum=self.get_slot(var_set[i],'g_mom')
          tmp=tf.multiply(0.5, tf.subtract(grad_post[i], grad[i]))
          momentum_new=self._get_hyper('momentum')*momentum+(1-self._get_hyper('momentum'))*tmp
          #momentum.assign(momentum_new)
          

  def _resource_apply_sparse_duplicate_indices(self, grad, var, indices, **kwargs):
      raise NotImplementedError
   

  def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
      raise NotImplementedError

  def get_config(self):
    config = super(SO_SGD, self).get_config()
    config.update({
        "learning_rate": self._serialize_hyperparameter("learning_rate"),
        "gamma": self._serialize_hyperparameter("gamma"),
        "momentum": self._serialize_hyperparameter("momentum"),
        "epsilon": self._serialize_hyperparameter("epsilon"),
        "nesterov": self.nesterov,
    })
    return config