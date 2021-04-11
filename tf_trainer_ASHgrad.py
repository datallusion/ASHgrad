# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 09:56:50 2020

@author: Jared
"""


import tensorflow as tf
import numpy as np
from datetime import datetime
import SO_SGD
import tensorflow_datasets as tfds



tf.config.experimental_run_functions_eagerly(True)

strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce(), devices = ["/gpu:0", "/gpu:1"])
batch_size=50
with strategy.scope():
    ###########################
    #Select a model by uncommenting one of the models below.
    #model= tf.keras.applications.ResNet50(
    #        include_top=True, weights=None, input_tensor=None,
    #        input_shape=(32,32,3), pooling=None, classes=10)
    model= tf.keras.applications.EfficientNetB1(
             include_top=True, weights=None, input_tensor=None,
             input_shape=(32,32,3), pooling=None, classes=10)    

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
    
    def compute_loss(labels, predictions):
        per_example_loss = loss_object(labels, predictions)
        return tf.nn.compute_average_loss(per_example_loss, global_batch_size=batch_size)
    
    ##################
    ##Load data  
    train_data=tfds.load('cifar10', split='train', as_supervised=True).prefetch(tf.data.experimental.AUTOTUNE).cache()
    train_data=train_data.batch(batch_size)
    train_data = strategy.experimental_distribute_dataset(train_data)
    
    test_data=tfds.load('cifar10', split='test', as_supervised=True).prefetch(tf.data.experimental.AUTOTUNE).cache()
    test_data=test_data.batch(batch_size)
    test_data = strategy.experimental_distribute_dataset(test_data)
    
    ##################
    #Select an optimizer by uncommenting one of the below
    optimizer = SO_SGD.SO_SGD()
    #optimizer = tf.keras.optimizers.Adam(  )
    #optimizer = tf.keras.optimizers.SGD(learning_rate=0.001  )
    
    #train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

#######################
#If running ASHgrad uncomment the trainstep function below, and comment out
#the other one.    
#SOSGD    
@tf.function  
def train_step(inputs):
    images, labels = inputs
    images = images/ np.float32(255)
    with tf.GradientTape() as D:
        with tf.GradientTape() as D2:
          predictions = model(images, training=True)
          loss = compute_loss(labels, predictions)
        gradients = D2.gradient(loss, model.trainable_variables)
    dhess = D.gradient(gradients, model.trainable_variables)
    
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    with tf.GradientTape() as D2_post:
        with tf.GradientTape() as D_post:
          predictions_post = model(images, training=True)
          loss_post = loss_object(labels, predictions_post)
    gradients_post = D_post.gradient(loss_post, model.trainable_variables)
    dhess_post = D2_post.gradient(loss_post, model.trainable_variables)
    
    optimizer.update(gradients,gradients_post,dhess,dhess_post,model.trainable_variables)
    
    train_loss = compute_loss(labels, predictions)
    train_accuracy.update_state(labels, predictions)
    return train_loss

#######################
#If not running ASHgrad uncomment the trainstep function below, and comment out
#the other one.    
#ADAM/SGD
# def train_step(inputs):
#     images, labels = inputs
#     images = images/ np.float32(255)
#     with tf.GradientTape() as D:
#         with tf.GradientTape() as D2:
#           # training=True is only needed if there are layers with different
#           # behavior during training versus inference (e.g. Dropout).
#           predictions = model(images, training=True)
#           loss = compute_loss(labels, predictions)
#         gradients = D2.gradient(loss, model.trainable_variables)
#     dhess = D.gradient(gradients, model.trainable_variables)
    
#     optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
#     train_loss = compute_loss(labels, predictions)
#     train_accuracy.update_state(labels, predictions)  
#     return train_loss
      
@tf.function    
def test_step(inputs):
    images, labels = inputs
    images = images/ np.float32(255)

    predictions = model(images, training=False)
    t_loss = loss_object(labels, predictions)
    test_loss.update_state(t_loss)
    test_accuracy.update_state(labels, predictions)
    return test_loss
      

@tf.function
def distributed_train_step(dataset_inputs):
  per_replica_losses = strategy.run(train_step, args=(dataset_inputs,))

  return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

@tf.function
def distributed_test_step(dataset_inputs):
  return strategy.run(test_step, args=(dataset_inputs,))    



print('test')
########################
##Test the initial accuracy
test_loss.reset_states()
test_accuracy.reset_states()
for test_sample in test_data:
    distributed_test_step(test_sample)
print('/test')
res=[]
template = ("Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}")    
template2 = ("{},{},{},{},{}")  
print(
  f'Epoch {0}, '
  f'Loss: {0}, '
  f'Accuracy: {0}, '
  f'Test Loss: {test_loss.result()}, '
  f'Test Accuracy: {test_accuracy.result() * 100}'
)

res.append(template2.format(0, 0, 0, test_loss.result(), test_accuracy.result()*100))
  
EPOCHS = 100

for epoch in range(EPOCHS):
  print(epoch)
  # Reset the metrics at the start of the next epoch
  #train_loss.reset_states()
  train_accuracy.reset_states()
  test_loss.reset_states()
  test_accuracy.reset_states()
  mom=None
  hess=None
  count=0
  ##########################
  ##Train
  tot_loss=0
  num_batch=0
  for train_sample in train_data:
    count+=1
    if count%10==0:
        datetime.now().strftime("%H:%M:%S")
    
    tot_loss+=distributed_train_step(train_sample)
    num_batch+=1
  ##########################
  ##Check Errors
  for test_sample in test_data:
    distributed_test_step(test_sample)
    
  template = ("Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}")
  print(template.format(epoch+1, tot_loss/num_batch, train_accuracy.result()*100, test_loss.result(), test_accuracy.result()*100))
  res.append(template2.format(epoch+1, tot_loss/num_batch, train_accuracy.result()*100, test_loss.result(), test_accuracy.result()*100))

final=np.zeros((len(res),5))
for i in range(0,len(res)):
    final[i,:]=res[i].split(',')
