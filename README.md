# ASHgrad
Approximate Stabilized Hessian Gradient Descent

This repository gives the code for the ASH grad optimization algorithm (SO_SGD) file.  Along with this file is an example training script. Both scripts are based on tensorflow 2.3.1 library. 

# Running:
To run the code, ensure that both scripts are in the same directory.
Run the **tf_trainer_ASHgrad.py** script.  This will train the EB1 architecture on CIFAR10 using ASHgrad. The final result table will be stored in the **final** variable, where the columns are {epoch number, training loss, training accuracy, test loss, test accuracy}.

# Alternative architectures:
To use ResNet uncomment lines 24-26 and comment lines 27-29.
To run with other EB architectures change **EfficientNetB1** on line 27 to **.EfficientNetBX** where **x** is the number of the EB architecture.

# Alternative Datasets:
On lines 39 and 43 change **'cifar10'** to the desired data set.  Be sure to use data sets which are consistent with the intended architecture.

# Alternative Optimizer:
The script is currently set up to run ASHgrad. However the paper compares the performance of several other optimization algorithms. To change the optimization algorithm do the following.
1) Comment line 49 (the ASHgrad optimizer)
2) Uncomment one of line 50 or 51 (ADAM or SGD)
3) Comment lines 63-87, this is the training process for ASHgrad.
4) Uncomment lines 93-110, this is the training process for the other algorithms.

# PreTrained Models
This repository does not contain pretrained models.  The purpose of the repository is to present the ASHgrad optimizer and provide example training code.

# Evaluation
The paper presents graphics comparing the test performance of ASHgrad versus SGD with two learning rates and ADAM.  These graphics were prepared by saving the **final** variable from each run into a csv file (see example for format). The **plotter** script produces the graphics. 
