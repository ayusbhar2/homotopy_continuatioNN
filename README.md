# Investigating Optimization Landscape of Deep Linear Networks using Homotopy Continuation Methods

This code is meant to generate and solve *gradient polynomial systems* arising out of deep linear (identity activation) neural networks as described in this thesis - https://github.com/ayusbhar2/mathematics_masters_thesis.

For a particular network architecture, the solutions to the gradient polynomial system are precisely the complex critical points of the loss function of the network.

Each instance handles a specific network architecture. Multiple instances can be created in parallel. The code running on each instance randomly generates polynomial systems for the specified architecture and records the solutions as well as metadata about the solutions.


Setup:

1. You should have Amazon CLI set up with permissions to create and access instances remotely from your computer.

2. Clone this repository and add the environment variables specified at the top of `deploy.sh`.


Usage:

1. Specify the network architecture in `config.json`. `H` = number of layers, `di` = number of neurons in each layer, `m` = number of training examples to be generated, `dx` = length of the input vector, `dy` = length of the output vector. `runcount` specifies the number of random systems to be generated for each architecture. Leave other entries as is unless you know what you are doing.

2. Next you need to run `deploy.sh` script. This script takes three arguments
    1) the type of the ec2 instance (e.g. `c5.4xlarge`)
    2) number of instances you want to launch
    3) appropriate tag for your instances (e.g. `di1H1m1dx1dy1_noconv`). Tags group your instances by the architecture type and make is easier to track experiments and collect results.
    
    cd into the `homotopy_continuatioNN` directory and run

        bash deploy.sh <TYPE> <COUNT> <TAG>

3. Once the instances are generated and your code is successfully shipped you can monitor the progress of your task through the cpu usage monitoring provided by AWS. You could also `ssh` into the instance for more detailed monitoring.

4. Once the taks has finished running on an instance, you can collect the results by running ```bash collect.sh <TAG>```. The output of the task will be downloaded into the `homotopy_continuatioNN/analysis/cloud_output/` directory. Debug logs will be downloaded in the `homotopy_continuatioNN/analysis/cloud_logs` directory.

5. You can use the `MATH899_Research.ipynb` notebook to analyze the results.

