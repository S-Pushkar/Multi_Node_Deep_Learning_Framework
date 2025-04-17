# Multi_Node_Deep_Learning_Framework

To install the dependencies, run the following (On Ubuntu based OS):
```bash
sudo apt-get install libyaml-cpp-dev
```

The model parameters are stored in a YAML file. The following is an example of such a file:
```yaml
num_threads: 4
num_hidden_layers: 2
neurons_per_hidden: [4, 3]
learning_rate: 0.1
activation_functions: ["relu", "tanh", "linear"]
num_epochs: 10
dataset_files: ["./dataset/data1.csv", "./dataset/data2.csv", "./dataset/data3.csv", "./dataset/data4.csv"]
```

To compile and run the code:
```bash
make
./run.sh <config YAML file>
```

### Note:

- If you don't use the run.sh script, you need to set the following environment variables:

```bash
export OMP_NUM_THREADS=4
export OMP_PROC_BIND=close
```

- Not setting the environment variables may lead to performance degradation.

<br>

To compile manually:
```bash
g++ -std=c++17 -fopenmp main.cpp neural_network.cpp -lyaml-cpp -o deep_learning_framework
```