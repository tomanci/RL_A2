# Deep Q Learning (DQN) CartPole

### Reinforcement Learning 2023, Master CS, Leiden University Assignment 2 on Deep Q Learning (DQN)



# Getting started

1. Use Python 3.10
2. Install all dependencies from `requirements.txt`
3. Run `python main.py -er -tn -f example-config.yaml`
4. Wait for the run to finish. You can observe the progress through the progress bars. It should take 5-60 minutes, depending on the performance of the agent.


# Reproducing experiments
Use the `-r` flag from `main.py` to specify the number of repetitions like `python main.py -er -tn -f example-config.yaml -r 3`, which defaults to 5.

## Everything

```sh
for DIR in configs/DQN/*; do python3 main.py -er -tn -d $DIR; done
```

## Specific experiments

### Sampling data
All experiments are generated from hyperparameter configurations located in `/configs`.
You can reproduce them by starting several computations in parallel.

```sh
python main.py -er -tn -f configs/DQN/epsilon.initial/epsilon.initial-1.yaml
python main.py -er -tn -f configs/DQN/epsilon.initial/epsilon.initial-2.yaml
...

or
python main.py -er -tn -d configs/DQN/epsilon.initial

or
for FILE in configs/DQN/epsilon.initial/*; do python3 main.py -er -tn -f $FILE; done
```

This will generate result files with the reward per epoch like `results/DQN/epsilon.initial/epsilon.initial-1.npy`.

### Generating plots
Using the result files, generated above, the plots can be generated using the `plotting.py` file. You can simply run the whole file with `python plotting.py`.
