# Deep Q Learning (DQN) CartPole

### Reinforcement Learning 2023, Master CS, Leiden University Assignment 2 on Deep Q Learning (DQN)



# Getting started

1. Use Python 3.10
2. Install all dependencies from `requirements.txt`
3. Run `python main.py -er -tn -f example-config.yaml`
4. Wait for the run to finish. You can observe the progress through the progress bars. It should take 15-60 minutes, depending on the performance of the agent.


# Reproducing plots
All plots are generated from hyperparameter configurations located in `/configs`.
You can reproduce them by starting several computations in parallel. 

```sh
python main.py -er -tn -f configs/DQN-ER-TN/epsilon.initial/epsilon.initial-1.yaml
python main.py -er -tn -f configs/DQN-ER-TN/epsilon.initial/epsilon.initial-2.yaml
...


for FILE in configs/DQN-ER-TN/epsilon.initial/*; do python3 main.py -er -tn -f $FILE; done
```

The plots are generated using the `plotting.ipynb` notebook. You'll have to adjust the configuration parameters at the top for your specific experiment.

