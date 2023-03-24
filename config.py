import yaml

EPSILON_KEY = "epsilon"
EPSILON_DECAY_KEY = "epsilon_decay"
ALPHA_KEY = "alpha"
GAMMA_KEY = "gamma"
SAMPLING_RATE_KEY = "sampling-rate"
BUFFER_SIZE_KEY = "buffer-size"


class Config:

    def __init__(self, epsilon, alpha, epsilon_decay, gamma, sampling_rate, buffer_size):
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.alpha = alpha
        self.gamma = gamma
        self.sampling_rate = sampling_rate
        self.buffer_size = buffer_size


defaultConfig = Config(
    epsilon=0.3,
    epsilon_decay=0.99,
    alpha=0.001,
    gamma=0.90,
    sampling_rate=2,
    buffer_size=int(10e5)
)


def load_from_yaml(file_path):
    with open(file_path, 'r') as yamlfile:
        data = yaml.load(yamlfile, Loader=yaml.FullLoader)
        return Config(
            epsilon=data[EPSILON_KEY] if EPSILON_KEY in data else defaultConfig.epsilon,
            epsilon_decay=data[EPSILON_DECAY_KEY] if EPSILON_DECAY_KEY in data else defaultConfig.epsilon_decay,
            alpha=data[ALPHA_KEY] if ALPHA_KEY in data else defaultConfig.alpha,
            gamma=data[GAMMA_KEY] if GAMMA_KEY in data else defaultConfig.gamma,
            sampling_rate=data[SAMPLING_RATE_KEY] if SAMPLING_RATE_KEY in data else defaultConfig.sampling_rate,
            buffer_size=data[BUFFER_SIZE_KEY] if BUFFER_SIZE_KEY in data else defaultConfig.sampling_rate
        )
