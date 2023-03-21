class Transition:

    def __init__(self, state, action, reward, next_state, terminated):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.terminated = terminated
