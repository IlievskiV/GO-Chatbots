from rl.agents.dqn import DQNAgent


class GOChatBotDQNAgent(DQNAgent):
    """ Class for the Goal-Oriented Chatbots with a DQN-based policy learning.
    
    This class is extending the class AbstractDQNAgent from keras-rl framework.
    This type of DQN agent should learn the policies of a dialogue in a given environment.
    
    The following methods are implemented:
    
    - `forward`
    - `backward`
    - `compile`
    - `load_weights`
    - `save_weights`
    - `layers`
    
    
    # Arguments
    
    From [Agent](#agent) class:
        - processor (`Processor` instance): See [Processor](#processor) for details.
    
    From [AbstractDQNAgent] class:
        - nb_actions: the number of all possible actions.
        - memory:
        - gamma: the discount reward factor. Default is 0.99, the agent remembers everything.
        - batch_size: the number of memories to replay in one training epoch. Default is 32.
        - nb_steps_warmup: the number of steps needed to warm up and fill the memory replay buffer. Default is 1000.
        - train_interval:
        - memory_interval:
        - target_model_update: after how many steps, the target DQN gets updated. Default is 
    
    """

    def __init__(self, *args, **kwargs):
        super(GOChatBotDQNAgent).__init__(*args, **kwargs)




