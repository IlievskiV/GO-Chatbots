"""
Author: Vladimir Ilievski <ilievski.vladimir@live.com>

A Python file for the GO Dialogue System Processor classes
"""

from rl.core import Processor
import copy

class GOProcessor(Processor):
    """
    Class for the Goal-Oriented Processor which the mediator between the agent and the environment.
    It is processing the observation and the reward from the environment, in order to be in a correct
    format for the agent.
    
    # Arguments:
    
        - ** feasible_actions **: all feasible actions the agent might take
    """

    def __init__(self, feasible_actions=None, *args, **kwargs):
        """
         Constructor of the [GOProcessor] class
        """
        super(GOProcessor, self).__init__(*args, **kwargs)
        self.feasible_actions = feasible_actions

    def process_observation(self, observation):
        """
        Method for processing an observation from the environment. Overrides the super class method.
        
        :param observation: the observation from the environment to be processed
        :return: processed observation
        """
        # TODO
        return observation

    def process_reward(self, reward):
        """
        Method for processing a reward from the environment. Overrides the super class method.
        
        :param reward: the reward from the environment to be processed
        :return: processed observation
        """

        # TODO
        return reward

    def process_info(self, info):
        """
        Method for processing the info from the environment. Overrides the super class method.
        
        :param info: the info from the environment to be processed
        :return: processed info
        """

        # TODO
        return info

    def process_action(self, action):
        """
        Method for processing the agent action, in order to be suitable for the environment.
        Overrides the super class method.
        
        :param action: the agent action provided as a number
        :return: corresponding agent action as a dialogue act
        """

        return copy.deepcopy(self.feasible_actions[action])

    def process_state_batch(self, batch):
        """
        Method for processing an entire batch of observations. Overrides the super class method.
        
        :param batch: the batch of observations to be processed
        :return: the processed batch
        """

        return batch
