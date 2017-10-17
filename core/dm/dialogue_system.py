"""
Author: Vladimir Ilievski <ilievski.vladimir@live.com>

"""

from core import constants as const
from core.env.environment import GOEnv
import core.agent.agents as agents


class GODialogSys():
    """
    The GO Dialogue System mediates the interaction between the environment and the agent.
    
    # Class members:
    
        - agent: the type of conversational agent. Default is None (temporarily).
        - env: the environment with which the agent and user interact. Default is None (temporarily).
        - act_set: the set of all intents used in the dialogue. Default is None (temporarily).
        - slot_set: the set of all slots used in the dialogue. Default is None (temporarily).
        - kb_path: path to any knowledge base
    
    """

    def __init__(self, act_set=None, slot_set=None, kb_path=None, params=None):
        """
         Constructor of the class.
        """

        # Initialize the act set and slot set
        self.act_set = act_set
        self.slot_set = slot_set
        self.kb_path = kb_path

        # create the environment
        self.env = self.__create_env(params)

        # create the specified agent type
        self.agent = self.__create_agent(params[const.AGENT_TYPE_KEY])

    def __create_env(self, params=None):
        """
        Private helper method for creating an environment given the parameters
        
        :param params: the params for creating the environment
        :return: the newly created environment
        """

        # Get all params
        simulation_mode = params[const.SIMULATION_MODE_KEY]
        is_training = params[const.IS_TRAINING_KEY]

        user_type = params[const.USER_TYPE_KEY]
        user_path = params[const.MODEL_BASED_USER_PATH_KEY]

        state_tracker_type = params[const.STATE_TRACKER_TYPE_KEY]
        dst_path = params[const.MODEL_BASED_STATE_TRACKER_PATH_KEY]

        nlu_path = params[const.NLU_PATH_KEY]
        nlg_path = params[const.NLG_PATH_KEY]

        # Create the environment
        env = GOEnv(simulation_mode, is_training, user_type, user_path, state_tracker_type, dst_path, nlu_path,
                    nlg_path)

        return env

    def __create_agent(self, agent_type_value):
        """
        Private helper method for creating an agent depending on the given type as a string.
        
        :param agent_type_str: string specifying the type of the agent
        :return: the newly created agent
        """

        agent = None

        if agent_type_value == const.AGENT_TYPE_DQN:
            agent = agents.GODQNAgent()

        return agent

    def initialize(self):
        """
        Method for initializing the dialogue.
        
        :return: 
        """
