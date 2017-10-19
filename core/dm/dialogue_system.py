"""
Author: Vladimir Ilievski <ilievski.vladimir@live.com>

"""

from core import constants as const
from core.env.environment import GOEnv
import core.agent.agents as agents
from core.agent.processor import GOProcessor


class GODialogSys():
    """
    The GO Dialogue System mediates the interaction between the environment and the agent.
    
    # Class members:
    
        - agent: the type of conversational agent. Default is None (temporarily).
        - env: the environment with which the agent and user interact. Default is None (temporarily).
        - act_set: the set of all intents used in the dialogue.
        - slot_set: the set of all slots used in the dialogue.
        - kb_path: path to any knowledge base
        - agt_feasible_actions: list of templates described as dictionaries, corresponding to each action the agent might take
                                (dict to be specified)
        - max_nb_turns: the maximal number of dialogue turns
    
    """

    def __init__(self, act_set=None, slot_set=None, agt_feasible_actions=None, params=None):
        """
         Constructor of the class.
        """

        # Initialize the act set and slot set
        self.act_set = act_set
        self.slot_set = slot_set

        # the path to the knowledge base
        self.kb_path = params[const.KB_PATH_KEY];

        # the actions the agent might take
        self.agt_feasible_actions = agt_feasible_actions

        self.max_nb_turns = params[const.MAX_NB_TURNS]

        # create the environment
        self.env = self.__create_env(params, act_set, slot_set, agt_feasible_actions)

        # create the specified agent type
        self.agent = self.__create_agent(params[const.AGENT_TYPE_KEY])

    def __create_env(self, params=None, act_set=None, slot_set=None, agt_feasible_actions=None):
        """
        Private helper method for creating an environment given the parameters.
        
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

        max_nb_turns = params[const.MAX_NB_TURNS]

        nlu_path = params[const.NLU_PATH_KEY]
        nlg_path = params[const.NLG_PATH_KEY]

        # Create the environment
        env = GOEnv(simulation_mode, is_training, user_type, user_path, state_tracker_type, dst_path, act_set, slot_set,
                    agt_feasible_actions, max_nb_turns, nlu_path, nlg_path)

        return env

    def __create_agent(self, agent_type_value):
        """
        Private helper method for creating an agent depending on the given type as a string.
        
        :param agent_type_str: string specifying the type of the agent
        :return: the newly created agent
        """

        agent = None

        if agent_type_value == const.AGENT_TYPE_DQN:
            go_processor = GOProcessor(feasible_actions=self.agt_feasible_actions)
            agent = agents.GODQNAgent(processor=go_processor)

        return agent

    def initialize(self):
        """
        Method for initializing the dialogue.
        
        :return: 
        """
