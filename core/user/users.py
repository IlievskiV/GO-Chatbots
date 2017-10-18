"""
Author: Vladimir Ilievski <ilievski.vladimir@live.com>

"""

from core import constants as const


class GOUser:
    """
    Abstract Base Class of all type of Goal-Oriented conversational users.
    
    # Class members:
        - id: the id of the user
        - current_turn_nb: the number of the current dialogue turn
        - history: a list of the user conversation history
        - simulation_mode: semantic frame or natural language sentence form of user utterances
        - goal: the goal of the user
    """

    def __init__(self, id=None, simulation_mode=None, goal_set=None):
        # Initialize the class members
        self.id = id
        self.current_turn_nb = 0
        self.history = []
        self.simulation_mode = simulation_mode
        self.goal_set = goal_set

    def reset(self):
        """
        Abstract method for restarting the user and getting the initial user action.
        
        :return: The initial user action
        """
        raise NotImplementedError()

    def step(self, agt_action):
        """
        Abstract method for getting the next user utterance given the last agent action.
        
        :param agt_action: last agent action
        :return: next user action
        """
        raise NotImplementedError()


class GORealUser(GOUser):
    """
    Class connecting a real user, writing on the standard input. Extends the [GOUser] class.
    
    # Class members:
    """

    def __init__(self, id=None, goal_set=None):
        super(GORealUser, self).__init__(id, const.NL_SIMULATION_MODE, goal_set)

    def reset(self):
        # TODO
        raise NotImplementedError()

    def step(self, agt_action):
        usr_nl = raw_input("Next utterance");

        # TODO
        raise NotImplementedError()


class GOSimulatedUser(GOUser):
    """
    Abstract Base Class for all simulated users in the Goal-Oriented Dialogue Systems.
    Extends the [GOUser] class.
    
    # Class members:
        - slot_set: the set of all slots in the dialogue scenario
        - act_set: the set of all acts (intents) in the dialogue scenario
    """

    def __init__(self, id=None, simulation_mode=None, goal_set=None, slot_set=None, act_set=None):
        super(GOSimulatedUser, self).__init__(id, simulation_mode, goal_set)

        self.slot_set = slot_set
        self.act_set = act_set

    def reset(self):
        raise NotImplementedError()

    def step(self, agt_action):
        raise NotImplementedError()


class GORuleBasedUser(GOSimulatedUser):
    """
    Class representing a rule based user in the Goal-Oriented Dialogue Systems.
    Extends the [GOUser] class.
    
    Class members:
    """

    def __init__(self, id=None, simulation_mode=None, goal_set=None, slot_set=None, act_set=None):
        super(GORuleBasedUser, self).__init__(id, simulation_mode, goal_set, slot_set, act_set)

    def reset(self):
        # TODO
        raise NotImplementedError()

    def step(self, agt_action):
        # TODO
        raise NotImplementedError()


class GOModelBasedUser(GOSimulatedUser):
    """
    Class representing a model based user in the Goal-Oriented Dialogue Systems.
    Extends the [GOUser] class.
    
    Class members:
        - is_training: boolean flag indicating the mode of using te model-based state tracker
        - model_path: the path to save or load the model
    """

    def __init__(self, id=None, simulation_mode=None, goal_set=None, slot_set=None, act_set=None, is_training=None,
                 model_path=None):
        super(GOModelBasedUser, self).__init__(id, simulation_mode, goal_set, slot_set, act_set)

        self.is_training = is_training
        self.model_path = model_path

    def reset(self):
        # TODO
        raise NotImplementedError()

    def step(self, agt_action):
        # TODO
        raise NotImplementedError()
