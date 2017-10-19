"""
Author: Vladimir Ilievski <ilievski.vladimir@live.com>

A Python file for the users in the Goal-Oriented Dialogue Systems
"""

from core import constants as const
import random


class GOUser:
    """
    Abstract Base Class of all type of Goal-Oriented conversational users. The user is taking an action after each turn.
    One user action is represented as a dictionary having the exact same structure as the agent action, which is:
    
        - ** diaact **: the act of the action
        - ** inform_slots **: the set of informed slots
        - ** request_slots **: the set of request slots
        - ** nl **: the natural language representation of the agent action
    
    Also, the user is having a goal to follow, which is also a dictionary with the same structure as the user or agent
    action, with the difference that all information is provided.
    
    Moreover, the agent is keeping its own internal state, for its last turn, represented as a dictionary, which is:
    
        - ** diaact **: the user's dialogue act
        - ** inform_slots **: the user's inform slots from its previous turn
        - ** request_slots **: the user's request slots from its previous turn
        - ** history_slots **: the history of all user's inform slots
        - ** rest_slots **: the history of all user's slots
        
    # Class members:
    
        - ** id **: the id of the user
        - ** current_turn_nb **: the number of the current dialogue turn
        - ** state **: user internal state, keeping record of the past and current actions
        - ** simulation_mode **: semantic frame or natural language sentence form of user utterances
        - ** goal_set **: the set of goals for the user
        - ** goal **: the user goal in the current dialogue turn
    """

    def __init__(self, id=None, simulation_mode=None, goal_set=None):
        # Initialize the class members
        self.id = id
        self.current_turn_nb = 0

        self.state = {}
        self.state[const.DIA_ACT_KEY] = ""
        self.state[const.USER_STATE_INFORM_SLOTS] = {}
        self.state[const.USER_STATE_REQUEST_SLOTS] = {}
        self.state[const.USER_STATE_HISTORY_SLOTS] = {}
        self.state[const.USER_STATE_REST_SLOTS] = {}

        self.simulation_mode = simulation_mode
        self.goal_set = goal_set

        self.goal = None

    def reset(self):
        """
        Abstract method for restarting the user and getting the initial user action.
        
        :return: The initial user action
        """
        raise NotImplementedError()

    def step(self, agt_action):
        """
        Abstract method for getting the next user action given the last agent action.
        
        :param agt_action: last agent action
        :return: next user action and the dialogue status
        """
        raise NotImplementedError()


class GORealUser(GOUser):
    """
    Class connecting a real user, writing on the standard input. Extends the `GOUser` class.
    
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
    Extends the `GOUser` class.
    
    # Class members:
    
        - ** slot_set **: the set of all slots in the dialogue scenario
        - ** act_set **: the set of all acts (intents) in the dialogue scenario
    """

    def __init__(self, id=None, simulation_mode=None, goal_set=None, slot_set=None, act_set=None):
        super(GOSimulatedUser, self).__init__(id, simulation_mode, goal_set)

        self.slot_set = slot_set
        self.act_set = act_set

    def __sample_random_init_action(self):
        """
        Abstract private helper method for sampling a random initial user action based on the goal.

        :return: random initial user action
        """
        raise NotImplementedError()

    def __sample_goal(self):
        """
        Abstract private helper method for sampling a random user goal, given the set of all available user goals.

        :return: random user goal
        """
        raise NotImplementedError()

    def reset(self):
        # reset the number of turns
        self.current_turn_nb = 0

        # reset the user state
        self.state = {}
        self.state[const.DIA_ACT_KEY] = ""
        self.state[const.USER_STATE_INFORM_SLOTS] = {}
        self.state[const.USER_STATE_REQUEST_SLOTS] = {}
        self.state[const.USER_STATE_HISTORY_SLOTS] = {}
        self.state[const.USER_STATE_REST_SLOTS] = {}

        # sample a random goal and set it as a user goal in the following episode
        self.goal = self.__sample_goal()

        # after sampling a goal, the user can take the initial actions
        init_action = self.__sample_random_init_action()

        return init_action

    def step(self, agt_action):
        raise NotImplementedError()


class GORuleBasedUser(GOSimulatedUser):
    """
    Abstract class representing a rule-based user in the Goal-Oriented Dialogue Systems.
    Since, it is a rule-based simulated user, it will be domain-specific.
    Extends the `GOUser` class.
    
    Class members:
    
        - ** init_dia_act_set **: the set of initial dialogue acts 
        - ** init_slots **: for each initial dialogue act, the set of initial inform slots
    """

    def __init__(self, id=None, simulation_mode=None, goal_set=None, slot_set=None, act_set=None, init_slots=None,
                 init_dia_act_set=None):
        super(GORuleBasedUser, self).__init__(id, simulation_mode, goal_set, slot_set, act_set)

        self.init_dia_act_set = init_dia_act_set
        self.init_slots = init_slots

    def __sample_random_init_action(self):
        """
        Overrides abstract method from the super class
        """
        # increase the dialogue number turn
        self.current_turn_nb += 1

        # the resulting initial action
        init_action = {}

        # randomly sample initial dialogue act
        temp_init_dia_act = random.choice(self.init_dia_act_set)
        self.state[const.DIA_ACT_KEY] = temp_init_dia_act

        # sample inform slots
        if len(self.goal[const.INFORM_SLOT_KEY]) > 0:
            # sample an inform slot from the current user goal insert it in the user's internal state
            sampled_inform_slot = random.choice(list(self.goal[const.INFORM_SLOT_KEY].keys()))
            self.state[const.USER_STATE_INFORM_SLOTS][sampled_inform_slot] = self.goal[const.INFORM_SLOT_KEY][
                sampled_inform_slot]

            # after sampling the initial inform slot, check the presence of the initial slots
            for init_slot in self.init_slots[temp_init_dia_act]:
                if init_slot in self.goal[const.INFORM_SLOT_KEY].keys():
                    self.state[const.USER_STATE_INFORM_SLOTS][init_slot] = self.goal[const.INFORM_SLOT_KEY][init_slot]

        # sample request slots
        sampled_request_slot = random.choice(list(self.goal[const.REQUEST_SLOT_KEY].keys()))
        self.state[const.USER_STATE_REQUEST_SLOTS][sampled_request_slot] = 'UNK'

        # if there are no request slots (too static?)
        if len(self.state[const.USER_STATE_REQUEST_SLOTS]) == 0:
            self.state[const.DIA_ACT_KEY] = 'inform'

        # create the user action
        init_action[const.DIA_ACT_KEY] = self.state[const.DIA_ACT_KEY]
        init_action[const.INFORM_SLOT_KEY] = self.state[const.USER_STATE_INFORM_SLOTS]
        init_action[const.REQUEST_SLOT_KEY] = self.state[const.USER_STATE_REQUEST_SLOTS]

        return init_action

    def __sample_goal(self):
        """
        Overrides the abstract method from the super class
        """
        sample_goal = random.choice(self.goal_set)
        return sample_goal

    def __response_inform(self, agt_action):
        """
        Private abstract method for generating response it the last agent action act is 'inform'.
        
        :param agt_action: the last agent action
        :return:
        """

        # TODO
        raise NotImplementedError()

    def __response_request(self, agt_action):
        """
        Private abstract method for generating response it the last agent action act is 'request'.
        
        :param agt_action: the last agent action
        :return:
        """

        # TODO
        raise NotImplementedError()

    def __response_confirm_answer(self, agt_action):
        """
        Private abstract method for generating response it the last agent action act is 'confirm_answer'.
        
        :param agt_action: the last agent action
        :return:
        """

        # TODO
        raise NotImplementedError()

    def __response_multiple_choice(self, agt_action):
        """
        Private abstract method for generating response it the last agent action act is 'multiple_choice'.
        
        :param agt_action: the last agent action
        :return:
        """

        # TODO
        raise NotImplementedError()

    def __response_thanks(self, agt_action):
        """
        Private abstract method for generating response it the last agent action act is 'thanks'.
        
        
        :param agt_action: the last agent action
        :return:
        """

        # TODO
        raise NotImplementedError()

    def __response_closing(self, agt_action):
        """
        Private abstract method for generating response it the last agent action act is 'closing'. 
                
        :param agt_action: the last agent action
        :return:
        """

        # TODO
        raise NotImplementedError()

    def step(self, agt_action):
        """
         Overrides the abstract method from the super class
        """

        # we need to increase it for 2, counting for the agent response afterwards
        self.current_turn_nb += 2
        agt_diaiact = agt_action[const.DIA_ACT_KEY]

        # the resulting next user action
        next_usr_action = {}
        dialogue_status = const.NO_OUTCOME_YET

        # based on the last agent action
        if agt_diaiact == "inform":
            self.__response_inform(agt_action)
        elif agt_diaiact == "multiple_choice":
            self.__response_multiple_choice(agt_action)
        elif agt_diaiact == "request":
            self.__response_request(agt_action)
        elif agt_diaiact == "confirm_answer":
            self.__response_confirm_answer(agt_action)
        elif agt_diaiact == "closing":
            self.__response_closing(agt_action)
            dialogue_status = const.SUCCESS_DIALOG
        elif agt_diaiact == "thanks":
            self.__response_thanks(agt_action)
            dialogue_status = const.SUCCESS_DIALOG

        # create the next user action
        next_usr_action[const.DIA_ACT_KEY] = self.state[const.DIA_ACT_KEY]
        next_usr_action[const.INFORM_SLOT_KEY] = self.state[const.USER_STATE_INFORM_SLOTS]
        next_usr_action[const.REQUEST_SLOT_KEY] = self.state[const.USER_STATE_REQUEST_SLOTS]

        return next_usr_action, dialogue_status


class GOModelBasedUser(GOSimulatedUser):
    """
    Class representing a model based user in the Goal-Oriented Dialogue Systems.
    Extends the `GOUser` class.
    
    Class members:
    
        - ** is_training **: boolean flag indicating the mode of using te model-based state tracker
        - ** model_path **: the path to save or load the model
    """

    def __init__(self, id=None, simulation_mode=None, goal_set=None, slot_set=None, act_set=None, is_training=None,
                 model_path=None):
        super(GOModelBasedUser, self).__init__(id, simulation_mode, goal_set, slot_set, act_set)

        self.is_training = is_training
        self.model_path = model_path

    def __sample_random_init_action(self):
        # TODO
        raise NotImplementedError()

    def __sample_goal(self):
        # TODO
        raise NotImplementedError()

    def step(self, agt_action):
        # TODO
        raise NotImplementedError()
