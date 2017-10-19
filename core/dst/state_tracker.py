"""
Author: Vladimir Ilievski <ilievski.vladimir@live.com>


A Python file for the Goal-Oriented Dialogue State Tracker classes.
"""

from core import constants as const

import numpy as np
import copy

class GOStateTracker:
    """
    Abstract Base Class of all state trackers in the Goal-Oriented Dialogue Systems.
    
    # Class members:
    
        - ** history **: list of both user and agent actions, such that they are in alternating order
        - ** act_set **: the set of all intents used in the dialogue.
        - ** slot_set **: the set of all slots used in the dialogue.
        - ** act_set_cardinality **: the cardinality of the act set.
        - ** slot_set_cardinality **: the cardinality of the slot set.
        - ** current_slots **: a dictionary that keeps a running record of which slots are filled 
                        (inform slots) and which are requested (request slots)
        - ** state_dim **: the dimensionality of the state. It is calculated afterwards.
        - ** max_nb_turns **: the maximal number of dialogue turns
    """

    def __init__(self, act_set=None, slot_set=None, max_nb_turns=None):
        """
        Constructor of the [GO State Tracker] class.
        """

        # the list of history
        self.history = []

        # The act and slot sets
        self.act_set = act_set
        self.slot_set = slot_set

        # The cardinality of the act and slot sets
        self.act_set_cardinality = len(self.act_set.keys())
        self.slot_set_cardinality = len(self.slot_set.keys())

        # initialize the running record of the slots
        self.current_slots = {}
        self.current_slots[const.INFORM_SLOT_KEY] = {}
        self.current_slots[const.REQUEST_SLOT_KEY] = {}
        self.current_slots[const.PROPOSED_SLOT_KEY] = {}
        self.current_slots[const.AGENT_REQUESTED_SLOT_KEY] = {}

        self.current_turn_nb = 0
        self.max_nb_turns = max_nb_turns

        # TODO
        self.state_dim = 0

    def __update_usr_action(self, usr_action):
        """
        Abstract private helper method to update the state tracker with the last user action.
        
        :param usr_action: the action user took
        :return: 
        """

        raise NotImplementedError()

    def __update_agt_action(self, agt_action):
        """
        Abstract private helper method to update the state tracker with the last agent action.
        
        :param agt_action: the action agent took
        :return: 
        """

        raise NotImplementedError()


    def get_history(self):
        """
        Getter method to get the dialogue history.
        
        :return: the history dialogue list 
        """
        return self.history

    def get_last_usr_action(self):
        """
        Getter method, to get the last user action, if any.
        
        :return: the last user action as dictionary, if any
        """
        return self.history[-1] if len(self.history) > 0 else None

    def get_last_agt_action(self):
        """
        Getter method, to get the last agent action, if any.
        
        :return: the last agent action as dictionary, if any
        """

        return self.history[-2] if len(self.history) > 1 else None

    def reset(self):
        """
        Abstract method for resetting the dialogue state tracker, usually at the beginning of a new episode.
        
        :return: true if the resetting was successful
        """

        raise NotImplementedError()

    def produce_state(self):
        """
        Abstract method to produce a representation for the current dialogue state.
        
        :return:
        """
        raise NotImplementedError()

    def update(self, action=None, speaker=""):
        """
        Abstract method to update the state tracker with the last user or agent action.
        
        :param action: user or agent action
        :param speaker: who took the action, the user or the agent
        :return: 
        """
        raise NotImplementedError()


class GORuleBasedStateTracker(GOStateTracker):
    """
    Class for Rule-Based state tracker in the Goal-Oriented Dialogue Systems.
    Extends the `GOStateTracker` class.
    
    # Class members:
        
        - ** state_dim **: the dimension of the state
    """

    def __init__(self, act_set=None, slot_set=None, max_nb_turns=None):
        """
        Constructor of the [GO Rule Based State Tracker] class.
        """

        super(GORuleBasedStateTracker, self).__init__(act_set, slot_set, max_nb_turns)

        # TODO
        self.state_dim = 0

    def __encode_action_intent(self, action_intent):
        """
        Private helper method to create one-hot encoding for the intent of the current user or agent action.

        :param action_intent: string, describing the intent of the user or agent action
        :return: list in one-hot format
        """

        action_intent_encoding = np.zeros((1, self.act_set_cardinality))
        action_intent_encoding[0, self.act_set[action_intent]] = 1.0

        return action_intent_encoding

    def __encode_action_inform_slots(self, action_inform_slots):
        """
        Private helper method to create bag encoding for the inform slots in the current user or agent action.

        :param action_inform_slots: a dictionary of inform slots present in the current user or agent action
        :return: list in bag format
        """

        action_inform_slots_encoding = np.zeros((1, self.slot_set_cardinality))
        for slot in action_inform_slots.keys():
            action_inform_slots_encoding[0, self.slot_set[slot]] = 1.0

        return action_inform_slots_encoding

    def __encode_action_request_slot(self, action_request_slots):
        """
        Private helper method to create bag encoding for the request slots in the current user or agent action.

        :param action_request_slots: a dictionary of request slots in the current user or agent action
        :return: list in bag format
        """

        action_request_slots_encoding = np.zeros((1, self.slot_set_cardinality))
        for slot in action_request_slots.keys():
            action_request_slots_encoding[0, self.slot_set[slot]] = 1.0

        return action_request_slots_encoding

    def __encode_all_inform_slots(self, all_inform_slots):
        """
        Private helper method to create bag encoding for all inform slots during the dialogue.

        :param all_inform_slots: a dictionary of all inform slots
        :return: list in bag format
        """

        all_inform_slots_encoding = np.zeros((1, self.slot_set_cardinality))
        for slot in all_inform_slots:
            all_inform_slots_encoding[0, self.slot_set[slot]] = 1.0

        return all_inform_slots_encoding

    def __encode_dialogue_turn_scaled(self, curr_turn_nb):
        """
        Private helper method for encoding the dialogue turn number scaled by 10

        :param curr_turn_nb: current dialogue turn number
        :return: one element list
        """

        scaled_turn_encoding = np.zeros((1, 1)) + curr_turn_nb / 10.
        return scaled_turn_encoding

    def __encode_dialogue_turn(self, curr_turn_nb):
        """
        Private helper method to create one-hot encoding for the current dialogue turn

        :param curr_turn_nb: current dialogue turn number
        :return: list in one-hot format
        """

        dialogue_turn_encoding = np.zeros((1, self.max_nb_turns))
        dialogue_turn_encoding[0, curr_turn_nb] = 1.0

        return dialogue_turn_encoding

    def __encode_kb_results_scaled(self, kb_results_dict):
        """
        Private helper method to create scaled counts encoding of the kb querying results

        :param kb_results_dict: dictionary of kb querying results
        :return: list of scaled kb querying results
        """

        kb_scaled_count_encoding = np.zeros((1, self.slot_set_cardinality + 1)) + kb_results_dict[
                                                                                      'matching_all_constraints'] / 100.
        for slot in kb_results_dict:
            if slot in self.slot_set:
                kb_scaled_count_encoding[0, self.slot_set[slot]] = kb_results_dict[slot] / 100.

        return kb_scaled_count_encoding

    def __encode_kb_results_binary(self, kb_results_dict):
        """
        Private helper method to create binary encoding of the kb querying results.

        :param kb_results_dict: dictionary of kb querying results
        :return: 
        """
        kb_binary_count_encoding = np.zeros((1, self.slot_set_cardinality + 1)) + np.sum(
            kb_results_dict[const.KB_MATCHING_ALL_CONSTRAINTS_KEY] > 0.)
        for slot in kb_results_dict:
            if slot in self.slot_set:
                kb_binary_count_encoding[0, self.slot_set[slot]] = np.sum(kb_results_dict[slot] > 0.)
        return kb_binary_count_encoding

    def __update_usr_action(self, usr_action):
        """
        Abstract method implementation.
        """

        # Iterate over the inform slots from the last user action and update the state tracker running record
        for slot in usr_action[const.INFORM_SLOT_KEY].keys():
            self.current_slots[const.INFORM_SLOT_KEY][slot] = usr_action[const.INFORM_SLOT_KEY][slot]
            # if the current inform slot was in the requested slots in the past, delete it
            if slot in self.current_slots[const.REQUEST_SLOT_KEY].keys():
                del self.current_slots[const.REQUEST_SLOT_KEY][slot]

        # Iterate over the request slots from the last user action and update the state tracker running record
        for slot in usr_action[const.REQUEST_SLOT_KEY].keys():
            if slot not in self.current_slots[const.REQUEST_SLOT_KEY].keys():
                self.current_slots[const.REQUEST_SLOT_KEY][slot] = const.UNKNOWN_SLOT_VALUE

        # Produce a record for the history and add the last user action in the history
        new_history_record = {}
        new_history_record[const.TURN_NB_KEY] = self.current_turn_nb
        new_history_record[const.SPEAKER_TYPE_KEY] = const.USR_SPEAKER_VAL
        new_history_record[const.DIA_ACT_KEY] = usr_action[const.DIA_ACT_KEY]
        new_history_record[const.INFORM_SLOT_KEY] = usr_action[const.INFORM_SLOT_KEY]
        new_history_record[const.REQUEST_SLOT_KEY] = usr_action[const.REQUEST_SLOT_KEY]

        self.history.append(copy.deepcopy(new_history_record))

        return True

    def __update_agt_action(self, agt_action):
        """
        Abstract method implementation.
        """

        # Make a copy and call KB helper methods to fill in the values for the inform slots
        agt_action_copy = copy.deepcopy(agt_action)
        inform_slots_from_kb = None #TODO

        # Iterate over the inform slots from the KB and update the state tracker running record
        for slot in inform_slots_from_kb.keys():
            self.current_slots[const.PROPOSED_SLOT_KEY][slot] = inform_slots_from_kb[slot]
            self.current_slots[const.INFORM_SLOT_KEY] [slot] = inform_slots_from_kb[slot]
            # if the current inform slot was in the requested slots in the past, delete it
            if slot in self.current_slots[const.REQUEST_SLOT_KEY].keys():
                del self.current_slots[const.REQUEST_SLOT_KEY][slot]

        # Iterate over the request slots from the last agent action and update the state tracker running record
        for slot in agt_action_copy[const.REQUEST_SLOT_KEY].keys():
            if slot not in self.current_slots[const.AGENT_REQUESTED_SLOT_KEY].keys():
                self.current_slots[const.AGENT_REQUESTED_SLOT_KEY][slot] = const.UNKNOWN_SLOT_VALUE

        # Produce a record for the history and add the last agent action in the history
        new_history_record = {}
        new_history_record[const.TURN_NB_KEY] = self.current_turn_nb
        new_history_record[const.SPEAKER_TYPE_KEY] = const.AGT_SPEAKER_VAL
        new_history_record[const.DIA_ACT_KEY] = agt_action_copy[const.DIA_ACT_KEY]
        new_history_record[const.INFORM_SLOT_KEY] = agt_action_copy[const.INFORM_SLOT_KEY]
        new_history_record[const.REQUEST_SLOT_KEY] = agt_action_copy[const.REQUEST_SLOT_KEY]

        self.history.append(copy.deepcopy(new_history_record))

        return True

    def reset(self):
        """
        Method to reset the rule-based dialogue state tracker. Overrides the super class method.
        
        :return: true if the resetting was successful, false otherwise
        """

        # clear the history
        self.history = []

        # clear the running record of filled slots
        self.current_slots = {}
        self.current_slots[const.INFORM_SLOT_KEY] = {}
        self.current_slots[const.REQUEST_SLOT_KEY] = {}
        self.current_slots[const.PROPOSED_SLOT_KEY] = {}
        self.current_slots[const.AGENT_REQUESTED_SLOT_KEY] = {}

        # set turn number to 0
        self.current_turn_nb = 0

        return True

    def produce_state(self):
        """
        Abstract method implementation.
        Method to produce a representation for the current dialogue state. In this rule-based state tracker it includes:
        
            - one-hot encoding of the last user and the agent action intent
            - bag encoding of the last user and the agent action inform slots
            - bag encoding of the last user and the agent action request slots
            - bag encoding of all inform slots in the dialogue so far
            - dialogue turn number scaled by 10
            - one-hot encoding of the dialogue turn number
            - kb querying results scaled by 100
            - kb querying results in a binary form, like present not present
        
        :return: list of numbers representing the current state
        """

        # get the last user and agent action
        last_usr_action = self.get_last_usr_action()
        last_agt_action = self.get_last_agt_action()

        # user action intent encoding
        usr_action_intent_encoding = self.__encode_action_intent(last_usr_action[const.DIA_ACT_KEY])

        # agent action intent encoding
        agt_action_intent_encoding = self.__encode_action_intent(last_agt_action[const.DIA_ACT_KEY])

        # user inform slots encoding
        usr_action_inform_slots_encoding = self.__encode_action_inform_slots(last_usr_action[const.INFORM_SLOT_KEY])

        # agent inform slots encoding
        agt_action_inform_slots_encoding = self.__encode_action_inform_slots(last_agt_action[const.INFORM_SLOT_KEY])

        # user inform slots encoding
        usr_action_request_slots_encoding = self.__encode_action_inform_slots(last_usr_action[const.REQUEST_SLOT_KEY])

        # agent inform slots encoding
        agt_action_request_slots_encoding = self.__encode_action_inform_slots(last_agt_action[const.REQUEST_SLOT_KEY])

        # all inform slots in the dialogue so far
        all_inform_slots_encoding = self.__encode_all_inform_slots(self.current_slots[const.INFORM_SLOT_KEY])

        # scaled dialogue turn number encoding
        scaled_turn_encoding = self.__encode_dialogue_turn_scaled(self.current_turn_nb)

        # one-hot dialogue turn number encoding
        dialogue_turn_encoding = self.__encode_dialogue_turn(self.current_turn_nb)

        # TODO: create the KB helper class to query the KB
        kb_results_dict = {}

        # kb scaled encoding
        kb_scaled_count_encoding = self.__encode_kb_results_scaled(kb_results_dict)

        # kb binary encoding
        kb_binary_count_encoding = self.__encode_kb_results_binary(kb_results_dict)

        # stack everything in one vector
        final_representation = np.hstack(
            [usr_action_intent_encoding, usr_action_inform_slots_encoding, usr_action_request_slots_encoding,
             agt_action_intent_encoding, agt_action_inform_slots_encoding, agt_action_request_slots_encoding,
             all_inform_slots_encoding, scaled_turn_encoding, dialogue_turn_encoding, kb_binary_count_encoding,
             kb_scaled_count_encoding])

        return final_representation

    def update(self, action=None, speaker=None):

        # the function should be called proplerly
        assert (not (action and speaker))

        # increase the turn number for one
        self.current_turn_nb += 1

        if speaker == const.USR_SPEAKER_VAL:
            return self.__update_usr_action(action)
        else:
            return self.__update_agt_action(action)


class GOModelBasedStateTracker(GOStateTracker):
    """
    Class for Model-Based state tracker in the Goal-Oriented Dialogue Systems.
    Extends the `GOStateTracker` class.
    
    # Class members:
    
        - ** is_training **: boolean flag indicating the mode of using the model-based state tracker
        - ** model_path **: the path to save or load the model
    """

    def __init__(self, act_set=None, slot_set=None, max_nb_turns=None, is_training=None, model_path=None):
        super(GOModelBasedStateTracker, self).__init__(act_set, slot_set, max_nb_turns)

        self.is_training = is_training
        self.model_path = model_path

    def __update_usr_action(self, usr_action):
        # TODO
        raise NotImplementedError()

    def __update_agt_action(self, agt_action):
        # TODO
        raise NotImplementedError()

    def reset(self):
        """
        Method to reset the model-based dialogue state tracker. Overrides the super class method.

        :return: true if the resetting was successful, false otherwise
        """

        # TODO
        raise NotImplementedError()

    def produce_state(self):
        # TODO
        raise NotImplementedError()

    def update(self, action=None, speaker=""):
        # TODO
        raise NotImplementedError()
