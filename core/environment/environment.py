"""
Author: Vladimir Ilievski <ilievski.vladimir@live.com>

A Python file
"""

from core import constants as const

import core.dst.state_tracker as state_trackers
import core.user.users as users

from nlp.nlu.nlu import nlu
from nlp.nlg.nlg import nlg

from rl.core import Env


class GOEnv(Env):
    """
    The Environment with which the agent is interacting with. It extends the keras-rl class Env.
    Therefore, the following methods are implemented:
    
    - `step`
    - `reset`
    - `render`
    - `close`
    - `seed`
    - `configure`
    
    # Class members:
    
    From `rl.Env` class:
    
        - ** reward_shape **: the shape of the reward matrix
        - ** action_space **: the space of possible actions
        - ** observation_space **: the space of observations to ... (have to find the exact meaning)
    
    Own:
    
        - ** simulation_mode **: the mode of the simulation, semantic frame or natural language sentences
        - ** is_training **: flag indicating the training/testing mode
        - ** max_nb_turns **: the maximal number of allowed dialogue turns. Afterwards, the dialogue is considered failed
        - ** usr **: a simulated or real user making a conversation with the agent
        - ** state_tracker **: the state tracker used for tracking the state of the dialogue
        - ** nlu_unit **: the NLU unit for transforming the user utterance to a dialogue act
        - ** nlg_unit **: the NLG unit for transforming the agent's action to a natural language sentence
        - ** act_set **: the set of all dialogue acts
        - ** slot_set **: the set of all dialogue slots
        - ** feasible_actions **: list of templates described as dictionaries, corresponding to each action the agent might take
                            (dict to be specified)
    """

    def __init__(self, simulation_mode=None, is_training=False, user_type_str="", user_path="", dst_type_str="",
                 dst_path="", act_set=None, slot_set=None, feasible_actions=None, max_nb_turns=None, nlu_path="",
                 nlg_path="", *args, **kwargs):
        """
        Constructor for the Environment class.
        
        :param simulation_mode: semantic frame or natural language sentence form of user utterances
        :param is_training: flag indicating the training/testing mode
        :param user_type_str: the type of the user (rule-based or model-based)
        :user_path: the path to save or load the user model (empty if not model-based)
        :param dst_type_str: the type of the dialogue state tracker (rule-based or model-based)
        :dst_path: the path to save or load the state tracker model (empty if not model-based)
        :act_set: the set of all dialogue acts (intents)
        :slot_set: the set of all dialogue slots
        :param nlu_path: the path to load the NLU unit
        :param nlg_path: the path to load the NLG unit
        """

        # call super class constructor
        super(GOEnv).__init__(*args, **kwargs)

        self.simulation_mode = simulation_mode
        self.is_training = is_training

        self.act_set = act_set
        self.slot_set = slot_set

        self.feasible_actions = feasible_actions

        self.current_turn_nb = 0
        self.max_nb_turns = max_nb_turns

        # create the user
        self.user = self.__create_user(user_type_str, user_path, is_training)

        # create the state tracker
        self.state_tracker = self.__create_state_tracker(dst_type_str, dst_path, is_training, act_set, slot_set,
                                                         max_nb_turns)

        # create the nlu unit
        self.nlu_unit = self.__create_nlu_unit(nlu_path)

        # create the nlg unit
        self.nlg_unit = self.__create_nlg_unit(nlg_path)

    def __create_user(self, user_type_str, user_path, is_training):
        """
        Private helper method for creating a user.
        
        :param user_type_str: the type of the user tp create (rule-based or model-based)
        :param user_path: the path to load a trained user model (empty is not model-based)
        :param is_training: flag indicating the training/testing mode of the user (for the model-based)
        :return: the newly created user
        """

        user = None

        if user_type_str == const.RULE_BASED_USER:
            user = users.GORuleBasedUser()
        elif user_type_str == const.MODEL_BASED_USER:
            user = users.GOModelBasedUser(is_training, user_path)
        elif user_type_str == const.REAL_USER:
            user = users.GORealUser()
        else:
            raise Exception()

        return user

    def __create_state_tracker(self, dst_type_str, dst_path, is_training, act_set, slot_set, max_nb_turns):
        """
        Private helper method for creating a state tracker.
        
        :param dst_type_str: the type of the dialogue state tracker (rule-based or model-based)
        :param dst_path: the path to load a trained state tracker model (empty if not model-based)
        :param is_training: flag indicating the training/testing mode of the user (for the model-based)
        :act_set: the set of all dialogue acts (intents)
        :slot_set: the set of all dialogue slots
        :return: the newly created state tracker
        """
        state_tracker = None

        if dst_type_str == const.RULE_BASED_STATE_TRACKER:
            state_tracker = state_trackers.GORuleBasedStateTracker(act_set, slot_set, max_nb_turns)
        elif dst_type_str == const.MODEL_BASED_STATE_TRACKER:
            state_tracker = state_trackers.GOModelBasedStateTracker(act_set, slot_set, max_nb_turns, is_training,
                                                                    dst_path)
        else:
            raise Exception()

        return state_tracker

    def __create_nlu_unit(self, nlu_path):
        """
        Private helper method for creating an NLU unit
        
        :param nlu_path: the path to load a trained NLU unit
        :return: the newly created NLU unit
        """
        nlu_unit = nlu()
        nlu_unit.load_nlu_model(nlu_path)

        return nlu_unit

    def __create_nlg_unit(self, nlg_path):
        """
        Private helper method for creating an NLU unit.
        
        :param nlg_path: the path to load a trained NLG unit
        :return: the newly created NLG unit
        """
        nlg_unit = nlg()
        nlg_unit.load_nlg_model(nlg_path)

        return nlg_unit

    def __process_usr_action(self, usr_action):
        """
        Private helper method for processing the user action.
        
        :param usr_action: the user action to be processed
        :return: processed user action
        """

        # by default add NL representation to the user action
        user_nlg_sentence = self.nlg_model.convert_diaact_to_nl(usr_action, const.USR_SPEAKER_VAL)
        usr_action[const.NL_KEY] = user_nlg_sentence

        # if the simulation mode is on Natural Language level, generate new user action
        if self.simulation_mode == const.NL_SIMULATION_MODE:
            user_nlu_res = self.nlu_model.generate_dia_act(usr_action[const.NL_KEY])
            usr_action.update(user_nlu_res)

        return usr_action

    def __process_agt_action(self, agt_action):
        """
        Private helper method for processing the agent action.
        
        :param agt_action: the agent action to be processed
        :return: processed agent action
        """

        # add NL representation to the agent action
        agent_nlg_sentence = self.nlg_model.convert_diaact_to_nl(agt_action, const.AGT_SPEAKER_VAL)
        agt_action[const.NL_KEY] = agent_nlg_sentence

        return agt_action

    def step(self, action):
        """
        Method for taking the environment one step further. In this case, to present the agent action to the user in order
        to make a response. Overrides the super class method.
        
        :param action: the last agent action 
        :return: user's response to the agent's action in form of a state
        """

        # TODO
        new_state = {}
        reward = 0
        done = False
        info = ""


        # increase the dialogue turn number
        self.current_turn_nb += 1
        # process the agent action
        proc_agt_action = self.__process_agt_action(action)
        # update the state tracker with the new agent action
        self.state_tracker.update(proc_agt_action, const.AGT_SPEAKER_VAL)



        if self.current_turn_nb >= self.max_nb_turns:
            done = True
        else:
            # get the new user action
            new_user_action, dialogue_status = self.user.step(proc_agt_action)
            # increase the dialogue turn number
            self.current_turn_nb += 1
            # process the new user action
            proc_new_user_action = self.__process_usr_action(new_user_action)
            # update the state tracker with the new user action
            self.state_tracker.update(proc_new_user_action, const.USR_SPEAKER_VAL)
            # produce new state for the
            new_state = self.state_tracker.produce_state()


        return new_state, reward, done, info

    def reset(self):
        """
        Method for resetting the dialogue state tracker and the user, called at the beginning of each new episode.
        Overrides the super class method.
        
        :return: the initial observation
        """

        # reset the dst
        self.state_tracker.reset()
        # reset the user and get the initial action
        init_usr_action = self.user.reset()
        # increase the dialogue turn number
        self.current_turn_nb += 1
        # process the init user action
        proc_init_usr_action = self.__process_usr_action(init_usr_action)
        # update the dialogue state tracker
        self.state_tracker.update(proc_init_usr_action, const.USR_SPEAKER_VAL)
        # produce state for the agent
        init_state = self.state_tracker.produce_state()

        return init_state

    def render(self, mode='human', close=False):

        # TODO
        raise NotImplementedError()

    def close(self):

        # TODO
        raise NotImplementedError()

    def seed(self, seed=None):

        # TODO
        raise NotImplementedError()

    def configure(self, *args, **kwargs):

        # TODO
        raise NotImplementedError()
