"""
Author: Vladimir Ilievski <ilievski.vladimir@live.com>

"""

########################################################################################################################
# Agent-related constants                                                                                              #
########################################################################################################################

# key for specifying the type of the agent
AGENT_TYPE_KEY = "agent_type"
# value for the dqn agent type
AGENT_TYPE_DQN = "agent_type_dqn"

########################################################################################################################
# User-related constants                                                                                               #
########################################################################################################################

# key for specifying the type of the user
USER_TYPE_KEY = "user_type"
# value for the rule-based user type
RULE_BASED_USER = "rule_based_user"
# value for the model-based user type
MODEL_BASED_USER = "model_based_user"
# key for specifying a path to an already trained model-based user
MODEL_BASED_USER_PATH_KEY = "model_based_user_path"
# value for the real user type
REAL_USER = "real_user"
# key for specifying the user inform slots in the user internal state
USER_STATE_INFORM_SLOTS="user_inform_slots"
# key for specifying the user request slots in the user internal state
USER_STATE_REQUEST_SLOTS="user_request_slots"
# key for specifying the history of all user inform slots in the user internal state
USER_STATE_HISTORY_SLOTS="user_history_slots"
# key for specifying the history of all user slots in the user internal state
USER_STATE_REST_SLOTS="user_rest_slots"

########################################################################################################################
# State Tracker-related constants                                                                                      #
########################################################################################################################

# key for specifying the type of the state tracker
STATE_TRACKER_TYPE_KEY = "state_tracker_type"
# value for the rule-based state-tracker type
RULE_BASED_STATE_TRACKER = "rule_based_state_tracker"
# value for the model-based state-tracker type
MODEL_BASED_STATE_TRACKER = "model_based_state_tracker"
# key for specifying a path to an already trained model-based state tracker
MODEL_BASED_STATE_TRACKER_PATH_KEY = "model_based_state_tracker_path"

########################################################################################################################
# Agent training related constants                                                                                     #
########################################################################################################################

# key for specifying the simulation mode
SIMULATION_MODE_KEY = "simulation_mode"
# value for the semantic frame simulation mode
SEMANTIC_FRAME_SIMULATION_MODE = "semantic_frame_simulation_mode"
# value for the natural language simulation mode
NL_SIMULATION_MODE = "nl_simulation_mode"
# flag indicating the mode of the dialogue system
IS_TRAINING_KEY = "is_training"
# key for specifying the maximal number of dialogue turns
MAX_NB_TURNS = "max_nb_turns"


# key for specifying the path to the nlu unit
NLU_PATH_KEY = "nlu_path"
# key for specifying the path to the nlg unit
NLG_PATH_KEY = "nlg_path"


########################################################################################################################
# User and Agent action related constants                                                                              #
########################################################################################################################
# key for specifying the intent (act) of the dialogue turn
DIA_ACT_KEY = "diaact"
# key for specifying an inform slot
INFORM_SLOT_KEY = "inform_slots"
# key for specifying a requested slot
REQUEST_SLOT_KEY = "request_slots"
# key for specifying the nl part of the action
NL_KEY = "nl"
# key for specifying a proposed slot
PROPOSED_SLOT_KEY = "proposed_slots"
# key for specifying an agent requested slot
AGENT_REQUESTED_SLOT_KEY = "agent_request_slots"
# value for the unknown slots
UNKNOWN_SLOT_VALUE = "UKN"
# key for specifying the speaker type (user or agent)
SPEAKER_TYPE_KEY = "speaker"
# value for the speaker key, when the user is the speaker
USR_SPEAKER_VAL = "user_speaks"
# value for the speaker key, when the agent is the speaker
AGT_SPEAKER_VAL = "agent_speaks"
# key for specifying the turn number
TURN_NB_KEY = "turn"

########################################################################################################################
# Knowledge Base related constants                                                                                     #
########################################################################################################################

# key for specifying a path to the knowledge base
KB_PATH_KEY = "kb_path"
# key for specifying a kb querying result where all of the constraints were matched
KB_MATCHING_ALL_CONSTRAINTS_KEY = "matching_all_constraints"

########################################################################################################################
# Dialog status related constants                                                                                      #
########################################################################################################################

# dialogue status
FAILED_DIALOG = -1
SUCCESS_DIALOG = 1
NO_OUTCOME_YET = 0

# Rewards
SUCCESS_REWARD = 50
FAILURE_REWARD = 0
PER_TURN_REWARD = -1


