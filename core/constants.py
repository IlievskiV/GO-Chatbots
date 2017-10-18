"""
Author: Vladimir Ilievski <ilievski.vladimir@live.com>

"""


# key for specifying the type of the agent
AGENT_TYPE_KEY = "agent_type"
# value for the dqn agent type
AGENT_TYPE_DQN = "agent_type_dqn"


# key for specifying the type of the user
USER_TYPE_KEY = "user_type"
# value for the rule-based user type
RULE_BASED_USER = "rule_based_user"
# value for the model-based user type
MODEL_BASED_USER = "model_based_user"
# key for specifying a path to an alreadt trained model-based user
MODEL_BASED_USER_PATH_KEY = "model_based_user_path"
# value for the real user
REAL_USER = "real_user"


# key for specifying the type of the state tracker
STATE_TRACKER_TYPE_KEY = "state_tracker_type"
# value for the rule-based state-tracker type
RULE_BASED_STATE_TRACKER = "rule_based_state_tracker"
# value for the model-based state-tracker type
MODEL_BASED_STATE_TRACKER = "model_based_state_tracker"
# key for specifying a path to an already trained model-based state tracker
MODEL_BASED_STATE_TRACKER_PATH_KEY = "model_based_state_tracker_path"


# key for specifying the simulation mode
SIMULATION_MODE_KEY = "simulation_mode"
# value for the semantic frame simulation mode
SEMANTIC_FRAME_SIMULATION_MODE = "semantic_frame_simulation_mode"
# value for the natural language simulation mode
NL_SIMULATION_MODE = "nl_simulation_mode"

# flag indicating the mode of the dialogue system
IS_TRAINING_KEY = "is_training"

# key for specifying the path to the nlu unit
NLU_PATH_KEY = "nlu_path"

# key for specifying the path to the nlg unit
NLG_PATH_KEY = "nlg_path"

# key for specifying the intent (act) of the dialogue turn
DIA_ACT_KEY = "diaact"
# key for specifying an inform slot
INFORM_SLOT_KEY = "inform_slot"
# key for specifying a requested slot
REQUEST_SLOT_KEY = "request_slot"
# key for specifying a proposed slot
PROPOSED_SLOT_KEY = "proposed_slot"
# key for specifying an agent requested slot
AGENT_REQUESTED_SLOT_KEY = "agent_request_slots"

# key for specifying a path to the knowledge base
KB_PATH_KEY = "kb_path"
# key for specifying a kb querying result where all of the constraints were matched
KB_MATCHING_ALL_CONSTRAINTS_KEY="matching_all_constraints"


# key for specifying the maximal number of dialogue turns
MAX_NB_TURNS = "max_nb_turns"
