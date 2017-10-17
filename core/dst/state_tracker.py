"""
Author: Vladimir Ilievski <ilievski.vladimir@live.com>

"""

class GOStateTracker:
    """
    Abstract Base Class of all state trackers in the Goal-Oriented Dialogue Systems.
    
    # Class members:
        - history: list of all dialogue state tracker history
    """

    def __init__(self):
        self.hist = []

    def produce_state(self):
        """
        Abstract method to produce a new belief for the current dialogue state.
        
        :return:
        """
        raise NotImplementedError()

    def update(self, action):
        """
        Abstract method to update the state tracker with the last user or agent action.
        
        :param action: user or agent action
        :return: 
        """

        raise NotImplementedError()


class GORuleBasedStateTracker(GOStateTracker):
    """
    Class for Rule-Based state tracker in the Goal-Oriented Dialogue Systems.
    Extends the [GOStateTracker] class.
    
    # Class members:
    """

    def __init__(self):
        super(GORuleBasedStateTracker, self).__init__()

    def produce_state(self):
        # TODO
        raise NotImplementedError()

    def update(self, action):
        # TODO
        raise NotImplementedError()


class GOModelBasedStateTracker(GOStateTracker):
    """
    Class for Model-Based state tracker in the Goal-Oriented Dialogue Systems.
    Extends the [GOStateTracker] class.
    
    # Class members:
        - is_training: boolean flag indicating the mode of using the model-based state tracker
        - model_path: the path to save or load the model
    """

    def __init__(self, is_training=None, model_path=None):
        super(GOModelBasedStateTracker, self).__init__()

        self.is_training = is_training
        self.model_path = model_path

    def produce_state(self):
        # TODO
        raise NotImplementedError()

    def update(self, action):
        # TODO
        raise NotImplementedError()
