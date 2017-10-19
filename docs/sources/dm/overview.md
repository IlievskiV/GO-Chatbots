<span style="float:right;">[[source]](https://github.com/matthiasplappert/keras-rl/blob/master/core/dm/dialogue_system.py#L12)</span>
### GODialogSys

```python
core.dm.dialogue_system.GODialogSys(act_set=None, slot_set=None, agt_feasible_actions=None, params=None)
```


The GO Dialogue System mediates the interaction between the environment and the agent.

__Class members:__


- agent: the type of conversational agent. Default is None (temporarily).
- env: the environment with which the agent and user interact. Default is None (temporarily).
- act_set: the set of all intents used in the dialogue.
- slot_set: the set of all slots used in the dialogue.
- kb_path: path to any knowledge base
- agt_feasible_actions: list of templates described as dictionaries, corresponding to each action the agent might take
			(dict to be specified)
- max_nb_turns: the maximal number of dialogue turns


