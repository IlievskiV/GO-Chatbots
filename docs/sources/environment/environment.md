<span style="float:right;">[[source]](https://github.com/matthiasplappert/keras-rl/blob/master/core/environment/environment.py#L18)</span>
### GOEnv

```python
core.environment.environment.GOEnv(simulation_mode=None, is_training=False, user_type_str='', user_path='', dst_type_str='', dst_path='', act_set=None, slot_set=None, feasible_actions=None, max_nb_turns=None, nlu_path='', nlg_path='')
```


The Environment with which the agent is interacting with. It extends the keras-rl class Env.
Therefore, the following methods are implemented:

- `step`
- `reset`
- `render`
- `close`
- `seed`
- `configure`

__Class members:__


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

----

<span style="float:right;">[[source]](https://github.com/matthiasplappert/keras-rl/blob/master/core/environment/environment.py#L201)</span>

### step


```python
step(self, action)
```



Method for taking the environment one step further. In this case, to present the agent action to the user in order
to make a response. Overrides the super class method.

:param action: the last agent action 
:return: user's response to the agent's action in form of a state

----

<span style="float:right;">[[source]](https://github.com/matthiasplappert/keras-rl/blob/master/core/environment/environment.py#L243)</span>

### reset


```python
reset(self)
```



Method for resetting the dialogue state tracker and the user, called at the beginning of each new episode.
Overrides the super class method.

:return: the initial observation

----

<span style="float:right;">[[source]](https://github.com/matthiasplappert/keras-rl/blob/master/core/environment/environment.py#L271)</span>

### close


```python
close(self)
```

----

<span style="float:right;">[[source]](https://github.com/matthiasplappert/keras-rl/blob/master/core/environment/environment.py#L276)</span>

### seed


```python
seed(self, seed=None)
```

----

<span style="float:right;">[[source]](https://github.com/matthiasplappert/keras-rl/blob/master/core/environment/environment.py#L281)</span>

### configure


```python
configure(self)
```

