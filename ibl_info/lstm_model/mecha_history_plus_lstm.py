"""Define explicit history plus MLP networks.
Contrast handling is taken care of by a separate MLP. History is tracked explicitely, and decayed in various ways (let's see what we can do in one class)

https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-recurrent-neural-networks
TODO: Do I want to scale softmaxes with a temp? -> I don't think this is necessary, network and history weight can handle that
TODO: better names
TODO: history slot 1 or 2 need special treatment or reshaping
!!!TODO: network decay works on the bunch of 0-vectors with which we initialise, leading to weird results!!!
TRY: Access to all of history, not just the sum
TODO: For extended input, current contrast seems to do better than previous contrast when saving action history vector?
TODO: Now that we are considering the first trial, the habit input that part of the network gets is total nonsense!
TODO: only pass motviational state to some networks?
"""
from typing import Optional

import haiku as hk
import jax
import jax.numpy as jnp

RNNState = jnp.array

def create_exp_filter(decay, length):
  """
    Create an exponential filter.
    Since the network learns a weight with which to multiply the outcome, we remove the normalisation, this will make the decay constant more interpretable.
  """
  weights = jnp.exp(- decay * jnp.arange(length))
  return weights[::-1]

default_params = {'n_hiddens': 5,
                  'n_contrast_hiddens': 5,
                  'n_encode_hidden': None,
                  'n_decay_hidden': None,
                  'n_decode_hidden': None,
                  'n_choices': 3,
                  'history_slots': 5,
                  'memory_size': 3,
                  'network_contrast': True,
                  'network_memory': False,  # why False?
                  'network_decay': False,
                  'pass_to_cont_net': True,
                  'pass_to_enc_net': True,
                  'pass_to_decay_net': True,
                  'do_history_inf': True,
                  'lstm_dim': 1,
                  'LSTM_out_split': False,
                  'LSTM_given': False,
                  'Pass_reward_scalar': False,
                  'pass_to_history_limit': None,
                  'separate_decay': False,
                  'free_filter': False,
                  'network_combo': False,
                  'fit_history_init': False,
                  'fit_converge_goal': False,
                  'contextual_update': False}

class Mecha_history_plust_lstm(hk.RNNCore):
  """A bifurcating RNN: "history" processes previous (block) information; "contrast" contrasts."""

  def __init__(self, **kwargs):
    """
      n_hiddens, int - number of hidden units for contrast network
      n_choices, int - number of possible actions, dimension of output
      history_slots, int or 'infinite' - number of history vectors to store, or filter over the entire past if 'infinite'
    """

    super().__init__()

    self._hidden_size = kwargs.get('n_hiddens', default_params['n_hiddens'])
    self.n_contrast_hiddens = kwargs.get('n_contrast_hiddens', default_params['n_contrast_hiddens'])
    self.n_encode_hidden = self._hidden_size if 'n_encode_hidden' not in kwargs else kwargs['n_encode_hidden']
    self.n_decay_hidden = self._hidden_size if 'n_decay_hidden' not in kwargs else kwargs['n_decay_hidden']
    self.n_decode_hidden = self._hidden_size if 'n_decode_hidden' not in kwargs else kwargs['n_decode_hidden']
    self._n_actions = kwargs.get('n_choices', default_params['n_choices'])
    self._history_slots = kwargs.get('history_slots', default_params['history_slots'])
    self.memory_size = kwargs.get('memory_size', default_params['memory_size'])

    self.network_memory = kwargs.get('network_memory', default_params['network_memory'])
    self.network_decay = kwargs.get('network_decay', default_params['network_decay'])
    self.network_contrast = kwargs.get('network_contrast', default_params['network_contrast'])

    self.pass_to_cont_net = kwargs.get('pass_to_cont_net', default_params['pass_to_cont_net'])
    self.pass_to_enc_net = kwargs.get('pass_to_enc_net', default_params['pass_to_enc_net'])
    self.pass_to_decay_net = kwargs.get('pass_to_decay_net', default_params['pass_to_decay_net'])
    self.do_history_inf = kwargs.get('do_history_inf', default_params['do_history_inf'])
    self.lstm_dim = kwargs.get('lstm_dim', default_params['lstm_dim'])
    self.LSTM_out_split = kwargs.get('LSTM_out_split', default_params['LSTM_out_split'])
    self.LSTM_given = kwargs.get('LSTM_given', default_params['LSTM_given'])
    self.Pass_reward_scalar = kwargs.get('Pass_reward_scalar', default_params['Pass_reward_scalar'])
    self.pass_to_history_limit = kwargs.get('pass_to_history_limit', default_params['pass_to_history_limit'])

    self.separate_decay = kwargs.get('separate_decay', default_params['separate_decay'])
    self.free_filter = kwargs.get('free_filter', default_params['free_filter'])  # use an array instead of a fitted exp_filter
    self.network_combo = kwargs.get('network_combo', default_params['network_combo'])  # whether to combine info with network, or simply additively
    self.fit_history_init = kwargs.get('fit_history_init', default_params['fit_history_init'])
    self.fit_converge_goal = kwargs.get('fit_converge_goal', default_params['fit_converge_goal'])
    self.contextual_update = kwargs.get('contextual_update', default_params['contextual_update'])

    # print the params which actually made it
    print("Mecha_history_plus_lstm agent")
    for key in default_params:
      print(key + ": " + str(kwargs.get(key, default_params[key])))

    # set some flags and check logic
    assert not (self.separate_decay and not self.network_decay), "Can't have separate decay without network decay"
    assert not (self.free_filter and self.network_decay), "Network decay precludes a free filter"
    assert not (self.memory_size != 3 and not self.network_memory), "To fill memory != 3, need network memory"
    assert not (self._history_slots == 'infinite' and self.free_filter), "Cannot have infinite history with free filter"
    assert not (self._history_slots == 'infinite' and self.separate_decay), "Cannot have infinite history with separate decay"
    assert not (self.fit_converge_goal and self.network_decay), "Converge_goal is only for non-network decay"
    assert not (self.contextual_update and self._history_slots != 'infinite'), "Contextual updates currently require a single, infinite memory"
    assert not (self.contextual_update and not self.network_memory), "Contextual updates requires network input encoding"
    assert not (self.pass_to_decay_net and not self.network_decay), "Cannot pass LSTM scalar to decay net without decay net"
    assert not self.LSTM_out_split or self.lstm_dim > 1, "If LSTM output is split, lstm_dim must be 2 or more"
    assert not (self.lstm_dim > 1 and self.do_history_inf) or self.LSTM_out_split, "Cannot have multiple LSTM dimensions with history inference at the moment"

    if self.free_filter:
      self.filter = hk.get_parameter("filter", shape=[self._history_slots], init=jnp.ones)
    else:
      # TODO: this is not always used, remove it when not
      self.decay = hk.get_parameter("decay", shape=[1], init=jnp.ones)
    if self.network_decay:
      if not self.separate_decay:  # only use one network for decaying things, otherwise instantiate new ones in habit_manipulator
        self.decay_1 = hk.Linear(self.memory_size, name='decay_1')
        self.decay_2 = hk.Linear(self.n_decay_hidden, name='decay_2')

    if self.free_filter or self.network_memory or self.network_combo:
      self.history_weighting = 1
    else:
      self.history_weighting = hk.get_parameter("history_weighting", shape=[1], init=jnp.ones)

    if self.fit_history_init:
      self.history_initialisation = hk.get_parameter("history_initialisation", shape=[self.memory_size], init=jnp.zeros)
    else:
      self.history_initialisation = jnp.zeros(self.memory_size)

    if self.fit_converge_goal:
      self.converge_goal = hk.get_parameter("converge_goal", shape=[self.memory_size], init=jnp.zeros)
    else:
      self.converge_goal = jnp.zeros(self.memory_size)

    if self.Pass_reward_scalar:
      self.reward_decay = hk.get_parameter("reward_decay", shape=[1], init=jnp.ones)
      self.easy_decay = hk.get_parameter("easy_decay", shape=[1], init=jnp.ones)

  def _state_lstm(self, inputs: jnp.array, prev_state):

    hidden_state, cell_state = prev_state

    forget_gate = jax.nn.sigmoid(
        hk.Linear(self._hidden_size, with_bias=False)(inputs)  # https://dm-haiku.readthedocs.io/en/latest/api.html#linear
        + hk.Linear(self._hidden_size)(hidden_state)
    )

    input_gate = jax.nn.sigmoid(
        hk.Linear(self._hidden_size, with_bias=False)(inputs)
        + hk.Linear(self._hidden_size)(hidden_state)
    )

    candidates = jax.nn.tanh(
        hk.Linear(self._hidden_size, with_bias=False)(inputs)
        + hk.Linear(self._hidden_size)(hidden_state)
    )

    next_cell_state = forget_gate * cell_state + input_gate * candidates

    output_gate = jax.nn.sigmoid(
        hk.Linear(self._hidden_size, with_bias=False)(inputs)
        + hk.Linear(self._hidden_size)(hidden_state)
    )
    next_hidden_state = output_gate * jax.nn.tanh(next_cell_state)

    motivational_state = hk.Linear(self.lstm_dim)(next_hidden_state)  # (batch_size, 1), just a 1D motivational state

    return motivational_state, (next_hidden_state, next_cell_state)

  def _contrast_mlp(self, contrast):

    next_state = jax.nn.tanh(hk.Linear(self.n_contrast_hiddens)(contrast))
    next_gist = hk.Linear(self._n_actions)(next_state)

    return next_gist
  
  def _contrast_regression(self, contrast):

    contrast_transform = jnp.tanh(5 * contrast) / jnp.tanh(5)   # the Nick Roy standard, could make 5 learnable
    next_gist = hk.Linear(self._n_actions, with_bias=True)(contrast_transform)

    return next_gist

  # @profile
  def _habit_manipulator(self, action, history_safe):
    if self.network_memory:
      if self.contextual_update:
        update_with = hk.Linear(self.memory_size)(jax.nn.tanh(hk.Linear(self.n_encode_hidden)(jnp.concatenate([action, history_safe], axis=1))))
      else:
        update_with = hk.Linear(self.memory_size)(jax.nn.tanh(hk.Linear(self.n_encode_hidden)(action)))
    else:
      update_with = action

    if self._history_slots == 'infinite':
      if self.contextual_update:
        # remove motivational state again
        updated = update_with + history_safe[:, -1:]
        return updated, updated
      else:
        if self.network_decay:
          # could also do a current dependent update, by also throwing in current contrast... but this makes things a whole lot more complicated, 
          # by allowing complex saving structures
          updated = update_with + self.decay_1(jax.nn.tanh(self.decay_2(history_safe)))
          return updated, updated
        else:
          updated = update_with + jnp.exp(- self.decay) * history_safe + (1- jnp.exp(- self.decay)) * self.converge_goal
          return updated, updated
    else:
      if self.network_decay:
        res = []
        if not self.separate_decay:
          res = [self.decay_1(jax.nn.tanh(self.decay_2(history_safe[:, 1:])))]
        else:
          for i in range(self._history_slots - 1):
            res.append(hk.Linear(self.memory_size)(jax.nn.tanh(hk.Linear(self.n_decay_hidden)(history_safe[:, i + 1])))[:, None])
        new_safe = jnp.concatenate(res + [update_with[:, None]], axis=1)  # Loop time 109.0156
        
        # history_safe = jax.lax.dynamic_update_slice(history_safe, res, (0, 0, 0))  # working solution, but also slow Loop time 103.6152
        # history_safe = jax.lax.dynamic_update_slice(history_safe, update_with[:, None], (0, self._history_slots - 1, 0))

        return new_safe.sum(1), new_safe
      else:
        if self.free_filter:
          local_filter = self.filter
        else:
          local_filter = create_exp_filter(self.decay, self._history_slots)
        new_safe = jnp.concatenate([history_safe[:, 1:], update_with[:, None]], axis=1)

        return (new_safe * local_filter[None, :, None]).sum(1) + (1 - local_filter).sum() * self.converge_goal, new_safe

  def __call__(self, inputs: jnp.array, total_state: RNNState) -> tuple[jnp.array, RNNState]:

    history_safe, prev_state = total_state
    if not self.LSTM_given:
        action = inputs[:, :-2]  # shape: (batch_size, n_actions), grab everything before the contrasts, actions and rewards
    else:
        action = inputs[:, :-3]
        motivational_state = inputs[:, -3:-2]
        lstm_state = prev_state
    contrast = inputs[:, -2:]  # shape: (batch_size, 2), grab the two contrasts

    # state module: update motivational state to pass to other networks
    if not self.LSTM_given:
        if self.Pass_reward_scalar:
          # [0, 1, 2, 7, 8, 5, 3, 4]
          prev_decayed_reward, prev_decayed_easy = prev_state
          decayed_reward = prev_decayed_reward * jnp.exp(- self.reward_decay) + action[:, -1]
          easy_prev_cont = action[:, [3, 4]].sum(1) == 1  # these are the previous contrasts, check whether it was a full strength one
          # we do some power shenanigans to only decay when it was an easy contrast
          decayed_easy = prev_decayed_easy * (jnp.exp(- self.easy_decay) ** easy_prev_cont) + action[:, -1] * easy_prev_cont

          motivational_state = jnp.array([decayed_reward, decayed_easy]).T
          lstm_state = (decayed_reward, decayed_easy)
        else:
          motivational_state, lstm_state = self._state_lstm(action, prev_state)

        if self.LSTM_out_split:
          contrast_mot, history_mot = jnp.split(motivational_state, [1], axis=1)
        else:
          contrast_mot = history_mot = motivational_state

    # Contrast module: compute contrast influence
    if self.network_contrast:
      next_contrast_gist = self._contrast_mlp(jnp.concatenate([contrast, contrast_mot], axis=-1) if self.pass_to_cont_net else contrast)
    else:
      next_contrast_gist = self._contrast_regression(contrast)

    # history_mot = jnp.full(history_mot.shape, 0.9)  # if in need of fixing scalar
    # History module: update/create block inference
    if self._history_slots == 'infinite':
      augmented_motivation = history_mot
    else:
      # this needs special treatment
      augmented_motivation = jnp.tile(history_mot, self._history_slots)[..., None]
    if self.do_history_inf:
        history_summary, history_safe = self._habit_manipulator(jnp.concatenate([action[:, :self.pass_to_history_limit], history_mot], axis=-1) if self.pass_to_enc_net else action[:, :self.pass_to_history_limit],
                                                                jnp.concatenate((history_safe, augmented_motivation), axis=-1) if self.pass_to_decay_net else history_safe)
    else:
        history_summary = jnp.zeros(self._n_actions)

    # Combine value and habit
    if self.network_combo:
      hv_combo = hk.Linear(self._n_actions)(jax.nn.tanh(hk.Linear(self.n_decode_hidden)(jnp.concatenate([next_contrast_gist, history_summary], axis=-1))))
    else:
      if self.memory_size != 3:
        processed_history = hk.Linear(self._n_actions)(jax.nn.tanh(hk.Linear(self.n_decode_hidden)(history_summary)))
      else:
        processed_history = self.history_weighting * history_summary
      hv_combo = next_contrast_gist + processed_history  # (bs, n_a)

    action_probs = jax.nn.softmax(hv_combo)  # (bs, n_a)

    return action_probs, (history_safe, lstm_state)

  def initial_state(self, batch_size: Optional[int]) -> RNNState:
    self.batch_size = batch_size
    if self._history_slots == 'infinite':
      if not self.Pass_reward_scalar:
        return (jnp.tile(self.history_initialisation, (batch_size, 1)), (jnp.zeros((batch_size, self._hidden_size)), jnp.zeros((batch_size, self._hidden_size))))
      else:  # we need a smaller array if we just use the reward filter as the motivational state
        return (jnp.tile(self.history_initialisation, (batch_size, 1)), (jnp.zeros((batch_size)), jnp.zeros((batch_size))))
    else:
      return (jnp.tile(self.history_initialisation, (batch_size, self._history_slots, 1)), (jnp.zeros((batch_size, self._hidden_size)), jnp.zeros((batch_size, self._hidden_size))))

# history_slots: 3, 6, 12, 20, 'infinite'
# input_list = [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4, 5, 3, 4], [0, 1, 2, 7, 8, 5, 3, 4]]


constellations = [{  # network decay is not necessary for contextual update
                'history_slots': 'infinite',
                'input_list': [0, 1, 2, 7, 8, 5, 3, 4],
                'network_memory': True,
                'network_decay': True,
            }]

pass_settings = [(True, False, False), (False, True, False), (False, False, True), (False, True, True), (True, False, True), (True, True, False)]

if __name__ == "__main__":
  import run_rnn_functions
  from itertools import product
  import sys

  if len(sys.argv) > 1:
    constellation = constellations[0]
    sweep_params = list(product([100, 101, 102, 103], [64, 16, 8], [8], [16], [8], [8], [1e-3], [1e-3, 1e-4, 1e-5]))
    seed, train_batch_size, n_hiddens, n_contrast_hiddens, n_encode_hidden, n_decay_hidden, learning_rate, weight_decay = sweep_params[int(sys.argv[1])]
    # if seed in [100, 101, 102] and train_batch_size in [8, 16, 64] and n_hiddens in [8] and n_contrast_hiddens in [8, 16] and weight_decay in [1e-3, 1e-4, 1e-5]:
    #     quit()
    pass_to_cont_net, pass_to_enc_net, pass_to_decay_net = True, True, True
    pass_to_history_limit = None
    do_history_inf = True
    lstm_dim = 1
    LSTM_out_split = False

    Pass_reward_scalar = True
    LSTM_given = False
    # if n_hiddens == n_encode_hidden == n_decay_hidden:
    #     quit()
    print("Mecha_history plus LSTM agent ()".format(constellation))
    print(seed, train_batch_size, n_hiddens, n_contrast_hiddens, n_encode_hidden, n_decay_hidden, learning_rate, weight_decay)
    print(do_history_inf, pass_to_history_limit, lstm_dim, LSTM_out_split)
    # assert not (constellation['memory_size'] == 3 and (not constellation['network_memory']) and (len(constellation['input_list']) > 5)), "Cannot put all the information into small memory without encoding network"
    run_rnn_functions.initialise_and_train(agent_class=Mecha_history_plust_lstm, save_info=True, n_training_steps=400000, seed=seed,
                                           train_batch_size=train_batch_size, n_hiddens=n_hiddens, n_contrast_hiddens=n_contrast_hiddens,
                                           learning_rate=learning_rate, weight_decay=weight_decay,
                                           n_encode_hidden=n_encode_hidden, n_decay_hidden=n_decay_hidden, LSTM_given=LSTM_given,
                                           do_history_inf=do_history_inf, pass_to_history_limit=pass_to_history_limit, lstm_dim=lstm_dim, LSTM_out_split=LSTM_out_split,
                                           Pass_reward_scalar=Pass_reward_scalar,
                                           pass_to_cont_net=pass_to_cont_net, pass_to_enc_net=pass_to_enc_net, pass_to_decay_net=pass_to_decay_net,
                                           **constellation)