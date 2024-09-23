"""
    Defining basic RNN.

    Q: how to see into the happenings more neatly to check e.g. the data?
    Best LSTM: Step 9900: nll = 0.3563637137413025
    LSTM prediction accuracy on training data: 70.02490758895874%
    LSTM prediction accuracy on held-out data: 70.1951801776886%

    https://datascience.stackexchange.com/questions/82808/whats-the-difference-between-the-cell-and-hidden-state-in-lstm
    https://medium.com/analytics-vidhya/lstms-explained-a-complete-technically-accurate-conceptual-guide-with-keras-2a650327e8f2
"""
import jax.numpy as jnp
import jax
import haiku as hk

default_params = {'n_hiddens': 5,
                  'n_choices': 3,
                  'memory': True}

class LstmAgent(hk.RNNCore):
  """LSTM that predicts action logits based on all inputs (action, reward, stimulus)."""

  def __init__(self, **kwargs):

    super().__init__()

    self._hidden_size = kwargs.get('n_hiddens', default_params['n_hiddens'])
    self._n_actions = kwargs.get('n_choices', default_params['n_choices'])  # in IBL, there are always only 3 choices (two relevant ones really)
    self.memory = kwargs.get('memory', default_params['memory'])


  def __call__(self, inputs: jnp.array, prev_state):

    hidden_state, cell_state = prev_state[0] * self.memory, prev_state[1] * self.memory

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

    action_probs = jax.nn.softmax(
        hk.Linear(self._n_actions)(next_hidden_state))  # (batch_size, n_act)

    return action_probs, (next_hidden_state, next_cell_state)

  def initial_state(self, batch_size):

    return (jnp.zeros((batch_size, self._hidden_size)),  # hidden_state
            jnp.zeros((batch_size, self._hidden_size)))  # cell_state
  

if __name__ == "__main__":
  import run_rnn_functions
  from itertools import product
  import sys

  if len(sys.argv) > 1:
    constellations = list(product([100, 101, 102, 103], [8, 16, 32, 64], [4, 8, 16, 32], [1e-3, 1e-4, 1e-5], [1e-3, 1e-4, 1e-5]))
    seed, train_batch_size, n_hiddens, learning_rate, weight_decay = constellations[int(sys.argv[1])]
    input_list = [0, 1, 2, 3, 4, 5]
    print("LSTM agent (input: {})".format(input_list))
    print(seed, train_batch_size, n_hiddens, learning_rate, weight_decay)
    run_rnn_functions.initialise_and_train(agent_class=LstmAgent, input_list=input_list, save_info=True,
                                           n_training_steps=400000, seed=seed, train_batch_size=train_batch_size, n_hiddens=n_hiddens,
                                           learning_rate=learning_rate, weight_decay=weight_decay)
  else:
    for memory, input_list in product([True], [[0, 1, 2], [3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4, 5]]): 
      print("LSTM agent (memory: {}, inputs: {})".format(memory, input_list))
      run_rnn_functions.initialise_and_train(agent_class=LstmAgent, input_list=input_list, save_info=True, memory=memory,
                                             n_training_steps=12000)  # , file='./processed_data/bayesian_observer_noise_0.03.csv'
