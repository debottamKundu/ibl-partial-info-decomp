"""
    General functions for fitting and evaluating RNN.

    https://github.com/google-deepmind/dm-haiku#why-haiku: Since the actual computation performed by our loss function doesn't rely on random numbers, we pass in None for the rng argument.
    TODO: cut off first trial?
    DONE: If I pass the feedback at the wrong spot, the model should be perfect (modulo 0 contrasts), cause the feedback tells it how the mouse answers -> Tested accidentaly, in fact true
    TODO: Design pickle saving nicer (don't overwrite)
    TODO: save all network params, not just the one in kwargs, so we can survive a default param change
"""
import optax
import matplotlib.pyplot as plt
from ibl_info.lstm_model import load_data
import numpy as np
import pandas as pd
import jax.numpy as jnp
import jax
import haiku as hk
import pickle
import time
from ibl_info.lstm_model.mecha_history_plus_lstm import default_params as mecha_params_plus_lstm
from ibl_info.lstm_model.mouse_rnn import default_params as params_lstm

# from mecha_history import default_params as mecha_params

# from rMLPs import default_params as params_rMLPs
# from bi_rnn import default_params as params_bi_rnns

# from interactive_past_net import default_params as params_interactive
# from bi_rnn_recur import default_params as params_bi_rnn_recur
# from infer_decay import default_params as params_infer_decay
# from bi_lstm_plus import default_params as params_bi_lstm_plus
# from indiv_decay_fit_mecha_hist import default_params as params_indiv_mecha
# from bi_lstm import default_params as params_bi_lstm
# from mirrored_mecha_plus import default_params as params_mirrored_mecha_plus

n_choices = load_data.n_choices

file_prefix = ['.', '/usr/src/app'][1]

network_type_to_params = {
                          "<class '__main__.Mecha_history_plust_lstm'>": mecha_params_plus_lstm,
                          "<class 'mecha_history_plus_lstm.Mecha_history_plust_lstm'>":  mecha_params_plus_lstm,
                          "<class '__main__.LstmAgent'>": params_lstm,
                          "<class 'mouse_rnn.LstmAgent'>": params_lstm,
                        }
                        #   "<class '__main__.Mecha_history'>": mecha_params,
                        #   "<class 'mecha_history.Mecha_history'>": mecha_params,
                        #   "<class '__main__.RNN'>": params_rMLPs,
                        #   "<class 'rMLPs.RNN'>": params_rMLPs,
                        #   "<class '__main__.BiRNN'>": params_bi_rnns,
                        #   "<class 'bi_rnn.BiRNN'>": params_bi_rnns,
                          
                        #   "<class '__main__.Interactive_past'>": params_interactive,
                        #   "<class 'interactive_past_net.Interactive_past'>": params_interactive,
                        #   "<class '__main__.BiRNN_recur'>": params_bi_rnn_recur,
                        #   "<class 'interactive_past_net.BiRNN_recur'>": params_bi_rnn_recur,
                        #   "<class '__main__.Infer_decay'>": params_infer_decay,
                        #   "<class 'infer_decay.Infer_decay'>": params_infer_decay,
                        #   "<class '__main__.BiLSTM_plus'>": params_bi_lstm_plus,
                        #   "<class 'infer_decay.BiLSTM_plus'>": params_bi_lstm_plus,
                        #   "<class 'indiv_decay_fit_mecha_hist.Mecha_history_indiv'>": params_indiv_mecha,
                        #   "<class '__main__.Mecha_history_indiv'>": params_indiv_mecha,
                        #   "<class 'bi_lstm.BiLstmAgent'>": params_bi_lstm,
                        #   "<class '__main__.BiLstmAgent'>": params_bi_lstm,
                        #   "<class 'mirrored_mecha_plus.Mirror_mecha_plus'>": params_mirrored_mecha_plus,
                        #   "<class '__main__.Mirror_mecha_plus'>": params_mirrored_mecha_plus


network_type_to_name = {"<class '__main__.Mecha_history'>": 'mecha_sweep_indep',
                        "<class 'mecha_history.Mecha_history'>": 'mecha_sweep_indep',
                        "<class '__main__.Mecha_history_plust_lstm'>": 'mecha_plus_sweep',
                        "<class 'mecha_history_plus_lstm.Mecha_history_plust_lstm'>":  'mecha_plus_sweep',
                        "<class '__main__.RNN'>": 'rmlp_sweep',
                        "<class 'rMLPs.RNN'>": 'rmlp_sweep',
                        "<class '__main__.BiRNN'>": 'birnn_sweep',
                        "<class 'bi_rnn.BiRNN'>": 'birnn_sweep',
                        "<class '__main__.LstmAgent'>": 'lstm_sweep',
                        "<class 'mouse_rnn.LstmAgent'>": 'lstm_sweep',
                        "<class '__main__.Interactive_past'>": 'interact_sweep',
                        "<class 'interactive_past_net.Interactive_past'>": 'interact_sweep',
                        "<class '__main__.BiRNN_recur'>": 'bi_rnn_recur',
                        "<class 'interactive_past_net.BiRNN_recur'>": 'bi_rnn_recur',
                        "<class '__main__.Infer_decay'>": 'infer_decay',
                        "<class 'infer_decay.Infer_decay'>": 'infer_decay',
                        "<class '__main__.BiLSTM_plus'>": 'bi_lstm_plus',
                        "<class 'infer_decay.BiLSTM_plus'>": 'bi_lstm_plus',
                          "<class 'bi_lstm.BiLstmAgent'>": 'bi_lstm',
                          "<class '__main__.BiLstmAgent'>": 'bi_lstm',
                          "<class 'mirrored_mecha_plus.Mirror_mecha_plus'>": 'mirror_mecha_plus',
                          "<class '__main__.Mirror_mecha_plus'>": 'mirror_mecha_plus'}



# @profile
def initialise_and_train(agent_class, input_list, n_training_steps=5000, learning_rate=1e-3, weight_decay=1e-4, seed=4, train_batch_size=None,
                         file=file_prefix + "/processed_data/all_mice.csv", eval_every=20, save_info=False, **kwargs):

    if file == file_prefix + "/processed_data/all_mice.csv":
        input_seq, train_mask, input_seq_test, test_mask = load_data.gib_data_fast()
    else:
        print(file)
        input_seq, train_mask, input_seq_test, test_mask, _, _ = load_data.gib_data(file=file)

    rng_seq = hk.PRNGSequence(seed)
    optimizer = optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)
    init_opt = jax.jit(optimizer.init)

    def _lstm_fn(input_seq, return_all_states=False):
        """Cognitive models function."""

        model = agent_class(**kwargs)  # TODO: move this outside, so it doesn't get called multiple times?

        batch_size = len(input_seq)
        initial_state = model.initial_state(batch_size)

        return hk.dynamic_unroll(
            model,
            input_seq,
            initial_state,
            time_major=False,
            return_all_states=return_all_states)


    @jax.jit
    def nll_fn_lstm(params, input_seq, length_mask):
        """Cross-entropy loss between model-predicted and input behavior."""

        action_probs_seq, _ = lstm_fn.apply(params, None, input_seq[:, :, input_list])
        # make sure there are no 0 probabilities
        action_probs_seq = (1 - 1e-5) * action_probs_seq + 1e-5 / n_choices

        # calculate loss (NLL aka cross-entropy)
        # "sum" and not "mean" so that missed trials don't influence the results
        action_seq = input_seq[:, 1:, :n_choices]  # first row is empty (all answers were shifted one trial backwards for the input)
        action_probs_seq = action_probs_seq[:, :-1]  # cut out last probability (the last row is just a placeholder for the last possible action)
        logs = (jnp.log(action_probs_seq) * action_seq).sum(-1)  # sum out the untaken actions
        nll = -jnp.sum(jnp.where(length_mask, logs, 0)) / length_mask.sum()

        return nll

    def nll_fn_lstm_with_var(params, input_seq, length_mask):
        """Same as above, but with NLL computation for every session individually as well.
            I wasn't able to get this done via just a bool flag, jax complained"""

        action_probs_seq, _ = lstm_fn.apply(params, None, input_seq[:, :, input_list])
        # make sure there are no 0 probabilities
        action_probs_seq = (1 - 1e-5) * action_probs_seq + 1e-5 / n_choices

        # calculate loss (NLL aka cross-entropy)
        # "sum" and not "mean" so that missed trials don't influence the results
        action_seq = input_seq[:, 1:, :n_choices]  # first row is empty (actions were all shifted backwards)
        action_probs_seq = action_probs_seq[:, :-1]  # cut out last probability (the last row is just a placeholder for the last possible action)
        logs = (jnp.log(action_probs_seq) * action_seq).sum(-1)  # sum out the untaken actions
        nll = -jnp.sum(jnp.where(length_mask, logs, 0)) / length_mask.sum()

        session_nlls = []
        for i in range(logs.shape[0]):
            session_nlls.append(np.mean(logs[i][length_mask[i]]))
        session_nlls = np.array(session_nlls)

        return nll, session_nlls

    @jax.jit
    def update_func(params, opt_state, input_seq, length_mask):
        """Updates function for the RNN."""

        nll, grads = jax.value_and_grad(nll_fn_lstm)(params, input_seq, length_mask)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        return new_params, new_opt_state, nll


    lstm_fn = hk.transform(_lstm_fn)
    params_lstm = lstm_fn.init(next(rng_seq), input_seq[:, :, input_list])
    opt_state = init_opt(params_lstm)

    init_nll = nll_fn_lstm(params_lstm, input_seq, train_mask)
    print(f'Average trialwise probability of initial model to act like mice: {100 * np.exp(-init_nll)}%')

    training_errors = []
    test_errors = []
    steps = []
    params_list = []
    info = {'agent_class': str(agent_class), 'input_list': input_list, 'n_training_steps': n_training_steps,
            'learning_rate': learning_rate, 'weight_decay': weight_decay, 'seed': seed, 'file': file, 'train_batch_size': train_batch_size}
    info.update(network_type_to_params[str(agent_class)])  # fill in default values
    info.update(kwargs)  # overwrite with set variables
    info['best_test_nll'] = 100  # placeholder for comparison
    clash_prevention = np.random.randint(100000000) # not so smart

    numpy_rng = np.random.default_rng(seed)  # reuse seed, what could go wrong


    print(file_prefix + "/results/{}_save_{}_intermediate.p".format(network_type_to_name[str(agent_class)], clash_prevention))
    import time
    a = time.time()
    for current_step in range(n_training_steps):

        if train_batch_size:
            train_indices = numpy_rng.choice(input_seq.shape[0], train_batch_size, replace=False)
            params_lstm, opt_state, nll = update_func(params_lstm, opt_state, input_seq[train_indices], train_mask[train_indices])
        else:
            params_lstm, opt_state, nll = update_func(params_lstm, opt_state, input_seq, train_mask)

        if current_step % eval_every == 0:
            steps.append(current_step)
            # nll = nll_fn_lstm(params_lstm, input_seq, train_mask)  # just evaluating on the batch seems not enough?
            nll_test = nll_fn_lstm(params_lstm, input_seq_test, test_mask)
            training_errors.append(nll)
            test_errors.append(nll_test)
            if nll_test < info['best_test_nll']:
                info['best_net'] = params_lstm
                info['best_test_nll'] = nll_test

        if current_step % 10 == 0:
            print(f'Step {current_step}: test perf. = {100 * np.exp(- info["best_test_nll"])}')
            if save_info:
                all_scalars = pd.DataFrame({'step': steps, 'train_nll': training_errors, 'test_nll': test_errors})
                params_list.append(params_lstm)
                info.update({'all_scalars': all_scalars, 'params_list': params_list})
        if current_step != 0 and (current_step % 100000 == 0 or current_step + 1 == n_training_steps) and save_info:
            pickle.dump(info, open(file_prefix + "/results/{}_save_{}_intermediate.p".format(network_type_to_name[str(agent_class)], clash_prevention), 'wb'))
    print("Loop time {:.4f}".format(time.time() - a))
    
    final_nll = nll_fn_lstm(params_lstm, input_seq, train_mask)
    all_scalars = pd.DataFrame({'step': steps, 'train_nll': training_errors, 'test_nll': test_errors})

    print(f'Average trialwise probability of final model to act like mice: {100 * np.exp(-final_nll)}%')

    all_scalars.reset_index(drop=True, inplace=True)

    for col in all_scalars.columns:
        all_scalars[col] = all_scalars[col].astype(float)

    print("Used inputs: {}".format([load_data.input_list[i] for i in input_list]))

    train_nll_lstm = nll_fn_lstm(params_lstm, input_seq, train_mask)
    print(f'{agent_class} prediction accuracy on training data: {100 * np.exp(-train_nll_lstm)}%')

    test_nll_lstm, session_nlls = nll_fn_lstm_with_var(params_lstm, input_seq_test, test_mask)
    print(f'{agent_class} prediction accuracy on held-out data: {100 * np.exp(-test_nll_lstm)}%')

    if save_info:
        del info['params_list']  # final results should be slimmer
        info.update({'train_nll_lstm': train_nll_lstm, 'test_nll_lstm': test_nll_lstm, 'all_scalars': all_scalars, 'session_nlls': session_nlls, 'params_lstm': params_lstm})
        pickle.dump(info, open(file_prefix + "/results/{}_save_{}.p".format(network_type_to_name[str(agent_class)], clash_prevention), 'wb'))

    return params_lstm