import matplotlib.pyplot as plt
import os
import pickle
import load_data
import numpy as np
import jax.numpy as jnp
import haiku as hk
from mecha_history_plus_lstm import Mecha_history, default_params as mecha_params
from mouse_rnn import LstmAgent
from mecha_history_plus_lstm import Mecha_history_plust_lstm
from rMLPs import RNN
from bi_rnn import BiRNN
import pickle
from infer_decay import Infer_decay
from mirrored_mecha_plus import Mirror_mecha_plus
# from mecha_history_plus_lstm import Mecha_history_plust_lstm


np.set_printoptions(suppress=True)

n_choices = load_data.n_choices

from run_rnn_functions import network_type_to_params

network_type2class = {"<class '__main__.Mecha_history'>": Mecha_history,
                      "<class 'mecha_history.Mecha_history'>": Mecha_history,
                      "<class '__main__.Mecha_history_plust_lstm'>": Mecha_history_plust_lstm,
                      "<class 'mecha_history_plus_lstm.Mecha_history_plust_lstm'>":  Mecha_history_plust_lstm,
                      "<class '__main__.RNN'>": RNN,
                      "<class 'rMLPs.RNN'>": RNN,
                      "<class '__main__.BiRNN'>": BiRNN,
                      "<class 'bi_rnn.BiRNN'>": BiRNN,
                      "<class '__main__.LstmAgent'>": LstmAgent,
                      "<class 'mouse_rnn.LstmAgent'>": LstmAgent,
                      "<class '__main__.Infer_decay'>": Infer_decay,
                      "<class 'infer_decay.Infer_decay'>": Infer_decay,
                      "<class '__main__.Mirror_mecha_plus'>": Mirror_mecha_plus,
                      "<class 'mirrored_mecha_plus.Mirror_mecha_plus'>": Mirror_mecha_plus}

contrasts = [np.array([1, 0]),
             np.array([0.25, 0]),
             np.array([0.125, 0]),
             np.array([0.0625, 0]),
             np.array([0, 0]),
             np.array([0, 0.0625]),
             np.array([0, 0.125]),
             np.array([0, 0.25]),
             np.array([0, 1])]


class loaded_network:

    def __init__(self, infos, use_best=True):
        """
            model_keyword: model setup, whether to use memory, how many hidden units, etc, recover from loaded network
            params: fitted network weights
        """

        self.agent_class = network_type2class[infos["agent_class"]]
        self.params = infos['params_lstm'] if not use_best else infos['best_net']  # TODO: this would be good to use, but a lot of the old code is not set up for it
        # select the necessary kewords out of the infos dict, for the chosen agent_class
        if 'fit_converge_goal' not in infos and self.agent_class == Mecha_history:
            infos['fit_converge_goal'] = False
        self.model_keywords = {key: infos[key] for key in network_type_to_params[infos['agent_class']] if key in infos}
        self.input_list = infos['input_list']
        self.infos = infos

        # get some matrices; TODO: this is dumb: we can just get the class name and search the dict
        if self.agent_class == Mecha_history:
            if 'network_memory' in infos and infos['network_memory']:
                self.input_matrix_1, self.input_bias_1 = self.params['mecha_history/~_habit_manipulator/linear']['w'], self.params['mecha_history/~_habit_manipulator/linear']['b']
                self.input_matrix_2, self.input_bias_2 = self.params['mecha_history/~_habit_manipulator/linear_1']['w'], self.params['mecha_history/~_habit_manipulator/linear_1']['b']
            if 'network_decay' in infos and infos['network_decay']:
                self.decay_matrix_1, self.decay_bias_1 = self.params['mecha_history/~/decay_1']['w'], self.params['mecha_history/~/decay_1']['b']
                self.decay_matrix_2, self.decay_bias_2 = self.params['mecha_history/~/decay_2']['w'], self.params['mecha_history/~/decay_2']['b']
            self.contrast_matrix_1, self.contrast_bias_1 = self.params['mecha_history/~_contrast_mlp/linear']['w'], self.params['mecha_history/~_contrast_mlp/linear']['b']
            self.contrast_matrix_2, self.contrast_bias_2 = self.params['mecha_history/~_contrast_mlp/linear_1']['w'], self.params['mecha_history/~_contrast_mlp/linear_1']['b']
        if self.agent_class == Infer_decay:
            self.contrast_matrix_1, self.contrast_bias_1 = self.params['infer_decay/~_contrast_mlp/linear']['w'], self.params['infer_decay/~_contrast_mlp/linear']['b']
            self.contrast_matrix_2, self.contrast_bias_2 = self.params['infer_decay/~_contrast_mlp/linear_1']['w'], self.params['infer_decay/~_contrast_mlp/linear_1']['b']
        if self.agent_class == BiRNN:
            self.contrast_matrix_1, self.contrast_bias_1 = self.params['bi_rnn/~_contrast_mlp/linear']['w'], self.params['bi_rnn/~_contrast_mlp/linear']['b']
            self.contrast_matrix_2, self.contrast_bias_2 = self.params['bi_rnn/~_contrast_mlp/linear_1']['w'], self.params['bi_rnn/~_contrast_mlp/linear_1']['b']
        if self.agent_class == Mecha_history_plust_lstm:
            if self.infos['network_memory']:
                self.input_matrix_1, self.input_bias_1 = self.params['mecha_history_plust_lstm/~_habit_manipulator/linear']['w'], self.params['mecha_history_plust_lstm/~_habit_manipulator/linear']['b']
                self.input_matrix_2, self.input_bias_2 = self.params['mecha_history_plust_lstm/~_habit_manipulator/linear_1']['w'], self.params['mecha_history_plust_lstm/~_habit_manipulator/linear_1']['b']
            if self.infos['network_decay']:
                self.decay_matrix_1, self.decay_bias_1 = self.params['mecha_history_plust_lstm/~/decay_1']['w'], self.params['mecha_history_plust_lstm/~/decay_1']['b']
                self.decay_matrix_2, self.decay_bias_2 = self.params['mecha_history_plust_lstm/~/decay_2']['w'], self.params['mecha_history_plust_lstm/~/decay_2']['b']

        def _lstm_fn(input_seq, return_all_states=True):
            """Cognitive models function."""

            self.model = self.agent_class(**self.model_keywords)

            batch_size = len(input_seq)
            initial_state = self.model.initial_state(batch_size)

            return hk.dynamic_unroll(
                self.model,
                input_seq,
                initial_state,
                time_major=False,
                return_all_states=return_all_states)
        self.lstm_fn = hk.transform(_lstm_fn)

    def return_predictions(self, input_seq, action_subsetting=False):
        """Cross-entropy loss between model-predicted and input behavior."""

        action_probs_seq, _ = self.lstm_fn.apply(self.params, None, input_seq[:, :, self.input_list])
        # make sure there are no 0 probabilities
        action_probs_seq = (1 - 1e-5) * action_probs_seq + 1e-5 / n_choices

        # calculate loss (NLL aka cross-entropy)
        # "sum" and not "mean" so that missed trials don't influence the results
        action_seq = input_seq[:, 1:, :n_choices]  # first row is empty (all answers were shifted one trial backwards for the input)
        action_probs_seq = action_probs_seq[:, :-1]  # cut out last probability (the last row is just a placeholder for the last possible action)

        if action_subsetting:
            return (action_probs_seq * action_seq).sum(-1)
        else:
            return action_probs_seq

    def nll_fn_lstm(self, input_seq, length_mask):
        """Cross-entropy loss between model-predicted and input behavior."""

        action_probs_seq, _ = self.lstm_fn.apply(self.params, None, input_seq[:, :, self.input_list])
        # make sure there are no 0 probabilities
        action_probs_seq = (1 - 1e-5) * action_probs_seq + 1e-5 / n_choices

        # calculate loss (NLL aka cross-entropy)
        # "sum" and not "mean" so that missed trials don't influence the results
        action_seq = input_seq[:, 1:, :n_choices]  # first row is empty (all answers were shifted one trial backwards for the input)
        action_probs_seq = action_probs_seq[:, :-1]  # cut out last probability (the last row is just a placeholder for the last possible action)
        logs = (jnp.log(action_probs_seq) * action_seq).sum(-1)  # sum out the untaken actions
        nll = -jnp.sum(jnp.where(length_mask, logs, 0)) / length_mask.sum()

        return nll

    def nll_fn_lstm_with_var(self, input_seq, length_mask):
        """Same as above, but with NLL computation for every session individually as well.
            I wasn't able to get this done via just a bool flag, jax complained"""

        action_probs_seq, _ = self.lstm_fn.apply(self.params, None, input_seq[:, :, self.input_list])
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

    # encoding of the last history input
    def input_process(self, input, addon=None):
        if self.infos['network_memory']:
            if self.infos['Mirror_nets']:
                from my_module import mirror_invariant_network
                network = hk.without_apply_rng(hk.transform(mirror_invariant_network()))
                return network.apply({'mirror_invariant_network/~/linear': self.params['mirror_mecha_plus/~_habit_manipulator/mirror_invariant_network/~/linear'],
                                      'mirror_invariant_network/~/linear_1': self.params['mirror_mecha_plus/~_habit_manipulator/mirror_invariant_network/~/linear_1']
                                      }, input, addon=addon)
            else:
                temp = (input[:, None] * self.input_matrix_2).sum(0) + self.input_bias_2
                new = np.tanh(temp)
                return (new[:, None] * self.input_matrix_1).sum(0) + self.input_bias_1
        else:
            return input

    def process(self, action):
        temp = (action[:, None] * self.decay_matrix_2).sum(0) + self.decay_bias_2
        new = np.tanh(temp)
        return (new[:, None] * self.decay_matrix_1).sum(0) + self.decay_bias_1
        
    def mult_apply(self, action, n):
        if self.infos['network_decay']:
            for i in range(n):
                action = self.process(action)
            return action
        else:
            return np.exp(- self.self.params['mecha_history']['decay']) ** n * action * self.self.params['mecha_history']['history_weighting']

    def mult_apply_augmented(self, action, augmenter, n):
        for i in range(n):
            action = self.process(np.append(action, augmenter))
        return action
         

    # # other way around here, because matrices are used in opposite order for decay
    def cont_process(self, contrast):
        temp = (contrast[:, None] * self.contrast_matrix_1).sum(0) + self.contrast_bias_1
        new = np.tanh(temp)
        return (new[:, None] * self.contrast_matrix_2).sum(0) + self.contrast_bias_2
    
    def manual_predict(self, trial, input_seq, s=0):
        total = np.zeros(3)
        for i in range(self.infos['history_slots']):
            history = self.mult_apply(np.zeros(3), trial + 1) if (trial - i) < 0 else \
                    self.mult_apply(self.input_process(input_seq[s, trial - i, self.infos['input_list'][:-2]]), i)
            total += history
        energies = total + self.cont_process(input_seq[s, trial, self.infos['input_list']][-2:])
        probs = np.exp(energies) / np.exp(energies).sum()
        action_probs = (1 - 1e-5) * probs + 1e-5 / 3  # fudging for non-zero probs
        return action_probs, total, energies

    def get_baseline(self):
        zero_energy = self.cont_process(np.array([0, 0]))
        return zero_energy[0] - zero_energy[2]

    def plot_pmf_energies(self, show=True):
        pmf_energies = np.zeros(9)
        for i, c in enumerate(contrasts):
            activities = self.cont_process(c)
            pmf_energies[i] = activities[0] - activities[2]

        print(pmf_energies)
        if self.agent_class == BiRNN:
            pmf_energies = 0.5 * pmf_energies
        plt.plot(pmf_energies)
        plt.axhline(0, c='k')
        plt.axvline(4, c='k')
        if show:
            plt.show() # TODO: only plot if asked, not only show if asked
        else:
            plt.close()
        return pmf_energies

if __name__ == "__main__":
    file = 'mirror_mecha_plus_save_18756433.p'
    infos = pickle.load(open("./222-2_mirror/" + file, 'rb'))
    reload_net = loaded_network(infos)

    input_seq, train_mask, input_seq_test, test_mask, _, _ = load_data.gib_data(file="./processed_data/all_mice.csv")

    print(infos['file'])
    print(infos['train_nll_lstm'])
    print(infos['test_nll_lstm'])
    print(infos['n_training_steps'])

    probs = reload_net.return_predictions(input_seq[:1, :25])

    print(probs[0][:25])
    print(input_seq[0, :25])
    quit()
    print(nll_fn_lstm(self.params, input_seq, train_mask))

    def create_exp_filter(decay, length):
        weights = jnp.exp(- decay * jnp.arange(length))
        weights /= weights.sum()
        return weights


    quit()
    fs = 20
    df = infos['all_scalars']
    plt.figure(figsize=(16, 9))
    plt.plot(df.step, 100 * np.exp(- df.train_nll), label='Train set')
    plt.plot(df.step, 100 * np.exp(- df.test_nll), label='Test set')
    plt.axhline(70.393, c='k', label="GLM test")
    plt.axhline(72.1110463142395, c='g', label="LSTM test")
    plt.axhline(71.92094922065735, c='r', label="BiLSTM test")
    plt.axhline(73.34137, ls='--', c='k', label="Theoretical limit")
    plt.xlim(1000, None)
    plt.ylim(66, None)
    plt.xlabel("Training step", size=fs+4)
    plt.ylabel("Prediction accuracy in %", size=fs+4)
    plt.title(infos['file'] + ' ' + infos['agent_class'], size=fs)
    plt.legend(frameon=False, fontsize=fs)
    plt.tight_layout()
    plt.savefig(infos['file'] + ' ' + infos['agent_class'] + '.png')
    plt.show()

