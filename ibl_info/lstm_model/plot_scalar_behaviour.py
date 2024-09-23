"""
    Extracts of the code from network_ana_1d_latent.py, to extract relevant info about LSTM scalar and how it influences the behaviour of the nets
"""
import pickle
from load_network import loaded_network
import numpy as np
import load_data
import matplotlib.pyplot as plt
import seaborn as sns
import json
from scipy.stats import gaussian_kde
import imageio
from load_network import contrasts
contrasts = np.array(contrasts)

def get_all_scalars_fast(reload_net, input, mask, infos):
    # Iterate over all sessions, saving the scalar
    all_scalar = []

    results = reload_net.lstm_fn.apply(reload_net.params, None, input[:, :, reload_net.input_list])
    hidden_states = results[1][1][0]  # extract the exact hidden states of the LSTM

    if infos['agent_class'] == "<class '__main__.Mirror_mecha_plus'>":
        mot_matrix, mot_bias = reload_net.infos['params_lstm']['mirror_mecha_plus/~_state_lstm/linear_8']['w'], reload_net.infos['params_lstm']['mirror_mecha_plus/~_state_lstm/linear_8']['b']
    else:
        mot_matrix, mot_bias = reload_net.infos['params_lstm']['mecha_history_plust_lstm/~_state_lstm/linear_8']['w'], reload_net.infos['params_lstm']['mecha_history_plust_lstm/~_state_lstm/linear_8']['b']
    if 'lstm_dim' not in infos or infos['lstm_dim'] == 1:
        mot_matrix = mot_matrix[:, 0]  # need to get rid of the flat last dimension for this

    for sess in range(input.shape[0]):

        if 'lstm_dim' not in infos or infos['lstm_dim'] == 1:
            motivations = (hidden_states[sess, :mask[sess].sum()] * mot_matrix).sum(1) + mot_bias
        else:
            motivations = (hidden_states[sess, :mask[sess].sum()] @ mot_matrix) + mot_bias  # TODO: here as well this clause might always work if not doing earlier if?

        all_scalar.append(motivations)

    return all_scalar

def get_block_switch_differences(biased_blocks, count_up_to=10):
    # biased_blocks has shape (n_sessions, n_trials), with entries 0.5 for neutral block, 0.2 for leftwards and 0.8 for rightwards
    # we want to find out how far away each trial is from the previous block switch (0.2 to 0.8 or vice versa)
    # count_up_to is the maximum number of trials to count up to from the last switch
    # this reports 1 on the first trial of a new block

    trial_list = []
    distances_from_last_switch = np.zeros_like(biased_blocks) + count_up_to  # Initialize the array with -1 to indicate that no switch has occurred yet
    for session in range(biased_blocks.shape[0]):
        last_switch_idx = None  # To track the index of the last block switch
        
        # Loop through each trial in the session
        for trial in range(biased_blocks.shape[1] - 1):
            if last_switch_idx is not None:
                # Calculate the distance from the last switch
                distances_from_last_switch[session, trial] = min(trial - last_switch_idx, count_up_to)
            
            # Check if the current trial is a block switch
            if trial > 0 and (biased_blocks[session, trial + 1] == 0.2 and biased_blocks[session, trial] == 0.8) or \
            (biased_blocks[session, trial + 1] == 0.8 and biased_blocks[session, trial] == 0.2):
                last_switch_idx = trial  # Update the last switch index

            if biased_blocks[session, trial] == 0.:
                trial_list.append(distances_from_last_switch[session, :trial])
                break
        else:
            trial_list.append(distances_from_last_switch[session])

    return trial_list

def return_standard_pmf():
    # PMF of best 111
    file = "mecha_sweep_save_6738724.p"
    ind = 19180
    nll = 70.8884806443464

    intermediates = pickle.load(open("./best_nets_params/" + file[:-2] + "_intermediate.p", 'rb'))
    infos = pickle.load(open("./best_nets_params/" + file, 'rb'))
    if 'params_list' in infos:
        intermediates['params_list'] = infos['params_list']
    infos['params_lstm'] = intermediates['params_list'][ind // (infos['all_scalars'].shape[0] // len(intermediates['params_list']))]
    reload_net_standard = loaded_network(infos, use_best=False)

    computed_nll = 100 * np.exp(np.log(reload_net_standard.return_predictions(test_input_seq, action_subsetting=True)[test_test_mask]).mean())
    print(nll, computed_nll)
    assert np.allclose(nll, computed_nll)

    standard_pmf = reload_net_standard.plot_pmf_energies(show=False)
    return standard_pmf

def return_standard_history():
    # history of best 111
    file = "mecha_sweep_save_6738724.p"
    ind = 19180
    nll = 70.8884806443464

    intermediates = pickle.load(open("./best_nets_params/" + file[:-2] + "_intermediate.p", 'rb'))
    infos = pickle.load(open("./best_nets_params/" + file, 'rb'))
    if 'params_list' in infos:
        intermediates['params_list'] = infos['params_list']
    infos['params_lstm'] = intermediates['params_list'][ind // (infos['all_scalars'].shape[0] // len(intermediates['params_list']))]
    reload_net_standard = loaded_network(infos, use_best=False)

    computed_nll = 100 * np.exp(np.log(reload_net_standard.return_predictions(test_input_seq, action_subsetting=True)[test_test_mask]).mean())
    print(nll, computed_nll)
    assert np.allclose(nll, computed_nll)

    results = reload_net_standard.lstm_fn.apply(reload_net_standard.params, None, input_seq[:, :, reload_net_standard.input_list])
    return results[1]

def plot_pmf_panels(long_array, cont_process, z, title=None, motivation_position=None, motivation_density=None):
    # given the array of scalars, a function to process the contrasts, and the density estimator z, plot the PMFs as they cover the distribution of scalars
    motivation_covering = np.quantile(long_array, [0.125, 0.4, 0.6, 0.875]) #, np.linspace(0.01, 0.99, 10))
    standard_pmf = return_standard_pmf()

    plt.figure(figsize=(16*0.85, 9*0.85))
    for j, mot in enumerate(motivation_covering):
        augmented_contrast = np.hstack((contrasts, np.full((contrasts.shape[0], 1), mot)))

        pmf_energies = np.zeros(9)
        for i, c in enumerate(augmented_contrast):
            activities = cont_process(c)
            pmf_energies[i] = activities[0] - activities[2]
        plt.subplot(2, int(len(motivation_covering) / 2), j+1)
        label = "Augmented PMF (222)" if j == 0 else ""
        plt.plot(pmf_energies, label=label)
        label = "Static PMF (111)" if j == 0 else ""
        plt.plot(standard_pmf, 'k', alpha=0.2, label=label)

        # plt.title("Scalar: {:.3f}\nDensity est.: {:.3f}".format(mot, z(mot)[0] if z is not None else 0), size=16)
        plt.axhline(0, c='k', alpha=0.1)
        plt.axvline(4, c='k', alpha=0.1)
        plt.ylim(-6.5, 6.5)
        plt.xlim(0, 8)
        plt.xticks([0, 2, 4, 6, 8], [-1, -0.125, 0, 0.125, 1])
        plt.gca().tick_params(axis='both', which='major', labelsize=22)

        if j not in [0, 2]:
            plt.yticks([])
        if j not in [2, 3]:
            plt.xticks([])

        sns.despine()
        if j == int(len(motivation_covering) / 2):
            plt.ylabel("Right - left logit", size=29)
            plt.xlabel("Contrast", size=29)
        if j == 0:
            plt.legend(frameon=False, fontsize=23)

        if motivation_position is not None:
            ax2 = plt.gca().inset_axes([0.6, 0.1, 0.33, 0.33])
            ax2.plot(motivation_position, motivation_density, 'k')
            ax2.set_ylim(0)
            ax2.axvline(mot, color='k', alpha=0.4)
            sns.despine(ax=ax2)

    plt.tight_layout()
    if title is not None:
        plt.savefig("pmf_development {}".format(title), dpi=300)
    plt.show()

def create_pmf_gif(cont_process, motivation_position, motivation_density, net_name):
    files = []
    standard_pmf = return_standard_pmf()
    for i, augmenter in enumerate(motivation_position):
        fig = plt.figure(figsize=(16*1.2, 9*1.2))
        augmented_contrast = np.hstack((contrasts, np.full((contrasts.shape[0], 1), augmenter)))

        pmf_energies = np.zeros(9)
        for j, c in enumerate(augmented_contrast):
            activities = cont_process(c)
            pmf_energies[j] = activities[0] - activities[2]
        label = "Modified PMF"
        plt.plot(pmf_energies, label=label, lw=2)
        label = "Static PMF"
        plt.plot(standard_pmf, 'k', alpha=0.2, label=label)

        plt.axhline(0, c='k', alpha=0.1)
        plt.axvline(4, c='k', alpha=0.1)
        plt.ylim(-6.5, 6.5)
        plt.xticks([0, 2, 4, 6, 8], [-1, -0.125, 0, 0.125, 1])
        sns.despine()
        plt.ylabel("Right - left logit", size=54)
        plt.xlabel("Contrast", size=54)
        plt.gca().tick_params(axis='both', which='major', labelsize=18)

        plt.tight_layout()

        ax1 = plt.gca()
        ax2 = fig.add_axes([0.1, 0.7, 0.25, 0.25])
        ax2.plot(motivation_position, motivation_density, 'k')
        ax2.set_ylim(0)
        ax2.axvline(augmenter, color='k', alpha=0.4)
        sns.despine()

        plt.savefig("gif_images_2/pmf_development_{}_{}".format(net_name, i))
        files.append("gif_images_2/pmf_development_{}_{}".format(net_name, i))
        plt.close()

    print('gif')
    images = []

    for file in files:
        images.append(imageio.imread(file + '.png'))
    for file in files[::-1]:
        images.append(imageio.imread(file + '.png'))

    print('saving')
    imageio.mimsave("pmf_{}.gif".format(net_name), images, format='GIF', fps=30, loop=0)


def scatter_scalars(long_array, color='b'):
    # scatter the two scalars against one another, plot a linear regression line through them and put the r^2 value in the title
    from scipy.stats import linregress, pearsonr
    plt.figure(figsize=(14, 8))

    x, y = long_array[:, 0], long_array[:, 1]
    plt.scatter(x, y, alpha=0.1, c=color)

    # linear regression
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    plt.plot([x.min(), x.max()], [slope * x.min() + intercept, slope * x.max() + intercept], 'r', label='fitted line')

    plt.xlabel("Contrast scalar", fontsize=22)
    plt.ylabel("History scalar", fontsize=22)
    plt.title("pearsonr={:.3f}, p={:.3f}".format(pearsonr(x, y).statistic, pearsonr(x, y).pvalue))
    print(r_value, p_value, std_err)
    plt.tight_layout()
    plt.savefig("scatter_{}".format(file[:-2]))
    plt.show()


left_act_strong = np.array([0, 0, 1, 1, 0, 1])
right_act_strong = np.array([1, 0, 0, 0, 1, 1])

left_act_medium = np.array([0, 0, 1, 0.25, 0, 1])
right_act_medium = np.array([1, 0, 0, 0, 0.25, 1])

left_act_small = np.array([0, 0, 1, 0.125, 0, 1])
right_act_small = np.array([1, 0, 0, 0, 0.125, 1])
left_act_tiny = np.array([0, 0, 1, 0.0625, 0, 1])
right_act_tiny = np.array([1, 0, 0, 0, 0.0625, 1])

left_act_zero_rewarded = np.array([0, 0, 1, 0, 0, 1])
right_act_zero_rewarded = np.array([1, 0, 0, 0, 0, 1])
left_act_zero_unrewarded = np.array([0, 0, 1, 0, 0, -1])
right_act_zero_unrewarded = np.array([1, 0, 0, 0, 0, -1])

wrong_right_act_small = np.array([1, 0, 0, 0.125, 0, -1])
wrong_left_act_small = np.array([0, 0, 1, 0, 0.125, -1])
wrong_right_act_tiny = np.array([1, 0, 0, 0.0625, 0, -1])
wrong_left_act_tiny = np.array([0, 0, 1, 0, 0.0625, -1])
wrong_right_act_medium = np.array([1, 0, 0, 0.25, 0, -1])
wrong_left_act_medium = np.array([0, 0, 1, 0, 0.25, -1])
wrong_right_act_strong = np.array([1, 0, 0, 1, 0, -1])
wrong_left_act_strong = np.array([0, 0, 1, 0, 1, -1])


plot_side_diffs = False
baseline = 0
apply_baseline = False

def reward_vs_scalar(all_scalar, input, ranges):
    # iterate over the array of scalars, and count up the rewards vs non-rewards when the scalar is within the bins provided in ranges

    rewards = np.zeros(len(ranges) - 1)
    counter = np.zeros(len(ranges) - 1)
    all_points = [[] for _ in range(len(ranges) - 1)]

    for sess in range(input.shape[0]):
        for j in range(len(ranges) - 1):
            if 'lstm_dim' not in infos or infos['lstm_dim'] == 1:
                mask = np.logical_and(all_scalar[sess] >= ranges[j], all_scalar[sess] < ranges[j+1])
            else:
                mask = np.logical_and(all_scalar[sess][:, 0] >= ranges[j], all_scalar[sess][:, 0] < ranges[j+1])
            rewards[j] += np.sum(np.clip(input[sess, 1:all_scalar[sess].shape[0] + 1, 5][mask], 0, 1))
            counter[j] += np.sum(mask)
            all_points[j] += list(np.clip(input[sess, 1:all_scalar[sess].shape[0] + 1, 5][mask], 0, 1))

    print(counter)
    return rewards / counter, all_points

def dynamics(starting_point, augmenter, infos):
    trajectory = []
    for i in range(infos['history_slots']):
        processed = reload_net.mult_apply_augmented(starting_point, augmenter, i)
        trajectory.append(processed)
    return trajectory

def plot_start_and_history(input, augmenter, infos, marker='*', **kwargs):
    if infos['agent_class'] == "<class '__main__.Mirror_mecha_plus'>":
        starting_point = reload_net.input_process(input[:3].reshape(1, -1), np.append(input[3:], augmenter).reshape(1, -1))[0]
        print(input)
        print(starting_point)
    else:
        starting_point = reload_net.input_process(np.append(input, augmenter))
    if infos['history_slots'] != 'infinite':
        trajectory = np.array(dynamics(starting_point, augmenter, infos))
        trajectory[:, 2] = trajectory[:, 2] - baseline * apply_baseline #!!! baseline cannot just be applied to starting-position, this will change how decay is done...
    starting_point = np.array(starting_point)
    starting_point[2] = starting_point[2] - baseline * apply_baseline #!!! baseline cannot just be applied to starting-position, this will change how decay is done...
    if plot_side_diffs:
        plt.scatter(starting_point[2], starting_point[0], marker=marker, s=80, zorder=2, **kwargs)
        if infos['history_slots'] != 'infinite':
            x, y = trajectory[:, 0] - trajectory[:, 2], trajectory[:, 1] - trajectory[:, 2]
            plt.quiver(y[:-1], x[:-1], y[1:]-y[:-1], x[1:]-x[:-1], scale_units='xy', angles='xy', scale=1)
            return x
        return starting_point
    else:
        plt.scatter(starting_point[2], starting_point[0], marker=marker, s=160, zorder=2, **kwargs)
        if infos['history_slots'] != 'infinite':
            x, y = trajectory[:, 0], trajectory[:, 2]
            plt.quiver(y[:-1], x[:-1], y[1:]-y[:-1], x[1:]-x[:-1], scale_units='xy', angles='xy', scale=1)
            return x
        return starting_point

def create_history_gif(file):
    gif_stuff = []
    for i, augmenter in enumerate(history_motivation_position):
        print(i)
        fig = plt.figure(figsize=(16, 9))

        if infos['pass_to_history_limit'] == 4:
            plot_start_and_history(np.array([0, 0, 1, 1]), augmenter, infos, c='b', label="Left choice, rewarded")
            plot_start_and_history(np.array([0, 0, 1, -1]), augmenter, infos, c='b', marker='d', label="Left choice, unrewarded")
            plot_start_and_history(np.array([1, 0, 0, 1]), augmenter, infos, c='r', label="Right choice, rewarded")
            plot_start_and_history(np.array([1, 0, 0, -1]), augmenter, infos, c='r', marker='d', label="Right choice, unrewarded")

        else:
            plot_start_and_history(left_act_strong, augmenter, infos, c='b', label="Left choice, 100% contrast, rewarded")
            plot_start_and_history(right_act_strong, augmenter, infos, c='r', label="Right choice, 100% contrast, rewarded")
            plot_start_and_history(left_act_medium, augmenter, infos, c='b', alpha=0.75)
            plot_start_and_history(right_act_medium, augmenter, infos, c='r', alpha=0.75)

            plot_start_and_history(left_act_small, augmenter, infos, c='b', alpha=0.625)
            plot_start_and_history(right_act_small, augmenter, infos, c='r', alpha=0.625)
            plot_start_and_history(left_act_tiny, augmenter, infos, c='b', alpha=0.375)
            plot_start_and_history(right_act_tiny, augmenter, infos, c='r', alpha=0.375)

            plot_start_and_history(left_act_zero_rewarded, augmenter, infos, c='b', label="Left choice, 0% contrast, rewarded", alpha=0.25)
            plot_start_and_history(right_act_zero_rewarded, augmenter, infos, c='r', label="Right choice, 0% contrast, rewarded", alpha=0.25)
            plot_start_and_history(left_act_zero_unrewarded, augmenter, infos, c='b', marker='d', label="Left choice, 0% contrast, unrewarded", alpha=0.25, edgecolors='b', linestyle='--', lw=2)
            plot_start_and_history(right_act_zero_unrewarded, augmenter, infos, c='r', marker='d', label="Right choice, 0% contrast, unrewarded", alpha=0.25, edgecolors='r', linestyle='--', lw=2)

            plot_start_and_history(wrong_left_act_tiny, augmenter, infos, c='b', marker='d', alpha=0.375, edgecolors='b', linestyle='--', lw=2)
            plot_start_and_history(wrong_right_act_tiny, augmenter, infos, c='r', marker='d', alpha=0.375, edgecolors='r', linestyle='--', lw=2)
            plot_start_and_history(wrong_left_act_small, augmenter, infos, c='b', marker='d', alpha=0.625, edgecolors='b', linestyle='--', lw=2)
            plot_start_and_history(wrong_right_act_small, augmenter, infos, c='r', marker='d', alpha=0.625, edgecolors='r', linestyle='--', lw=2)
            plot_start_and_history(wrong_left_act_medium, augmenter, infos, c='b', marker='d', alpha=0.75, edgecolors='b', linestyle='--', lw=2)
            plot_start_and_history(wrong_right_act_medium, augmenter, infos, c='r', marker='d', alpha=0.75, edgecolors='r', linestyle='--', lw=2)
            plot_start_and_history(wrong_left_act_strong, augmenter, infos, c='b', marker='d', label="", edgecolors='b', linestyle='--', lw=2)
            plot_start_and_history(wrong_right_act_strong, augmenter, infos, c='r', marker='d', label="", edgecolors='r', linestyle='--', lw=2)


        plt.xlabel("Leftwards logit", fontsize=40)
        plt.ylabel("Rightwards logit", fontsize=40)

        plt.plot([0], [0], 'k', marker='o', ms=10)
        plt.gca().set_aspect('equal', adjustable='box')
        
        ax1 = plt.gca()
        ax2 = fig.add_axes([0.1, 0.7, 0.25, 0.25])
        ax2.plot(history_motivation_position, history_motivation_density, 'k')
        ax2.set_ylim(0)
        ax2.axvline(augmenter, color='k', alpha=0.4)

        ax1.set_xlim(-2.6, 2.6)
        ax1.set_ylim(-2.6, 2.6)

        ax1.plot([-1.6, 1.6], [-1.6, 1.6], 'k')

        sns.despine()
        plt.tight_layout()
        title_prefix = "sidediff_" if plot_side_diffs else ""
        plt.savefig("gif_images/" + str(file[:-2]).replace('.', '_') + "_sample_{}".format(i).replace('.', '_'))
        plt.close()

        gif_stuff.append("gif_images/" + str(file[:-2]).replace('.', '_') + "_sample_{}".format(i).replace('.', '_') + '.png')

    print('gif')
    images = []

    for f in gif_stuff:
        images.append(imageio.imread("./" + f))
    for f in gif_stuff[::-1]:
        images.append(imageio.imread("./" + f))

    print('saving')
    imageio.mimsave("encoding_{}.gif".format(file[:-2]), images, format='GIF', fps=30, loop=0)

# if this is the main file
if __name__ == "__main__":
    training_data = True
    if training_data:
        input_seq, train_mask, _, _ = load_data.gib_data_fast()
        mask_to_use = train_mask
        biased_blocks = np.load("./processed_data/train_bias.npy")
    else:
        _, _, input_seq, test_mask = load_data.gib_data_fast()
        mask_to_use = test_mask
        biased_blocks = np.load("./processed_data/test_bias.npy")

    _, _, test_input_seq, test_test_mask = load_data.gib_data_fast()
    file="./processed_data/all_mice.csv"
    train_eids, test_eids = json.load(open("train_eids", 'r')), json.load(open("test_eids", 'r'))  # map between eids and input array

    # file = 'mecha_plus_sweep_save_84306356.p'
    # infos = pickle.load(open("./222-20/" + file, 'rb'))
    # reload_net = loaded_network(infos)

    # file = 'mecha_plus_sweep_save_17480287.p'
    # infos = pickle.load(open("./LSTM_2D_all/" + file, 'rb'))
    # reload_net = loaded_network(infos)

    # file = 'mecha_plus_sweep_save_34254649.p'
    # infos = pickle.load(open("./LSTM2D-all-reduced_input_4/" + file, 'rb'))
    # reload_net = loaded_network(infos)

    # file = 'mecha_plus_sweep_save_25119790.p'
    # infos = pickle.load(open("./mecha_plus_sweep/" + file, 'rb'))
    # reload_net = loaded_network(infos)

    # new reduced input net, with fit history init
    # file = 'mecha_plus_sweep_save_48530959.p'
    # infos = pickle.load(open("./LSTM2D-all-reduced_input_4/" + file, 'rb'))
    # reload_net = loaded_network(infos)

    # reduced input with mirror net
    # file = 'mirror_mecha_plus_save_84564729.p'
    # infos = pickle.load(open("./222-2_reduc_mirror/" + file, 'rb'))
    # reload_net = loaded_network(infos)

    # full mirror net
    file = 'mirror_mecha_plus_save_18756433.p'
    infos = pickle.load(open("./222-2_mirror/" + file, 'rb'))
    reload_net = loaded_network(infos)

    print(100 * np.exp(-infos['best_test_nll']), 100 * np.exp(np.log(reload_net.return_predictions(test_input_seq, action_subsetting=True)[test_test_mask]).mean()))
    assert np.allclose(100 * np.exp(-infos['best_test_nll']), 100 * np.exp(np.log(reload_net.return_predictions(test_input_seq, action_subsetting=True)[test_test_mask]).mean()))

    # temp ###
    all_scalar = []
    all_history_comps = []

    results = reload_net.lstm_fn.apply(reload_net.params, None, input_seq[:, :, reload_net.input_list])
    hidden_states = results[1][1][0]  # extract the exact hidden states of the LSTM
    history_safe = results[1][0]

    if infos['agent_class'] == "<class '__main__.Mirror_mecha_plus'>":
        mot_matrix, mot_bias = infos['params_lstm']['mirror_mecha_plus/~_state_lstm/linear_8']['w'], infos['params_lstm']['mirror_mecha_plus/~_state_lstm/linear_8']['b']
    else:
        mot_matrix, mot_bias = infos['params_lstm']['mecha_history_plust_lstm/~_state_lstm/linear_8']['w'], infos['params_lstm']['mecha_history_plust_lstm/~_state_lstm/linear_8']['b']
    if 'lstm_dim' not in infos or infos['lstm_dim'] == 1:
        mot_matrix = mot_matrix[:, 0]  # need to get rid of the flat last dimension for this

    standard_hist = return_standard_history()

    contrast_motzis = []
    history_motzis = []

    for sess in range(15, min(55, len(train_eids))):

        plt.figure(figsize=(12, 7))
        if 'lstm_dim' not in infos or infos['lstm_dim'] == 1:
            motivations = (hidden_states[sess, :train_mask[sess].sum()+1] * mot_matrix).sum(1) + mot_bias
        else:
            motivations = (hidden_states[sess, :train_mask[sess].sum()+1] @ mot_matrix) + mot_bias  # this might work in both cases? at least if don't cut out the last flat dimension a couple lines earlier?
        # TODO: check that this works in fact

        all_scalar.append(motivations)

        if infos['history_slots'] == 'infinite':
            the_stuff = history_safe[sess, :train_mask[sess].sum()+1, :]
        else:
            the_stuff = history_safe[sess, :train_mask[sess].sum()+1, :].sum(1)
        all_history_comps.append(the_stuff[:, 2] - the_stuff[:, 0])

        if 'lstm_dim' not in infos or infos['lstm_dim'] == 1:
            plt.plot(all_scalar[-1], label="Motivation", c='k')
        else:
            plt.plot(all_scalar[-1][:, 0], label="Contrast mot.")
            plt.plot(all_scalar[-1][:, 1], label="History mot.")
            contrast_motzis.append(all_scalar[-1][:, 0])
            history_motzis.append(all_scalar[-1][:, 1])

        plt.plot(input_seq[sess, :train_mask[sess].sum()+1, 0] - input_seq[sess, :train_mask[sess].sum()+1, 2] + 1, 'ko', label="Action", alpha=0.5)

        diff = standard_hist[sess, :train_mask[sess].sum()+1, 2] - standard_hist[sess, :train_mask[sess].sum()+1, 0]

        # reward rate plotting
        avg_rewards = []
        for i in range(train_mask[sess].sum()):
            avg_rewards.append(np.mean(np.clip(input_seq[sess, max(1, i-15):i+1, 5], 0, 1)))
        # plt.plot(avg_rewards, 'k', ls='--', label="Windowed reward rate", alpha=0.5)

        # block plotting
        previous = 0.5
        start = 0
        for i in range(len(biased_blocks[sess])):
            if biased_blocks[sess][i] == 0.2 and previous != 0.2:
                if previous == 0.5:
                    start = i
                    previous = 0.2
                else:
                    plt.axvspan(start, i, color='r', alpha=0.2)
                    start = i
                    previous = 0.2
            elif biased_blocks[sess][i] == 0.8 and previous != 0.8:
                if previous == 0.5:
                    start = i
                    previous = 0.8
                else:
                    plt.axvspan(start, i, color='b', alpha=0.2)
                    start = i
                    previous = 0.8
            if biased_blocks[sess][i] == 0.:
                if previous == 0.2:
                    plt.axvspan(start, i, color='b', alpha=0.2)
                elif previous == 0.8:
                    plt.axvspan(start, i, color='r', alpha=0.2)
                break

        # plt.plot(diff - np.mean(diff), label="Standard hist.")
        # plt.plot(all_history_comps[-1] - np.mean(all_history_comps[-1]), label="History")
        plt.xlabel("Trial (example session)", fontsize=34)
        plt.ylabel("Scalar / reward rate", fontsize=34)
        plt.gca().tick_params(axis='both', which='major', labelsize=24)
        plt.ylim(-3, 2.2)
        plt.xlim(0, train_mask[sess].sum())
        sns.despine()
        plt.legend(frameon=False, fontsize=28)
        plt.tight_layout()
        plt.savefig("tempi_{}".format(sess)) # , dpi=300)
        plt.show()
    # quit()
    # end temp ###

    fake_obs = np.zeros((1, 46, 9))
    fake_obs[0, 1:16, 0] = 1
    fake_obs[0, 16:31, 2] = 1
    fake_obs[0, 31::2, 0] = 1
    fake_obs[0, 32::2, 2] = 1
    fake_obs[0, 0:16, 8] = 1
    fake_obs[0, 16:31, 7] = 1
    fake_obs[0, 31::2, 8] = 1
    fake_obs[0, 32::2, 7] = 1
    fake_obs[0, :, 5] = 1

    results = reload_net.lstm_fn.apply(reload_net.params, None, fake_obs[:, :, reload_net.input_list])
    history_safe = results[1][0]

    # np.save("temp_0_9", (history_safe[:, :, 2] - history_safe[:, :, 0])[0])
    # diff_0_9 = np.load("temp_0_9.npy")
    # diff_0_11 = np.load("temp_-0_11.npy")
    # plt.plot(diff_0_9, label='high scalar')
    # plt.plot(diff_0_11, label='mode')
    # plt.ylabel("History right - left logit", size=20)
    # plt.xlabel("Simulated trial", size=20)
    # plt.plot(np.arange(0, 15), -1.8 * np.ones(15), 'ko', label='actions')
    # plt.plot(np.arange(15, 30), 2.2 * np.ones(15), 'ko')
    # plt.plot(np.arange(30, 45, 2), -1.8 * np.ones(8), 'ko')
    # plt.plot(np.arange(31, 45, 2), 2.2 * np.ones(7), 'ko')
    # plt.legend(fontsize=13)
    # plt.show()

    all_scalars = get_all_scalars_fast(reload_net, input_seq, mask_to_use, infos)
    long_array = np.concatenate(all_scalars)

    avg_rewards = []
    std_rewards = []
    for sess in range(input_seq.shape[0]):
        mask = mask_to_use[sess]
        avg_rewards.append(np.mean(np.clip(input_seq[sess, :-1, 5][mask], 0, 1)))
        std_rewards.append(np.std(np.clip(input_seq[sess, :-1, 5][mask], 0, 1)))

    avg_bias = []
    for sess in range(input_seq.shape[0]):
        mask = mask_to_use[sess]
        rewarded = input_seq[sess, :-1, 5][mask] == 1
        unrewarded = ~ rewarded
        rewarded_left = np.logical_and(input_seq[sess, :-1, 0][mask] == 1, rewarded)
        rewarded_right = np.logical_and(input_seq[sess, :-1, 2][mask] == 1, rewarded)
        unrewarded_left = np.logical_and(input_seq[sess, :-1, 0][mask] == 1, unrewarded)
        unrewarded_right = np.logical_and(input_seq[sess, :-1, 2][mask] == 1, unrewarded)
        avg_bias.append(rewarded_left.sum() / (rewarded_left.sum() + unrewarded_left.sum()) - rewarded_right.sum() / (rewarded_right.sum() + unrewarded_right.sum()))


    # plot a heatmap of the density estimation of the scalar for each session as rows
    # points = np.linspace(long_array.min(), long_array.max(), 100)
    # session_trajectories = np.zeros((len(all_scalars), 100))
    # for i in range(len(all_scalars)):
    #     # interpolate all_scalars[i] onto the points from 0 to 100
    #     session_trajectories[i] = np.interp(np.linspace(0, all_scalars[i].shape[0], 100), np.arange(all_scalars[i].shape[0]), all_scalars[i])

    # # sort the sessions by the average reward rate, reversed
    # session_trajectories = session_trajectories[np.argsort(avg_rewards)[::-1]]

    # plt.imshow(session_trajectories, aspect='auto', cmap='viridis')

    # # turn the x-ticks into the positions from points
    # plt.xticks([10, 30, 50, 70, 90], np.round(points[[10, 30, 50, 70, 90]], 2))
    # plt.ylabel("Sessions", size=20)
    # plt.xlabel("Scalar", size=20)
    # # colorbar label
    # cbar = plt.colorbar()
    # cbar.set_label('Density', rotation=270, fontsize=20, labelpad=20)

    # plt.tight_layout()
    # plt.savefig("scalar_session_trajectories_{}".format(file[:-2]), dpi=300)
    # plt.show()

    if False: # not yet brought in line with 2D scalars
        # plot a heatmap of the density estimation of the scalar for each session as rows
        points = np.linspace(long_array.min(), long_array.max(), 200)
        session_marginals = np.zeros((len(all_scalars), 200))
        for i in range(len(all_scalars)):
            density = gaussian_kde(all_scalars[i])
            session_marginals[i] = density(points)

        # sort the sessions by the average reward rate, reversed
        sort_by = 'reward'
        if sort_by == 'reward':
            session_marginals = session_marginals[np.argsort(avg_rewards)[::-1]]
        elif sort_by == 'bias':
            session_marginals = session_marginals[np.argsort(avg_bias)[::-1]]

        # add a column above the plot, showing the overall scalar density
        density = gaussian_kde(long_array)
        overall_density = density(points)
        
        # plot on a separate axis
        import matplotlib.gridspec as gridspec
        fig = plt.figure(figsize=(10, 10))
        gs = gridspec.GridSpec(2, 2, width_ratios=[1, 0.1], height_ratios=[0.05, 1], wspace=0.05, hspace=0.05)

        ax0 = fig.add_subplot(gs[0, 0])
        im0 = ax0.imshow(overall_density.reshape(1, -1), cmap='viridis', aspect='auto')
        ax0.set_title("Marginal over sessions", size=30)
        
        ax1 = fig.add_subplot(gs[1, 0])
        im1 = ax1.imshow(session_marginals, aspect='auto', cmap='viridis')

        # plt.imshow(session_marginals, aspect='auto', cmap='viridis')

        # turn the x-ticks into the positions from points
        plt.xticks([20, 60, 100, 140, 180], np.round(points[[20, 60, 100, 140, 180]], 2))
        plt.ylabel("Sessions (reward rate sorted)", size=32)
        plt.xlabel("Scalar distribution", size=32)
        # colorbar label
        # cbar = plt.colorbar(im, ax=ax)
        # cbar.set_label('Density', rotation=270, fontsize=20, labelpad=20)
        cax = fig.add_subplot(gs[1, 1])
        cbar = fig.colorbar(im1, cax=cax, orientation='vertical')
        cbar.set_label('Density', rotation=270, fontsize=30, labelpad=24)
        ax0.set_xticks([])  # Removes x ticks
        ax0.set_yticks([])  # Removes y ticks
        ax1.tick_params(axis='both', which='major', labelsize=22)

        plt.tight_layout()
        if sort_by == 'reward':
            plt.savefig("scalar_session_densities_{}".format(file[:-2]), dpi=300)
        elif sort_by == 'bias':
            plt.savefig("scalar_session_densities_bias_{}".format(file[:-2]), dpi=300)
        plt.show()


        plt.scatter(avg_rewards, [np.mean(x) for x in all_scalars])
        plt.xlabel("Average reward rate", size=20)
        plt.ylabel("Mean scalar", size=20)

        plt.tight_layout()
        plt.savefig("rew_vs_scalar_mean_{}".format(file[:-2]), dpi=300)
        plt.show()

        # plt.scatter(std_rewards, [np.std(x) for x in all_scalars])
        # plt.show()

        plt.scatter(avg_rewards, [np.std(x) for x in all_scalars])
        plt.xlabel("Average reward rate", size=20)
        plt.ylabel("Standard deviation of scalar", size=20)

        plt.tight_layout()
        plt.savefig("rew_vs_scalar_std_{}".format(file[:-2]), dpi=300)
        plt.show()

    # compute the average reward rate of each session, and plot it against the mean scalar


    if 'lstm_dim' not in infos or infos['lstm_dim'] == 1:
        xy_contrast = long_array
        z_contrast = gaussian_kde(xy_contrast)
        xy_history = xy_contrast
        z_history = z_contrast
    else:
        xy_contrast = long_array[:, 0]
        z_contrast = gaussian_kde(xy_contrast)
        xy_history = long_array[:, 1]
        z_history = gaussian_kde(xy_history)


    contrast_motivation_position = np.linspace(xy_contrast.min(), xy_contrast.max(), num=200)
    contrast_motivation_density = z_contrast(np.linspace(xy_contrast.min(), xy_contrast.max(), num=200))

    history_motivation_position = np.linspace(xy_history.min(), xy_history.max(), num=200)
    history_motivation_density = z_history(np.linspace(xy_history.min(), xy_history.max(), num=200))

    np.save("long_array_{}".format(file[:-2]), long_array)
    np.save("lstm_contrast_scalar_positions_{}".format(file[:-2]), contrast_motivation_position)
    np.save("lstm_contrast_scalar_density_{}".format(file[:-2]), contrast_motivation_density)
    np.save("lstm_history_scalar_positions_{}".format(file[:-2]), history_motivation_position)
    np.save("lstm_history_scalar_density_{}".format(file[:-2]), history_motivation_density)


    contrast_motivation_position = np.load("lstm_contrast_scalar_positions_{}.npy".format(file[:-2]))
    contrast_motivation_density = np.load("lstm_contrast_scalar_density_{}.npy".format(file[:-2]))
    history_motivation_position = np.load("lstm_history_scalar_positions_{}.npy".format(file[:-2]))
    history_motivation_density = np.load("lstm_history_scalar_density_{}.npy".format(file[:-2]))


    if False: # also missing 2D scalar handling I think
        from scipy.stats import sem
        rew_rate, all_points = reward_vs_scalar(all_scalars, input_seq, np.linspace(contrast_motivation_position[0], contrast_motivation_position[-1], 50))

        plt.plot(contrast_motivation_position, contrast_motivation_density, label="Scalar")
        # plt.plot(np.linspace(contrast_motivation_position[0], contrast_motivation_position[-1], 49), rew_rate)
        plt.errorbar(np.linspace(contrast_motivation_position[0], contrast_motivation_position[-1], 49), [np.mean(x) for x in all_points], [sem(x) / 2 for x in all_points], label="Reward rate")

        plt.ylim(0)
        plt.legend(frameon=False, fontsize=18)
        plt.ylabel("Density / Reward rate", size=24)
        plt.xlabel("LSTM scalar", size=24)
        sns.despine()
        plt.tight_layout()
        plt.savefig("rew_vs_scalar_{}".format(file[:-2]), dpi=300)
        plt.close()


    # long_array = np.load("long_array_{}.npy".format(file[:-2]))

    if infos['agent_class'] == "<class '__main__.Mirror_mecha_plus'>":
        contrast_matrix_1, contrast_bias_1 = infos['params_lstm']['mirror_mecha_plus/~_contrast_mlp/linear']['w'], infos['params_lstm']['mirror_mecha_plus/~_contrast_mlp/linear']['b']
        contrast_matrix_2, contrast_bias_2 = infos['params_lstm']['mirror_mecha_plus/~_contrast_mlp/linear_1']['w'], infos['params_lstm']['mirror_mecha_plus/~_contrast_mlp/linear_1']['b']
    else:
        contrast_matrix_1, contrast_bias_1 = infos['params_lstm']['mecha_history_plust_lstm/~_contrast_mlp/linear']['w'], infos['params_lstm']['mecha_history_plust_lstm/~_contrast_mlp/linear']['b']
        contrast_matrix_2, contrast_bias_2 = infos['params_lstm']['mecha_history_plust_lstm/~_contrast_mlp/linear_1']['w'], infos['params_lstm']['mecha_history_plust_lstm/~_contrast_mlp/linear_1']['b']

    # scatter_scalars(long_array)

    # block_dist = get_block_switch_differences(biased_blocks)
    # block_dist = np.concatenate(block_dist)

    # scatter_scalars(long_array, color=block_dist)
    # quit()

    def cont_process(contrast):
        temp = (contrast[:, None] * contrast_matrix_1).sum(0) + contrast_bias_1
        new = np.tanh(temp)
        return (new[:, None] * contrast_matrix_2).sum(0) + contrast_bias_2

    if True:
        plot_pmf_panels(long_array[:, 0] if long_array.shape[1] == 2 else long_array, cont_process, z=None, title=file[:-2], motivation_position=contrast_motivation_position, motivation_density=contrast_motivation_density)
        create_pmf_gif(cont_process, contrast_motivation_position, contrast_motivation_density, file[:-2])

    create_history_gif(file)