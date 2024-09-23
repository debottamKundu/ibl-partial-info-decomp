"""
    Unified data loader and train and test split creation.
	TODO: write down data format and column meanings nicely
	TODO: first trial is cut away I think, but still, why is session number not present on first trial, and should we cut it out anyway?
"""
import pandas as pd
import numpy as np
import json

n_choices = 3
input_list = ['action_{}'.format(i) for i in range(n_choices)] + ['contrastLeft', 'contrastRight', 'feedbackType',
																  'session_number', 'prevContrastLeft', 'prevContrastRight', 'reward_average']
# this info is what we use as the basis of that feature
input_base_list = ['action_{}'.format(i) for i in range(n_choices)] + ['contrastLeft', 'contrastRight', 'feedbackType',
																  	   'session_number', 'contrastLeft', 'contrastRight', 'feedbackType']

# TODO: repeat for actually used data
# some info
# >>> np.unique(mice_data.choice, return_counts=1)
# (array([-1.,  0.,  1.]), array([203119,   2852, 206036]))
# in percent: array([0.49388505, 0.00750572, 0.49860923])
# >>> ((mice_data.feedbackType + 1) / 2).mean()
# 0.8003820323441106  # mice are correct in 80% of cases
# >>> a = mice_data.choice.values
# >>> np.mean(a[:-1] == a[1:])
# 0.6775411037703335  # mice repeat previous choice in 68% of cases
# >>> ((mice_data[np.logical_or(mice_data['contrastRight'] == 0, mice_data['contrastLeft'] == 0)].feedbackType + 1) / 2).mean()
# 0.5798191515959663
# >>> ((mice_data[mice_data['contrastRight'] > 0].feedbackType + 1) / 2).mean()
# 0.8309518179463793
# >>> ((mice_data[mice_data['contrastLeft'] > 0].feedbackType + 1) / 2).mean()
# 0.8280461176117129

# mice_data.choice: 1 is Leftwards, -1 is rightwards! -> action_0 is rightwards, action_2 is leftwards (action_1 is timeout)
# mice_data.feedbackType: 1 is correct, -1 is false

def smooth_array(array, window_width):
    """
    Smooth an array of 0s and 1s with a rectangular filter of a given width. Causally now!
    Thanks CGPT.
    
    :param array: List or numpy array of 0s and 1s
    :param window_width: Integer, the width of the rectangular filter
    :return: Numpy array of the smoothed values
    """
    if window_width < 1:
        raise ValueError("Window width must be at least 1")
    
    # Convert input to numpy array for easier manipulation
    array = np.array(array)
    
    # Create an array to hold the smoothed values
    smoothed_array = np.zeros_like(array, dtype=float)
    
    # Apply the rectangular filter (moving average)
    for i in range(len(array)):
        # Determine the start and end indices of the window
        start_index = max(0, i - window_width)
        
        # Compute the average within the window
        window_average = np.mean(array[start_index:i+1])
        
        # Store the smoothed value
        smoothed_array[i] = window_average
    
    return smoothed_array

def gib_data_fast():
	# load the default dataset directly from numpy saved arrays
	input_seq = np.load("./processed_data/input_seq.npy")
	train_mask = np.load("./processed_data/train_mask.npy")
	input_seq_test = np.load("./processed_data/input_seq_test.npy")
	test_mask = np.load("./processed_data/test_mask.npy")
	return input_seq, train_mask, input_seq_test, test_mask

def gib_data(file="./processed_data/all_mice.csv", alternate_split=False):
	"""Return train and test sequences (and the corresponding masks) from the specified file"""
	if type(file) == tuple:
		mice_data = pd.read_csv(file[0], low_memory=False)
	else:
		mice_data = pd.read_csv(file, low_memory=False)

	# some light processing
	mice_data[['contrastLeft', 'contrastRight']] = mice_data[['contrastLeft', 'contrastRight']].fillna(0)
	mice_data.choice = (mice_data.choice + 1).astype(int)  # so we can use them as indices later and avoid using jax.nn.one_hot

	_, trial_counts = np.unique(mice_data.session_start_time, return_counts=1)
	max_trials = trial_counts.max()

	action_cols = ['action_{}'.format(i) for i in range(n_choices)]  # this creates the following vector: 'action_0', 'action_1', 'action_2']
	action_seq = mice_data.choice
	targets = action_seq.values.reshape(-1)  # might not be needed?
	mice_data[action_cols]  = np.eye(n_choices)[targets]  # note how the dataset now contains 3 more columns, "action_0", "action_1", "action_2"

	# train and test split
	if file == "./processed_data/all_mice_plus_trained.csv":
		assert len(np.unique(mice_data.session)) == 952, "# of sessions has changed"
		train_eids_1, test_eids_1 = json.load(open("train_eids", 'r')), json.load(open("test_eids", 'r'))
		train_eids_2, test_eids_2 = json.load(open("train_eids_trained", 'r')), json.load(open("test_eids_trained", 'r'))  # extended dataset
		train_eids = train_eids_1 + train_eids_2
		test_eids = test_eids_1 + test_eids_2
		assert len(train_eids) == 403 + 177 and len(test_eids) == 68 + 32, "Numbers got messed up"
	else:
		assert len(np.unique(mice_data.session)) == 693, "# of sessions has changed"
		if not alternate_split:
			train_eids, test_eids = json.load(open("train_eids", 'r')), json.load(open("test_eids", 'r'))
		else:
			train_eids, test_eids = json.load(open("train_eids_alt", 'r')), json.load(open("test_eids_alt", 'r'))
		assert len(train_eids) == 403 and len(test_eids) == 68, "Numbers got messed up"

	if type(file) == tuple:
		if type(file[1] == tuple):
			train_eids, test_eids = file[1][0], [file[1][1]] if type(file[1][1]) != list else file[1][1]
		else:
			train_eids, test_eids = [file[1]], [file[1]]  # overwrite the eids

	input_seq, train_mask, train_bias = create_input_array(train_eids, mice_data, max_trials)
	input_seq_test, test_mask, test_bias = create_input_array(test_eids, mice_data, max_trials)
	return input_seq, train_mask, input_seq_test, test_mask, train_bias, test_bias

def create_input_array(eids, data, max_trials, apply_tanh=False):
	"""Based on eids, the session identifiers, we select data and agglomorate them in a big array (we pad to size max_trials), then return the data and necessary mask"""
	input_seq = []
	input_seq_lens = []
	bias_seq = []
	for eid in eids:
		temp_data = data[data.session == eid]
		temp_data_input = temp_data[input_base_list]
		input_seq_lens.append(temp_data_input.values.shape[0])
		# add 1 to max_trials, to make room for answers and rewards which get shifted one trial backwards
		input_seq.append(np.pad(temp_data_input.values, [(0, max_trials + 1 - temp_data_input.values.shape[0]), (0, 0)], 'constant'))

		# also collect information on bias blocks
		bias_data = temp_data['probabilityLeft']
		bias_seq.append(np.pad(bias_data.values, [(0, max_trials - bias_data.values.shape[0])], 'constant'))

	# the :-1 here are fine, everything got padded with one extra 0 (max_trials + 1)
	input_seq = np.array(input_seq)
	input_seq[:, 1:, :n_choices] = input_seq[:, :-1, :n_choices]  # move action to one timepoint later
	input_seq[:, 0, :n_choices] = 0
	input_seq[:, 1:, 5] = input_seq[:, :-1, 5]  # move reward to one timepoint later. Don't code extensible lists with -!
	input_seq[:, 0, 5] = 0
	input_seq[:, 1:, 7] = input_seq[:, :-1, 7]  # move left contrast to one timepoint later.
	input_seq[:, 0, 7] = 0
	input_seq[:, 1:, 8] = input_seq[:, :-1, 8]  # move right contrast to one timepoint later. Don't code extensible lists with -!
	input_seq[:, 0, 8] = 0
	for i in range(input_seq.shape[0]):
		input_seq[i, :, 9] = smooth_array(input_seq[i, :, 5], window_width=102)  # smooth over reward to use as LSTM scalar replacement, optimal setup as determind by reward2mot_net.py

	input_seq_lens = np.array(input_seq_lens)
	mask = (np.arange(max_trials) < input_seq_lens[:, None])
	if apply_tanh:
		input_seq[:, :, n_choices:n_choices + 2] = np.tanh(5 * input_seq[:, :, n_choices:n_choices + 2]) / np.tanh(5)

	return input_seq, mask, np.array(bias_seq)

if __name__ == "__main__":
	input_seq, train_mask, input_seq_test, test_mask, train_bias, test_bias = gib_data(file="./processed_data/all_mice.csv")
	assert np.array_equal(input_seq, np.load("./processed_data/input_seq.npy"))
	assert np.array_equal(train_mask, np.load("./processed_data/train_mask.npy"))
	assert np.array_equal(input_seq_test, np.load("./processed_data/input_seq_test.npy"))
	assert np.array_equal(test_mask, np.load("./processed_data/test_mask.npy"))
	assert np.array_equal(train_bias, np.load("./processed_data/train_bias.npy"))
	assert np.array_equal(test_bias, np.load("./processed_data/test_bias.npy"))
	print("Data is the same")
	if False:
		print("Really re-save data?")
		confirm = input()
		if confirm != 'y':
			print('Quitting')
			quit()
		print("Re-recording...")
		np.save("./processed_data/input_seq.npy", input_seq)
		np.save("./processed_data/train_mask.npy", train_mask)
		np.save("./processed_data/input_seq_test.npy", input_seq_test)
		np.save("./processed_data/test_mask.npy", test_mask)
		np.save("./processed_data/train_bias.npy", train_bias)
		np.save("./processed_data/test_bias.npy", test_bias)