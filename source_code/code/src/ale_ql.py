from dqn2 import *

import cv2
from collections import OrderedDict, deque
from ale_python_interface import *
from datetime import datetime
import copy
import lasagne.nonlinearities

rng = np.random.RandomState(123456)

def save_model_parameters(outfile, model):
    model_params = model.get_model_params_to_save()
    np.savez(outfile, model_params=model_params)
    print "Saved model parameters to %s." % outfile

def load_model_parameters(path, model):
    npzfile = np.load(path)
    model_params = npzfile['model_params'].tolist()
    model.load_model_params_from_file(model_params)
    print "Loaded model parameters from %s." % (path)

def copy_params(model_from, model_to):
    model_params = model_from.get_model_params()
    model_to.load_model_params(model_params)

# Global options
ROM_PATH = '../../roms/breakout.bin'
ROM_NAME = ROM_PATH.split('.bin')[0].split('/')[1]
CROP = (18, 0) # (crop_up, crop_left)
CROPV = (110,84)
TEST = False

# ALE related inits
ale = ALEInterface()
ale.setInt('random_seed', rng.randint(1000))
repeat_action_probability = 0.
ale.setFloat('repeat_action_probability', repeat_action_probability)
ale.loadROM(ROM_PATH)
(raw_screen_width, raw_screen_height) = ale.getScreenDims()
total_lives = ale.lives()
action = ale.getMinimalActionSet()
#action = [1, 11, 12]
noop_act = 0
num_action = len(action)
action2ind = OrderedDict()
ind2action = OrderedDict()
for i in range(num_action):
    action2ind[action[i]] = i
    ind2action[i] = action[i]

# NN related options and inits
batch_size = 32
num_in_fmap = 4
(INPUT_HEIGHT, INPUT_WIDTH) = (84, 84)
ext_in_shape = (INPUT_HEIGHT, INPUT_WIDTH)
filter_type = [CONV_LAYER, CONV_LAYER, CONV_LAYER, FULL_CONN_LAYER, FULL_CONN_LAYER]
filter_shape = [(32, num_in_fmap, 8, 8), (64, 32, 4, 4), (64, 64, 3, 3), (1,512), (1, num_action)]
nonlinearities = [lasagne.nonlinearities.rectify, lasagne.nonlinearities.rectify, lasagne.nonlinearities.rectify,
                     lasagne.nonlinearities.rectify, None]
filter_stride = [(4, 4), (2, 2), (1, 1), (), ()]
alpha = 0.01
learning_rate = 0.00025
momentum = 0.95
epsilon = 0.01
alpha_rmsprop = 0.0
clip_err = 1.
alpha = np.asscalar(np.asarray(alpha).astype(np.float32))
learning_rate = np.asscalar(np.asarray(learning_rate).astype(np.float32))
momentum = np.asscalar(np.asarray(momentum).astype(np.float32))
epsilon = np.asscalar(np.asarray(epsilon).astype(np.float32))
alpha_rmsprop = np.asscalar(np.asarray(alpha_rmsprop).astype(np.float32))

model = DQN(batch_size, num_in_fmap, ext_in_shape, filter_type, filter_shape, filter_stride, nonlinearities, clip_err)
model_cap = DQN(batch_size, num_in_fmap, ext_in_shape, filter_type, filter_shape, filter_stride, nonlinearities, clip_err)
copy_params(model, model_cap)
print "model_cap parameters set to model parameters."

# Other Training and testing related options
max_epoch = 2000000
test_after = 10
save_state_after = 50
history_store_capacity = 500000
history_store_capacity_before_train = 25000
history_store = deque()
cur_hist = 0
frame_store_capacity = history_store_capacity
frame_store = [None]*frame_store_capacity
cur_frame = 0
epsilon_init = 1.
epsilon_final = 0.1
epsilon_test = 0.05
final_exploration_frame = 1000000
epsilon_decay = (epsilon_init-epsilon_final)/(final_exploration_frame)
epsilon_cur = epsilon_init
gamma = 0.99
C_steps = 10000
rng = np.random
debug_flag = False
num_random_actions = 0
num_greedy_actions = 0
to_show_frame_at_test = True
input_scale = 255.0
steps_per_epoch = 250000
max_null_ops = 30

load_path = ""
epsilon_cur_load = None
if load_path != "":
    load_model_parameters(load_path, model)
    epsilon_cur = epsilon_cur_load

X_prev = np.zeros((batch_size, num_in_fmap, ext_in_shape[0], ext_in_shape[1])).astype(np.float32)
X_forw = np.zeros((batch_size, num_in_fmap, ext_in_shape[0], ext_in_shape[1])).astype(np.float32)
Y = np.zeros((batch_size, num_action)).astype(np.float32)
filter_target = np.zeros((batch_size, num_action)).astype(np.float32)

def preprocess_screen(scrn, l_frame):
    if l_frame is not None:
        scrn_max = scrn > l_frame
        scrn = scrn * scrn_max + (1-scrn_max) * l_frame
    scrn2 = scrn[:,:,0]*0.299 + scrn[:,:,1]*0.587 + scrn[:,:,2] * 0.114
    #res = cv2.resize(scrn2, (CROPV[1], CROPV[0]), interpolation=cv2.INTER_AREA)
    res = cv2.resize(scrn2, (CROPV[1], CROPV[0]), interpolation=cv2.INTER_LINEAR)
    return res[CROP[0]:(CROP[0]+ext_in_shape[0]), CROP[1]:(CROP[1]+ext_in_shape[1])]/input_scale
    #res = cv2.resize(scrn2, (ext_in_shape[1], ext_in_shape[0]), interpolation=cv2.INTER_LINEAR)
    #return res/input_scale

def choose_action(to_test):
    global num_random_actions
    global num_greedy_actions
    global X_prev
    X_prev = X_prev * 0
    eps = max(epsilon_cur, epsilon_final)
    if to_test:
        eps = epsilon_test
    r = rng.uniform(0., 1.)
    if r < eps:
        # take random action
        to_act = rng.randint(0, num_action)
        num_random_actions += 1
        return to_act
    else:
        # take argmax_{action}_Q(last_frames)
        frame_ind = (cur_frame - num_in_fmap + frame_store_capacity) % frame_store_capacity
        for i in range(num_in_fmap):
            X_prev[0,i,:,:] = frame_store[frame_ind]
            frame_ind = (frame_ind + 1) % frame_store_capacity
        pred = model.prediction(X_prev.astype(np.float32))
        #print pred.shape
        num_greedy_actions += 1
        return np.argmax(pred[0,:])

def execute_action(to_act, num_times):
    score = 0
    for i in range(num_times):
        score += ale.act(to_act)
    return score

def prepare_input(batch):
    global X_forw
    global X_prev
    X_prev = X_prev * 0
    X_forw = X_forw * 0
    for i in range(batch_size):
        b = batch[i]
        frame_ind_prev = (history_store[b][0] - num_in_fmap + frame_store_capacity) % frame_store_capacity
        frame_ind_forw = (history_store[b][1] - num_in_fmap + frame_store_capacity) % frame_store_capacity
        for j in range(num_in_fmap):
            X_prev[i, j, :, :] = frame_store[frame_ind_prev]
            X_forw[i, j, :, :] = frame_store[frame_ind_forw]
            frame_ind_prev = (frame_ind_prev + 1) % frame_store_capacity
            frame_ind_forw = (frame_ind_forw + 1) % frame_store_capacity

def prepare_target(batch, X):
    global Y
    global filter_target
    Y = Y * 0
    filter_target = filter_target * 0
    pred = model_cap.prediction(X)
    if debug_flag:
        print np.sum(np.isnan(X))
        print np.sum(np.isnan(pred))
    if np.sum(np.isnan(X)) or np.sum(np.isnan(pred)):
        raw_input('NAN X or pred')
    for i in range(batch_size):
        b = batch[i]
        history = history_store[b]
        to_act = history[2]
        reward = history[3]
        game_over = history[4]
        Y[i, to_act] =  reward
        if not game_over:
            Y[i, to_act] += gamma * np.max(pred[i,:])
        filter_target[i, to_act] = 1

global_err_list = []
n_steps = 0
for epoch in range(1, max_epoch+1):
    last_color_frame = None
    to_test = False
    if epoch % test_after == 0 or TEST:
        to_test = True
    epoch_score = 0
    first_time = True
    ale.reset_game()
    total_lives = ale.lives()
    error_list = []
    while not ale.game_over():
        if first_time:
            #for i in range(num_in_fmap):
            for i in range(rng.randint(num_in_fmap, max_null_ops+1)):
                #to_act = rng.randint(0, num_action)
                to_act = noop_act
                num_random_actions += 1
                #epoch_score += execute_action(ind2action[to_act], num_in_fmap)
                epoch_score += execute_action(to_act, num_in_fmap)
                my_scrn = ale.getScreenRGB()
                frame_store[cur_frame] = preprocess_screen(my_scrn, last_color_frame)
                last_color_frame = my_scrn
                cur_frame = (cur_frame + 1) % frame_store_capacity
            first_time = False
        
        to_act = choose_action(to_test)
        last_score = execute_action(ind2action[to_act], num_in_fmap-1)
        last_color_frame = ale.getScreenRGB()
        last_score += ale.act(ind2action[to_act])
        epoch_score += last_score
        
        frame_store[cur_frame] = preprocess_screen(ale.getScreenRGB(), last_color_frame)
        cur_frame = (cur_frame + 1) % frame_store_capacity
        reward = (1. if last_score > 0 else (-1. if last_score < 0 else 0.))
        
        if debug_flag:
            print "ACTION", to_act, "REWARD", last_score
            cv2.imshow('game', frame_store[(cur_frame - 1 + frame_store_capacity) % frame_store_capacity])
            cv2.waitKey(10)
        if to_test and to_show_frame_at_test:
            cv2.imshow('game', cv2.resize(frame_store[(cur_frame - 1 + frame_store_capacity) % frame_store_capacity], (500,500)))
            cv2.waitKey(100)
        
        if not to_test:
            history = ((cur_frame-2+frame_store_capacity) % frame_store_capacity,
                                        (cur_frame-1+frame_store_capacity) % frame_store_capacity,
                                        to_act,
                                        reward,
                                        ale.game_over() or ale.lives()!=total_lives)
            if ale.lives()!=total_lives:
                total_lives = ale.lives()
            history_store.append(history)
            if len(history_store) >= history_store_capacity:
                history_store.popleft()
            
            # train
            if len(history_store) > history_store_capacity_before_train:
                batch = rng.randint(2*num_in_fmap, len(history_store), batch_size)
                cnt = 0
                while cnt < batch_size:
                    x = batch[cnt]
                    flag = False
                    for ll in range(x-num_in_fmap, x):
                        if history_store[ll][4]:
                            flag = True
                    if flag:
                        batch[cnt] = rng.randint(2*num_in_fmap, len(history_store))
                    else:
                        cnt += 1
                    
                prepare_input(batch)
                prepare_target(batch, X_forw)
                if np.sum(np.isnan(X_prev)) or np.sum(np.isnan(Y)) or np.sum(np.isnan(filter_target)):
                    continue
                err = model.rmsprop_step(X_prev, Y, filter_target, learning_rate, momentum, alpha_rmsprop, epsilon)
                error_list.append(err[0])
                global_err_list.append(err[0])
                n_steps += 1
                #print "CUR_ERR", err[0], "CUR_GRAD_SUM", err[1], err[2]
                epsilon_cur -= epsilon_decay
                if n_steps % C_steps == 0:
                    copy_params(model, model_cap)
                if n_steps % steps_per_epoch == 0:
                    print n_steps, steps_per_epoch
                    print "EPOCH_ORIG: ", n_steps/steps_per_epoch
    if len(error_list)==0:
        error_list.append(0)
    error_arr = np.asarray(error_list)
    global_err_arr = np.asarray(global_err_list)
    print "-"*80
    print "EPOCH", epoch
    print ("TEST" if to_test else "TRAIN"), "EP. SCORE", epoch_score
    print "# EP. FRAMES", ale.getEpisodeFrameNumber()
    print "EPSILON", (epsilon_final if to_test else max(epsilon_cur, epsilon_final))
    print "MAX ERROR:", np.max(error_arr)
    print "MEAN ERROR:", np.mean(error_arr)
    print "MEAN GLOBAL ERROR:", np.mean(global_err_arr)
    print "# Random actions: ", num_random_actions
    print "# Greedy actions: ", num_greedy_actions
    print "-"*80   
    num_random_actions = 0
    num_greedy_actions = 0 
    
    if epoch % save_state_after == 1:
        time_now=datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        save_model_parameters("../../model/ntm-%s-%s-model.npz" % ("QL", time_now), model)
        save_model_parameters("../../model/ntm-%s-%s-model-cap.npz" % ("QL", time_now), model)