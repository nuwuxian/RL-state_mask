import glob
import numpy as np
import tensorflow as tf
import pickle
import copy
import gym
from PIL import Image
from abc import ABC, abstractmethod
from stable_baselines.common.policies import MlpPolicy


def rollout(victim_agent, adv_agent, mask_agent, env, num_traj, max_ep_len=1e3, save_path=None):

    # load agent-0 / agent-1 

    np.random.seed(100)
    reward_record = []
    win_count = 0
    valid_num = num_traj
    for i in range(num_traj):
        env.seed(i)
        if i != 0 and i % 10 == 0:
            print('Traj %d out of %d. Winning rate %f' %(i, num_traj, win_count * 1.0 /i))
        observation = env.reset()

        episode_length, epr, eploss, done = 0, 0, 0, False  # bookkeeping
        
        mask_pos = []
        value_seq = []
        vic_action_seq = []
        adv_action_seq = []
        mask_probs = []

        while not done and episode_length < max_ep_len:
            
            actions = []

            for id, obs in enumerate(observation):
                if id==0:
                   act = victim_agent.act(observation=obs)[0]
                   vic_action_seq.append(act)
                   clipped_actions = np.clip(act, env.action_space.spaces[0].low, env.action_space.spaces[0].high)
                else:
                    act, value = adv_agent.act(observation=obs)
                    value_seq.append(value['vpred'])
                    '''mask action'''
                    mask_act, neglogp = mask_agent.act(observation=obs)
                    mask_act = mask_act[0]
                    prob = np.exp(-neglogp)[0]
                    if mask_act == 1:
                        # act = act + np.random.rand(act.shape[0]) * 3 -1
                        mask_pos.append(episode_length)
                        mask_probs.append([prob, 1-prob])
                    else:
                        # if prob>0.9999:
                            # act = act + np.random.rand(act.shape[0]) * 3 -1
                            # change_num+=1
                        mask_probs.append([1-prob, prob])
                    '''mask action finish'''
                    adv_action_seq.append(act)
                    clipped_actions = np.clip(act, env.action_space.spaces[0].low, env.action_space.spaces[0].high)

                actions.append(clipped_actions)

            actions = tuple(actions)
            observation, rewards, dones, infos = env.step(actions)
            done = dones
            episode_length += 1
        
        if infos[1]['reward_remaining'] > infos[0]['reward_remaining']:
            win_count +=1
            reward_record.append(1)
        elif infos[0]['reward_remaining'] == infos[1]['reward_remaining']:
            reward_record.append(0)
            valid_num -=1
        else:
            reward_record.append(-1)


        mask_pos_filename = "./recording/mask_pos_" + str(i) + ".out" 
        np.savetxt(mask_pos_filename, mask_pos)


        eps_len_filename = "./recording/eps_len_" + str(i) + ".out" 
        np.savetxt(eps_len_filename, [episode_length])

        value_seq_filename = './recording/value_seq_' + str(i) + '.out'
        np.savetxt(value_seq_filename, value_seq)

        vic_act_seq_filename = "./recording/vic_act_seq_" + str(i) + ".out" 
        np.savetxt(vic_act_seq_filename, vic_action_seq)

        adv_act_seq_filename = "./recording/adv_act_seq_" + str(i) + ".out" 
        np.savetxt(adv_act_seq_filename, adv_action_seq)

        mask_probs_filename = "./recording/mask_probs_" + str(i) + ".out" 
        np.savetxt(mask_probs_filename, mask_probs)
        
    print("The winning rate is: ", win_count / valid_num)
    # print("average change:", change_num/valid_num)
    np.savetxt("./recording/reward_record.out", reward_record)


def load_agent(env_name, agent_type, agent_path):
    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1)
    sess = tf.Session(config=tf_config)
    sess.__enter__()

    env = gym.make(env_name)

    policy = []
    for i in range(2):
        if agent_type[i] == 'zoo':
            policy.append(MlpPolicyValue(scope="policy" + str(i), reuse=False,
                                         ob_space=env.observation_space.spaces[i],
                                         ac_space=env.action_space.spaces[i],
                                         hiddens=[64, 64], normalize=True))
        elif agent_type[i] == 'adv':
            policy.append(MlpPolicy(sess, env.observation_space.spaces[i], env.action_space.spaces[i],
                                    1, 1, 1, reuse=False))

    sess.run(tf.variables_initializer(tf.global_variables()))

    for i in range(2):
        if agent_type[i] == 'zoo':
            param_path = agent_path + '/agent' + str(i + 1) + '_parameters-v1.pkl'
            param = load_from_file(param_pkl_path=param_path)
            setFromFlat(policy[i].get_variables(), param)
        else:
            param = load_from_model(agent_path + '/model.pkl')
            adv_agent_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='model')
            setFromFlat(adv_agent_variables, param)
    return policy


def rl_fed(env, seed, model, obs_rms, agent_type, original_traj, max_ep_len, importance,
           exp_agent_id=1, render=False, mask_act=False):

    acts_orin = original_traj['actions']
    values_orin = original_traj['values']

    traj_len = np.count_nonzero(values_orin)

    start_step = max_ep_len - traj_len
    env.seed(seed)

    episode_length, epr, done = 0, 0, False  # bookkeeping
    observation = env.reset()

    for i in range(traj_len+200):
        actions = []
        for id, obs in enumerate(observation):
            if agent_type[id] == 'zoo':
                if id != exp_agent_id:
                    # fixed opponent agent
                    act, _ = model[id].act(stochastic=False, observation=obs)
                else:
                    # victim agent we need to explain
                    # action = acts_orin[start_step+i]
                    act, _ = model[id].act(stochastic=False, observation=obs)
                    if mask_act:
                        if start_step + i in importance:
                            # print(start_step + i)
                            # add noise into the action
                            # print(np.min(act))
                            # print(np.max(act))
                            act = act + np.random.rand(act.shape[0]) * 3 - 1
                            # print(np.min(act))
                            # print(np.max(act))
#                            act = np.clip(act, env.action_space.spaces[exp_agent_id].low,
#                                        env.action_space.spaces[exp_agent_id].high)
            else:
                obs = np.clip((obs - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8), -10, 10)
                act = model[id].step(obs=obs[None, :], deterministic=True)[0][0]
            
            actions.append(act)

        actions = tuple(actions)
        observation, _, done, infos = env.step(actions)
        reward = infos[exp_agent_id]['reward_remaining']
        episode_length += 1

        if render: env.render()
        if done: 
            assert reward != 0
            epr = reward
            break 
    print('step # {}, reward {:.0f}.'.format(episode_length, epr))
    return epr


def load_from_file(param_pkl_path):

    if param_pkl_path.endswith('.pkl'):
       with open(param_pkl_path, 'rb') as f:
            params = pickle.load(f)
    else:
        params = np.load(param_pkl_path)
    return params


def load_from_model(param_pkl_path):

    if param_pkl_path.endswith('.pkl'):
       with open(param_pkl_path, 'rb') as f:
            params = pickle.load(f)
       policy_param = params[1][0]
       flat_param = []
       for param in policy_param:
           flat_param.append(param.reshape(-1))
       flat_param = np.concatenate(flat_param, axis=0)
    else:
        flat_param = np.load(param_pkl_path, allow_pickle=True)
        if len(flat_param)==3:
            flat_param_1 = []
            for i in flat_param[0]:
                    flat_param_1.append(i)
            flat_param = []
            for param in flat_param_1:
                flat_param.append(param.reshape(-1))
            flat_param = np.concatenate(flat_param, axis=0)
    return flat_param


def setFromFlat(var_list, flat_params, sess=None):
    shapes = list(map(lambda x: x.get_shape().as_list(), var_list))
    total_size = np.sum([int(np.prod(shape)) for shape in shapes])
    if total_size != flat_params.shape[0]:
        redundant = flat_params.shape[0] - total_size
        flat_params = flat_params[redundant:]
        assert flat_params.shape[0] == total_size, \
            print('Number of variables does not match when loading pretrained victim agents.')
    theta = tf.placeholder(tf.float32, [total_size])
    start = 0
    assigns = []
    for (shape, v) in zip(shapes, var_list):
        size = int(np.prod(shape))
        assigns.append(tf.assign(v, tf.reshape(theta[start:start + size], shape)))
        start += size
    op = tf.group(*assigns)
    if sess == None:
        tf.get_default_session().run(op, {theta: flat_params})
    else:
        sess.run(op, {theta: flat_params})


class Policy(object):
    def reset(self, **kwargs):
        pass

    def act(self, observation):
        # should return act, info
        raise NotImplementedError()

    @property
    def value_flat(self):
        return self.vpred

    @property
    def obs_ph(self):
        return self.observation_ph

    @abstractmethod
    def step(self, obs, state=None, mask=None):
        """
        Returns the policy for a single step
        :param obs: ([float] or [int]) The current observation of the environment
        :param state: ([float]) The last states (used in recurrent policies)
        :param mask: ([float]) The last masks (used in recurrent policies)
        :return: ([float], [float], [float], [float]) actions, values, states, neglogp
        """
        raise NotImplementedError

    @abstractmethod
    def proba_step(self, obs, state=None, mask=None):
        """
        Returns the action probability for a single step
        :param obs: ([float] or [int]) The current observation of the environment
        :param state: ([float]) The last states (used in recurrent policies)
        :param mask: ([float]) The last masks (used in recurrent policies)
        :return: ([float]) the action probability
        """
        raise NotImplementedError


class RunningMeanStd(object):
    def __init__(self, scope="running", reuse=False, epsilon=1e-2, shape=()):
        with tf.variable_scope(scope, reuse=reuse):
            self._sum = tf.get_variable(
                dtype=tf.float32,
                shape=shape,
                initializer=tf.constant_initializer(0.0),
                name="sum", trainable=False)
            self._sumsq = tf.get_variable(
                dtype=tf.float32,
                shape=shape,
                initializer=tf.constant_initializer(epsilon),
                name="sumsq", trainable=False)
            self._count = tf.get_variable(
                dtype=tf.float32,
                shape=(),
                initializer=tf.constant_initializer(epsilon),
                name="count", trainable=False)
            self.shape = shape

            self.mean = tf.to_float(self._sum / self._count)
            var_est = tf.to_float(self._sumsq / self._count) - tf.square(self.mean)
            self.std = tf.sqrt(tf.maximum(var_est, 1e-2))


def dense(x, size, name, weight_init=None, bias=True):
    w = tf.get_variable(name + "/w", [x.get_shape()[1], size], initializer=weight_init)
    ret = tf.matmul(x, w)
    if bias:
        b = tf.get_variable(name + "/b", [size], initializer=tf.zeros_initializer())
        return ret + b
    else:
        return ret


def switch(condition, if_exp, else_exp):
    x_shape = copy.copy(if_exp.get_shape())
    x = tf.cond(tf.cast(condition, 'bool'),
                lambda: if_exp,
                lambda: else_exp)
    x.set_shape(x_shape)
    return x


class DiagonalGaussian(object):
    def __init__(self, mean, logstd):
        self.mean = mean
        self.logstd = logstd
        self.std = tf.exp(logstd)

    def sample(self):
        return self.mean + self.std * tf.random_normal(tf.shape(self.mean))

    def mode(self):
        return self.mean

    def neglogp(self, x):
        return 0.5 * tf.reduce_sum(tf.square((x - self.mean) / self.std), axis=-1) \
               + 0.5 * np.log(2.0 * np.pi) * tf.cast(tf.shape(x)[-1], tf.float32) \
               + tf.reduce_sum(self.logstd, axis=-1)

    def kl(self, other):
        return tf.reduce_sum(other.logstd - self.logstd + (tf.square(self.std) + tf.square(self.mean - other.mean)) /
                             (2.0 * tf.square(other.std)) - 0.5, axis=-1)

    def entropy(self):
        return tf.reduce_sum(self.logstd + .5 * np.log(2.0 * np.pi * np.e), axis=-1)


class MlpPolicyValue(Policy):
    def __init__(self, scope, *, ob_space, ac_space, hiddens, rate=0.0, convs=[], n_batch_train=1,
                 sess=None, reuse=False, normalize=False):
        self.sess = sess
        self.recurrent = False
        self.normalized = normalize
        self.zero_state = np.zeros(1)
        with tf.variable_scope(scope, reuse=reuse):
            self.scope = tf.get_variable_scope().name

            assert isinstance(ob_space, gym.spaces.Box)

            self.observation_ph = tf.placeholder(tf.float32, [None] + list(ob_space.shape), name="observation")
            self.stochastic_ph = tf.placeholder(tf.bool, (), name="stochastic")
            self.taken_action_ph = tf.placeholder(dtype=tf.float32, shape=[None, ac_space.shape[0]], name="taken_action")

            if self.normalized:
                if self.normalized != 'ob':
                    self.ret_rms = RunningMeanStd(scope="retfilter")
                self.ob_rms = RunningMeanStd(shape=ob_space.shape, scope="obsfilter")

            obz = self.observation_ph
            if self.normalized:
                obz = tf.clip_by_value((self.observation_ph - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)

            last_out = obz
            
            for i, hid_size in enumerate(hiddens):
                last_out = tf.nn.tanh(dense(last_out, hid_size, "vffc%i" % (i + 1)))

            self.vpredz = dense(last_out, 1, "vffinal")[:, 0]

            self.vpred = self.vpredz
            # reverse normalization. because the reward is normalized, reversing it to see the real value.

            if self.normalized and self.normalized != 'ob':
                self.vpred = self.vpredz * self.ret_rms.std + self.ret_rms.mean
            last_out = obz


            # ff activation policy
            ff_out = []
            for i, hid_size in enumerate(hiddens):
                last_out = tf.nn.tanh(dense(last_out, hid_size, "polfc%i" % (i + 1)))
                ff_out.append(last_out)
                last_out = tf.nn.dropout(last_out, rate=rate)

            self.policy_ff_acts = tf.concat(ff_out, axis=-1)

            mean = dense(last_out, ac_space.shape[0], "polfinal")
            logstd = tf.get_variable(name="logstd", shape=[n_batch_train, ac_space.shape[0]],
                                     initializer=tf.zeros_initializer())

            self.pd = DiagonalGaussian(mean, logstd)
            self.proba_distribution = self.pd
            self.sampled_action = switch(self.stochastic_ph, self.pd.sample(), self.pd.mode())
            self.neglogp = self.proba_distribution.neglogp(self.sampled_action)
            self.policy_proba = [self.proba_distribution.mean, self.proba_distribution.std]

    def make_feed_dict(self, observation, taken_action):
        return {
            self.observation_ph: observation,
            self.taken_action_ph: taken_action
        }

    def act(self, observation, stochastic=True, extra_op=None):
        outputs = [self.sampled_action, self.vpred]
        if extra_op == None:
            a, v = tf.get_default_session().run(outputs, {
                self.observation_ph: observation[None],
                self.stochastic_ph: stochastic})
            return a[0], v[0]
        else:
            outputs.append(self.policy_ff_acts)
            a, v, ex = tf.get_default_session().run(outputs, {
                self.observation_ph: observation[None],
                self.stochastic_ph: stochastic})
            return a[0], {'vpred': v[0]}, ex[0, ]

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    @property
    def initial_state(self):
        return None

    def step(self, obs, state=None, mask=None, deterministic=False):
        stochastic = not deterministic
        if self.sess==None:
            action, value, neglogp = tf.get_default_session().run([self.sampled_action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs, self.stochastic_ph: stochastic})
        else:
            action, value, neglogp = self.sess.run([self.sampled_action, self.value_flat, self.neglogp],
                                              {self.obs_ph: obs, self.stochastic_ph: stochastic})
        return action, value, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None):
        if self.sess==None:
            return tf.get_default_session().run(self.policy_proba, {self.obs_ph: obs})
        else:
            return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        if self.sess==None:
            return tf.get_default_session().run(self.value_flat, {self.obs_ph: obs})
        else:
            return self.sess.run(self.value_flat, {self.obs_ph: obs})