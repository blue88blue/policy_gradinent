import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow as tf
import numpy as np
import pandas as pd
import gym
import time
import os


class Agent_Policy:

    def __init__(self,
                 input_shape,
                 trained_weight_dir="None",
                 action_space=3,
                 reward_decay=0.97,
                 learning_rate=1e-3,
                 average_round_num=1,
                 baseline_decay=0.5
                 ):

        self.action_space = action_space
        self.input_shape = input_shape
        self.reward_decay = reward_decay
        self.learning_rate = learning_rate
        self.average_round_num = average_round_num
        self.baseline_decay = baseline_decay

        self.round_state, self.round_action, self.round_reward = [], [], []
        self.grads_average = []

        # 初始化模型
        self.network = self.build_model()
        if os.path.isfile(trained_weight_dir):  # 导入模型参数
            self.network.load_weights(trained_weight_dir)

    def build_model(self):
        # 构建网络
        input_state = keras.Input(shape=self.input_shape)
        h1 = layers.Dense(32, activation='relu', kernel_initializer=keras.initializers.GlorotUniform())(input_state)
        h2 = layers.Dense(64, activation='relu', kernel_initializer=keras.initializers.GlorotUniform())(h1)
        h3 = layers.Dense(128, activation='relu', kernel_initializer=keras.initializers.GlorotUniform())(h2)
        h4 = layers.Dropout(0.5)(h3)
        h5 = layers.Dense(self.action_space, kernel_initializer=keras.initializers.GlorotUniform())(h4)
        out = layers.Softmax()(h5)

        # 定义网络
        model = keras.Model(inputs=input_state, outputs=out)

        keras.utils.plot_model(model, './output/policy_netwrok.jpg', show_shapes=True)
        return model

    # 根据网络输出的动作概率采取动作
    def action(self, state, test=False):
        state = tf.reshape(state, (1, -1))
        p_actions = self.network(state)
        p_actions = np.squeeze(p_actions)  # 去掉多余的维度

        print('\r%s-------' % p_actions, end='')

        action = np.random.choice(self.action_space, 1, p=p_actions)
        if test:
           action = tf.argmax(p_actions)

        return int(action)

    def collect_round_data(self, s, a, r):

        self.round_state.append(s)
        self.round_action.append(a)
        self.round_reward.append(r)

    def discount_and_norm_rewards(self):
        # 将列表中存储的回合数据，转换成矩阵
        round_state = np.array(self.round_state)
        round_action = np.array(self.round_action)
        round_reward = np.array(self.round_reward)

        # 将reward标准化，有正有负。可以使得每个action的概率都可以减少
        # (reward - baseline) 作为概率的权重，决定梯度的大小。若不减去baseline，则所有动作的概率只会提升，未采样到的动作概率将会相对得减小
        discounted_round_rewards = np.zeros_like(round_reward)
        cumulative = 0
        for t in reversed(range(round_reward.shape[0])):
            cumulative = cumulative * self.reward_decay + round_reward[t]
            discounted_round_rewards[t] = cumulative

        discounted_round_rewards -= np.mean(discounted_round_rewards)*self.baseline_decay
        discounted_round_rewards /= np.std(discounted_round_rewards)
        return round_state, round_action, discounted_round_rewards

    # 梯度上升
    def update(self, round):

        round_state, round_action, discounted_round_rewards = self.discount_and_norm_rewards()

        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)

        with tf.GradientTape() as tape:

            p = self.network(round_state)  # 输入状态，返回各个动作的概率

            # p_a_s = tf.reduce_sum(-tf.math.log(p) * tf.one_hot(round_action, self.action_space), axis=1)  # 从p中取出action对应的概率，对该量做梯度上升
            # gole1 = tf.reduce_mean(p_a_s * round_reward)  # 交叉熵多乘了一个与reward相关的权重，决定该动作的概率要减小还是增大，以及步幅的大小
            # 标签为概率1。 若该动作不好，则weight<0，会减少该动作概率。若该动作好，weight>0，会往1的方向提高该动作的概

            loss = keras.losses.categorical_crossentropy(y_pred=p, y_true=tf.one_hot(round_action, self.action_space))
            gole = tf.reduce_sum(loss * discounted_round_rewards)

            grads = tape.gradient(gole, self.network.trainable_variables)  # 求一个路径的导数

            # 清空回合数据
            self.round_state, self.round_action, self.round_reward = [], [], []

        if round % self.average_round_num == 0 and self.grads_average == []:
            self.grads_average = grads
        else:
            for i in range(len(grads)):
                self.grads_average[i] += grads[i]/self.average_round_num   # 采样多次路径，取平均

        if round % self.average_round_num == self.average_round_num-1:
            optimizer.apply_gradients(zip(self.grads_average, self.network.trainable_variables))  # 最大化奖励
            self.grads_average = []
        return gole



def train(game_name, round_num = 10000):

    env = gym.make(game_name)  # 游戏环境
    ob_min = env.observation_space.low
    ob_mm = env.observation_space.high - env.observation_space.low
    agent = Agent_Policy(input_shape=env.observation_space.shape, action_space=env.action_space.n)
    train_gole = []
    train_reward =[]

    for round in range(round_num):
        accmulated_reward = []
        state = env.reset()
        #state = (state-ob_min)/ob_mm    # 归一化到0-1
        while True:
            time1 = time.time()
            # 刷新画面
            env.render()

            # agent采取动作
            action = agent.action(state)

            # 环境返回下一个状态，以及得分
            next_state, reward, done, info = env.step(action)
            accmulated_reward.append(reward)
            #next_state = (next_state - ob_min) / ob_mm  # 归一化到0-1

            # 存储回合数据
            agent.collect_round_data(state, action, reward)

            # 回合结束，更新参数
            if done:
                gole = agent.update(round)
                train_gole.append([round, float(gole)])
                time2 = time.time()
                if round%agent.average_round_num==0:
                    train_reward.append(sum(accmulated_reward))
                    print("Round:%s   time:%0.5f   gole:%.4f   reward:%s" % (round, time2-time1, gole, sum(accmulated_reward)))
                break

            # 更新状态
            state = next_state

        # 保存权重, 并进行测试-----------------------------------------------------
        if round%50 == 0:
            train_gole_df = pd.DataFrame(train_gole)
            train_gole_df.to_csv('./output/train_gole.csv')  # 保存csv
            train_reward_df = pd.DataFrame(train_reward)
            train_reward_df.to_csv('./output/train_reward.csv')  # 保存csv
            agent.network.save_weights('./weight/policy_net_weight1.h5')

            r = []
            state = env.reset()
            while True:
                env.render()
                action = agent.action(state, test=True)
                next_state, reward, done, info = env.step(action)
                #########################################################
                '''
                position, v = next_state
                if position > -0.5:
                    reward = 100 * np.abs(v) + 5 * (0.5 + position)
                else:
                    reward = 100 * np.abs(v)
                '''
                #########################################################
                r.append(reward)
                if done:
                    break
                state = next_state
            print('test reward ---------- %.2f' % sum(r))


# 测试
def test(game_name, trained_weight_dir="./weight/policy_net_weight.h5"):
    env = gym.make(game_name)  # 游戏环境

    atest = Agent_Policy(input_shape=env.observation_space.shape, action_space=env.action_space.n, trained_weight_dir = trained_weight_dir)  # 初始化智能体
    # 测试
    state = env.reset()
    r = []
    while True:  # 回合未结束
        env.render()
        # agent采取动作

        action = atest.action(state, test=True)
        # 环境返回下一个状态，以及得分
        next_state, reward, done, info = env.step(action)

        r.append(reward)

        state = next_state
        if done:
            print("reward:%s" %(sum(r)))
            break


if __name__ == '__main__':
    train('LunarLander-v2', 10000)
    #test('LunarLander-v2')
