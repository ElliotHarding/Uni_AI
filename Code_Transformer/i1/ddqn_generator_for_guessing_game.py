from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from collections import deque
import numpy as np
import random
import argparse
import gym
from gym import wrappers, logger

breeds = []
sizes = []
num_breeds = None

#######################################################
# ReadFiles
#   
#   Reads breed-and-information file into global arrays
#   breeds & sizes
#######################################################
try:            

    with open("drive/My Drive/breeds.txt") as f:
        breeds = f.read().splitlines()

    with open("drive/My Drive/sizes.txt") as f:
        sizes = f.read().splitlines()

    num_breeds = len(sizes)
        
except (IOError) as e:
        print("Error occured opening one of the files! " + e.strerror)
        sys.exit()
        

#######################################################
# DQNAgent
#######################################################
class DQNAgent:
    def __init__(self,
                 state_space, 
                 action_space, 
                 episodes=500):
        """DQN Agent on CartPole-v0 environment

        Arguments:
            state_space (tensor): state space
            action_space (tensor): action space
            episodes (int): number of episodes to train
        """
        self.action_space = action_space

        # experience buffer
        self.memory = []

        # discount rate
        self.gamma = 0.9

        # initially 90% exploration, 10% exploitation
        self.epsilon = 1.0
        # iteratively applying decay til 
        # 10% exploration/90% exploitation
        self.epsilon_min = 0.1
        self.epsilon_decay = self.epsilon_min / self.epsilon
        self.epsilon_decay = self.epsilon_decay ** \
                             (1. / float(episodes))

        # Q Network weights filename
        self.weights_file = 'drive/My Drive/ddqn_gameGuesser.h5'
        # Q Network for training
        n_inputs = state_space.shape[0]
        n_outputs = action_space.n
        self.q_model = self.build_model(n_inputs, n_outputs)
        self.q_model.compile(loss='mse', optimizer=Adam())
        # target Q Network
        self.target_q_model = self.build_model(n_inputs, n_outputs)
        # copy Q Network params to target Q Network
        self.update_weights()

        self.replay_counter = 0

    def build_model(self, n_inputs, n_outputs):
        """Q Network is 256-256-256 MLP

        Arguments:
            n_inputs (int): input dim
            n_outputs (int): output dim

        Return:
            q_model (Model): DQN
        """
        inputs = Input(shape=(n_inputs, ), name='state')
        x = Dense(256, activation='relu')(inputs)
        x = Dense(256, activation='relu')(x)
        x = Dense(256, activation='relu')(x)
        x = Dense(n_outputs,
                  activation='linear', 
                  name='action')(x)
        q_model = Model(inputs, x)
        q_model.summary()
        return q_model

    def save_weights(self):
        """save Q Network params to a file"""
        self.q_model.save_weights(self.weights_file)

    def update_weights(self):
        """copy trained Q Network params to target Q Network"""
        self.target_q_model.set_weights(self.q_model.get_weights())

    def act(self, state):
        """eps-greedy policy
        Return:
            action (tensor): action to execute
        """
        if np.random.rand() < self.epsilon:
            # explore - do random action
            return self.action_space.sample()

        # exploit
        q_values = self.q_model.predict(state)
        # select the action with max Q-value
        action = np.argmax(q_values[0])
        return action

    def remember(self, state, action, reward, next_state, done):
        """store experiences in the replay buffer
        Arguments:
            state (tensor): env state
            action (tensor): agent action
            reward (float): reward received after executing
                action on state
            next_state (tensor): next state
        """
        item = (state, action, reward)
        self.memory.append(item)

    def get_target_q_value(self, next_state, reward):
        """compute Q_max
           Use of target Q Network solves the 
            non-stationarity problem
        Arguments:
            reward (float): reward received after executing
                action on state
            next_state (tensor): next state
        Return:
            q_value (float): max Q-value computed
        """
        # max Q value among next state's actions
        # DQN chooses the max Q value among next actions
        # selection and evaluation of action is 
        # on the target Q Network
        # Q_max = max_a' Q_target(s', a')
        q_value = np.amax(\
                     self.target_q_model.predict(next_state)[0])

        # Q_max = reward + gamma * Q_max
        q_value *= self.gamma
        q_value += reward
        return q_value

    def replay(self, batch_size):
        """experience replay addresses the correlation issue 
            between samples
        Arguments:
            batch_size (int): replay buffer batch 
                sample size
        """
        # sar = state, action, reward
        sar_batch = random.sample(self.memory, batch_size)
        state_batch, q_values_batch = [], []

        # fixme: for speedup, this could be done on the tensor level
        # but easier to understand using a loop
        for state, action, reward in sar_batch:
            # policy prediction for a given state
            q_values = self.q_model.predict(state)
            
            #Removing use of next_state since dosent depend on it
            # get Q_max
            #q_value = self.get_target_q_value(next_state, reward)
            # correction on the Q value for the action used
            #q_values[0][action] = reward if done else q_value

            q_values[0][action] = reward if done else 0

            # collect batch state-q_value mapping
            state_batch.append(state[0])
            q_values_batch.append(q_values[0])

        # train the Q-network
        self.q_model.fit(np.array(state_batch),
                         np.array(q_values_batch),
                         batch_size=batch_size,
                         epochs=1,
                         verbose=0)

        # update exploration-exploitation probability
        self.update_epsilon()

        # copy new params on old target after 
        # every 10 training updates
        if self.replay_counter % 10 == 0:
            self.update_weights()

        self.replay_counter += 1

    def update_epsilon(self):
        """decrease the exploration, increase exploitation"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
#######################################################
# DDQNAgent
#######################################################
class DDQNAgent(DQNAgent):
    def __init__(self,
                 state_space, 
                 action_space, 
                 episodes=500):
        super().__init__(state_space, 
                         action_space, 
                         episodes)
        """DDQN Agent on CartPole-v0 environment

        Arguments:
            state_space (tensor): state space
            action_space (tensor): action space
            episodes (int): number of episodes to train
        """

        # Q Network weights filename
        #self.weights_file = 'ddqn_cartpole.h5'
        print("-------------DDQN------------")

    def get_target_q_value(self, next_state, reward):
        """compute Q_max
           Use of target Q Network solves the 
            non-stationarity problem
        Arguments:
            reward (float): reward received after executing
                action on state
            next_state (tensor): next state
        Returns:
            q_value (float): max Q-value computed
        """
        # max Q value among next state's actions
        # DDQN
        # current Q Network selects the action
        # a'_max = argmax_a' Q(s', a')
        action = np.argmax(self.q_model.predict(next_state)[0])
        # target Q Network evaluates the action
        # Q_max = Q_target(s', a'_max)
        q_value = self.target_q_model.predict(\
                                      next_state)[0][action]

        # Q_max = reward + gamma * Q_max
        q_value *= self.gamma
        q_value += reward
        return q_value

#######################################################
# Main
#######################################################
if __name__ == '__main__':

    agent = DDQNAgent(np.array([1]), gym.spaces.Discrete(3))

    # should be solved in this number of episodes
    episode_count = 40
    batch_count = 400

    # Q-Learning sampling and fitting
    dog_index = random.randrange(0, num_breeds)
    for episode in range(episode_count):
        for batch_run in range(batch_count):

            #guess=0 is S guess=1 is M guess=2 is L
            state = np.reshape(np.array([dog_index]), [1, 1])            
            guess = agent.act(state)        

            reward = 0
            size = sizes[dog_index]
            if size == "S" and guess == 0:
                reward = 1
            elif size == "M" and guess == 1:
                reward = 1
            elif size == "L" and guess == 2:
                reward = 1

            next_dog_index = random.randrange(0, num_breeds)
            next_state = np.reshape(np.array([next_dog_index]), [1, 1])     

            agent.remember(state, action, reward, next_state, done)
            dog_index = next_dog_index

        print("Running episode " + str(episode))
        agent.replay(batch_count)

    agent.save_weights()  
      
    print(agent.act(np.reshape(np.array([0]), [1, 1])))
    print(agent.act(np.reshape(np.array([0]), [1, 1])))
    print(agent.act(np.reshape(np.array([1]), [1, 1])))
    print(agent.act(np.reshape(np.array([1]), [1, 1])))
    print(agent.act(np.reshape(np.array([2]), [1, 1])))
    print(agent.act(np.reshape(np.array([2]), [1, 1])))
    print(agent.act(np.reshape(np.array([3]), [1, 1])))
    print(agent.act(np.reshape(np.array([3]), [1, 1])))
    print(agent.act(np.reshape(np.array([4]), [1, 1])))
    print(agent.act(np.reshape(np.array([4]), [1, 1])))
    print(agent.act(np.reshape(np.array([5]), [1, 1])))
    print(agent.act(np.reshape(np.array([5]), [1, 1])))
    print(agent.act(np.reshape(np.array([6]), [1, 1])))
    print(agent.act(np.reshape(np.array([6]), [1, 1])))
        
