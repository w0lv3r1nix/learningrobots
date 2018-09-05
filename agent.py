import numpy as np
from keras.models import Model
from keras.layers import Input, Conv2D, Dense, MaxPooling2D, Flatten, Multiply, Concatenate, LSTM, TimeDistributed, Add, Lambda
from keras.optimizers import Adam
import keras.backend as K
import tensorflow as tf
from memory import *

def huber_loss(y_true, y_pred):
    """ Wrapper function to use tensorflows huber-loss in keras. """
    return tf.losses.huber_loss(y_true,y_pred)

class DRQNAgent(object):

    ID = 'DRQNAgent'

    def __init__(self, input_dims, action_dims, sample_rate=4, memory_size=2048, gamma=.99, epsilon=[1,.001,.1]):
        """
        Initializes agent with input and output dimensions, discount rate and epsilon-greedy schedule.
        Sets up experience replay memory.
        Builds and compiles action-selection and target models.
        """
        self.input_dims = np.array(input_dims)
        self.sample_rate = sample_rate
        self.action_dims = action_dims
        self.epsilon, self.epsilon_decay, self.epsilon_min = epsilon
        self.gamma = gamma
        self.memory = ExperienceReplay(memory_size)

        self.path = [] # movement path for simulation
        self.model = self.build_model()
        self.model.compile(optimizer=Adam(lr=.001),loss=huber_loss)
        print(self.model.summary())

        self.target_model = self.build_model()

    def build_model(self):
        """ Returns the complete model of the agent. """
        model_in = Input(shape=self.input_dims)
        mask_in_1 = Input(shape=[self.action_dims[0]])
        mask_in_2 = Input(shape=[self.action_dims[1]])
        # convolutional part
        conv1 = Conv2D(16,(3,3),activation='elu',padding='valid')(model_in)
        pool1 = MaxPooling2D((3,3),strides=(2,2))(conv1)
        conv2 = Conv2D(16,(3,3),activation='elu',padding='valid')(pool1)
        pool1 = MaxPooling2D((3,3),strides=(2,2))(conv2)
        conv3 = Conv2D(32,(3,3),activation='elu',padding='valid')(pool1)
        pool2 = MaxPooling2D((3,3),strides=(2,2))(conv3)
        conv4 = Conv2D(4,(1,1),activation='elu',padding='valid')(pool2)
        flat = Flatten()(conv4)
        self.cnn = Model(inputs=(model_in),outputs=(flat))
        # recurrent part
        tdis_in = Input(shape=(self.sample_rate,self.input_dims[0],self.input_dims[1],self.input_dims[2]))
        tdis = TimeDistributed(self.cnn)(tdis_in)
        fc1 = LSTM(8,activation='tanh')(tdis)
        # dueling part
        # value
        val_stream = Dense(32,activation='elu')(fc1)
        val_out = Dense(1,activation='linear')(val_stream)
        # advantage
        adv_stream = Dense(32,activation='elu')(fc1)
        adv_out1 = Dense(self.action_dims[0],activation='linear',name='out1')(adv_stream)
        adv_out2 = Dense(self.action_dims[1],activation='linear',name='out2')(adv_stream)
        adv_out_norm1 = Lambda(lambda a: a - K.mean(a),output_shape=[self.action_dims[0]])(adv_out1)
        adv_out_norm2 = Lambda(lambda a: a - K.mean(a),output_shape=[self.action_dims[1]])(adv_out2)
        # Q(s,a) = V(s) + A(s,a)
        out1 = Add()([adv_out_norm1,val_out])
        out2 = Add()([adv_out_norm2,val_out])
        # mask by actions (all 1 for action-selection, one-hot for fitting)
        mask_out_1 = Multiply()([out1,mask_in_1])
        mask_out_2 = Multiply()([out2,mask_in_2])
        return Model(inputs=(tdis_in,mask_in_1,mask_in_2),outputs=(mask_out_1,mask_out_2))

    def choose_action(self, state, exploration=True):
        """
        Either selects random action based on epsilon-greedy or
        performs forward-pass through network and picks argmax as action.
        Returns both actions and action-values (q-values).
        """
        if np.random.uniform() < self.epsilon and exploration:
            action = np.array([np.random.randint(self.action_dims[0]),np.random.randint(self.action_dims[1])])
            action_values = 0
        else:
            state = np.expand_dims(state,0)
            action_values = np.squeeze(self.model.predict([state,np.ones([1,self.action_dims[0]]),np.ones([1,self.action_dims[0]])]))
            action_1 = np.argmax(action_values[0])
            action_2 = np.argmax(action_values[1])
            action = np.array([action_1,action_2])
        return action, action_values

    def replay(self,batch_size):
        """
        Samples batch from memory, extracts individual elements,
        calculates target values and trains network on batch.
        """
        batch = self.memory.get_batch(batch_size)
        states = np.array([x[0] for x in batch])
        actions = np.array([x[1] for x in batch])
        rewards = np.array([x[2] for x in batch])
        next_states = np.array([x[3] for x in batch])
        done = np.array([x[4] for x in batch])

        # perform forward pass on next states for targets
        next_Q_values = self.target_model.predict([next_states,np.ones((len(next_states),self.action_dims[0])),np.ones((len(next_states),self.action_dims[1]))])
        # at episode end, the target is just the reward -> set q-values to 0
        for i in range(len(rewards)):
            if done[i]:
                next_Q_values[0][i] = 0
                next_Q_values[1][i] = 0
        # calculate targets
        Q_values_1 = rewards + self.gamma * np.max(next_Q_values[0], axis=1)
        Q_values_2 = rewards + self.gamma * np.max(next_Q_values[1], axis=1)

        # set target arrays and one-hot masks to only train the action that was actually performed
        masks, targets = np.zeros_like(next_Q_values),np.zeros_like(next_Q_values)
        for i in range(len(rewards)):
            targets[0][i][actions[i][0]] = Q_values_1[i]
            targets[1][i][actions[i][1]] = Q_values_2[i]
            masks[0][i][actions[i][0]] = 1
            masks[1][i][actions[i][1]] = 1
        # fit model
        self.model.train_on_batch([states, masks[0], masks[1]], [targets[0],targets[1]])

    def update_target_model(self):
        """ Updates target model with new parameters by saving and reloading into target model. """
        self.model.save_weights('tmp_weights.h5')
        self.target_model.load_weights('tmp_weights.h5')

    def set_location(self, location):
        """ Sets location of agent by appending location to path """
        self.path.append(location)

    def get_location(self,step=-1):
        """ Returns location for a given simulation step """
        return self.path[step]

    def get_path(self):
        """ Returns complete movement path of agent """
        return self.path

    def get_field_of_view(self,step=-1):
        """ Returns corner points of agent field of view at simulation step. """
        corner_points = np.round(np.array([
            self.path[step][0]-self.input_dims[0]/2,
            self.path[step][0]+self.input_dims[0]/2,
            self.path[step][1]-self.input_dims[1]/2,
            self.path[step][1]+self.input_dims[1]/2,
        ])).astype(int)
        return corner_points

    def reset(self):
        """ Resets agent (clears path). """
        self.path = []

    def decay_epsilon(self):
        """ Decay epsilon if minimum value not reached. """
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay

class EffDRQNAgent(DRQNAgent):

    ID = 'EffDRQNAgent'

    def build_model(self):
        """ Returns the complete model of the agent. """
        model_in = Input(shape=self.input_dims)
        mask_in_1 = Input(shape=[self.action_dims[0]])
        mask_in_2 = Input(shape=[self.action_dims[1]])
        action_in = Input(shape=(2,))

        # convolutional part & action wrapped as model
        conv1 = Conv2D(16,(3,3),activation='elu',padding='valid')(model_in)
        pool1 = MaxPooling2D((3,3),strides=(2,2))(conv1)
        conv2 = Conv2D(16,(3,3),activation='elu',padding='valid')(pool1)
        pool1 = MaxPooling2D((3,3),strides=(2,2))(conv2)
        conv3 = Conv2D(32,(3,3),activation='elu',padding='valid')(pool1)
        pool2 = MaxPooling2D((3,3),strides=(2,2))(conv3)
        conv4 = Conv2D(4,(1,1),activation='elu',padding='valid')(pool2)
        flat = Flatten()(conv4)
        self.cnn = Model(inputs=(model_in),outputs=(flat))
        self.action_state = Model(inputs=action_in,outputs=action_in)
        # recurrent part
        tdis_in_1 = Input(shape=(self.sample_rate,self.input_dims[0],self.input_dims[1],self.input_dims[2]))
        tdis_in_2 = Input(shape=(self.sample_rate,2))
        tdis_1 = TimeDistributed(self.cnn)(tdis_in_1)
        tdis_2 = TimeDistributed(self.action_state)(tdis_in_2)
        # concatenate last actions to time-distributed output of convolutions
        tdis = Concatenate()([tdis_1,tdis_2])
        fc1 = LSTM(10,activation='tanh')(tdis)
        # dueling part
        # value
        val_stream = Dense(32,activation='elu')(fc1)
        val_out = Dense(1,activation='linear')(val_stream)
        # advantage
        adv_stream = Dense(32,activation='elu')(fc1)
        adv_out1 = Dense(self.action_dims[0],activation='linear',name='out1')(adv_stream)
        adv_out2 = Dense(self.action_dims[1],activation='linear',name='out2')(adv_stream)
        adv_out_norm1 = Lambda(lambda a: a - K.mean(a),output_shape=[self.action_dims[0]])(adv_out1)
        adv_out_norm2 = Lambda(lambda a: a - K.mean(a),output_shape=[self.action_dims[1]])(adv_out2)
        # Q(s,a) = V(s) + A(s,a)
        out1 = Add()([adv_out_norm1,val_out])
        out2 = Add()([adv_out_norm2,val_out])
        # mask by actions (all 1 for action-selection, one-hot for fitting)
        mask_out_1 = Multiply()([out1,mask_in_1])
        mask_out_2 = Multiply()([out2,mask_in_2])
        return Model(inputs=(tdis_in_1,tdis_in_2,mask_in_1,mask_in_2),outputs=(mask_out_1,mask_out_2))

    def choose_action(self, state_pair, exploration=True):
        """
        Either selects random action based on epsilon-greedy or
        performs forward-pass through network and picks argmax as action.
        Returns both actions and action-values (q-values).
        """
        if np.random.uniform() < self.epsilon and exploration:
            action = np.array([np.random.randint(self.action_dims[0]),np.random.randint(self.action_dims[1])])
            action_values = 0
        else:
            state = np.expand_dims(state_pair[0],0)
            action_state = np.expand_dims(state_pair[1],0)
            action_values = np.squeeze(self.model.predict([state,action_state,np.ones([1,self.action_dims[0]]),np.ones([1,self.action_dims[0]])]))
            action_1 = np.argmax(action_values[0])
            action_2 = np.argmax(action_values[1])
            action = np.array([action_1,action_2])
        return action, action_values

    def replay(self,batch_size):
        batch = self.memory.get_batch(batch_size)
        states = np.array([x[0][0] for x in batch])
        action_states = np.array([x[0][1] for x in batch])
        actions = np.array([x[1] for x in batch])
        rewards = np.array([x[2] for x in batch])
        next_states = np.array([x[3][0] for x in batch])
        next_action_states = np.array([x[3][1] for x in batch])
        done = np.array([x[4] for x in batch])

        # perform forward pass on next states for targets
        next_Q_values = self.target_model.predict([next_states,next_action_states,np.ones((len(next_states),self.action_dims[0])),np.ones((len(next_states),self.action_dims[1]))])
        # at episode end, the target is just the reward -> set q-values to 0
        for i in range(len(rewards)):
            if done[i]:
                next_Q_values[0][i] = 0
                next_Q_values[1][i] = 0
        # calculate targets
        Q_values_1 = rewards + self.gamma * np.max(next_Q_values[0], axis=1)
        Q_values_2 = rewards + self.gamma * np.max(next_Q_values[1], axis=1)

        # set target arrays and one-hot masks to only train the action that was actually performed
        masks, targets = np.zeros_like(next_Q_values),np.zeros_like(next_Q_values)
        for i in range(len(rewards)):
            targets[0][i][actions[i][0]] = Q_values_1[i]
            targets[1][i][actions[i][1]] = Q_values_2[i]
            masks[0][i][actions[i][0]] = 1
            masks[1][i][actions[i][1]] = 1
        # fit model
        self.model.train_on_batch([states, action_states, masks[0], masks[1]], [targets[0],targets[1]])
