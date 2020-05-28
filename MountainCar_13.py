import gym
import random
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
import numpy as np
from scipy.interpolate import interp1d
import keras
from gym import wrappers


input_shape = (2,1)
NUMBER_OF_ACTIONS=3
runs = 1000000
gamma = 0.8
alfa = 0.8
train_every=1
batch_size=32
epsilon=0.02
experiene_size = 30000
oservation_dimenstion =  2
number_of_action=3

class Experience:

    def __init__(self):
        self.experience_Mountain_car=[]
        self.size = experiene_size
        self.batch_size = batch_size

    def experience_model(self,state,action,reward,next_state,done):

        if len(self.experience_Mountain_car)>self.size:
            del self.experience_Mountain_car[0]
        self.experience_Mountain_car.append([state,action,reward,next_state,done])

    def get_sample(self):
        choose = random.sample(self.experience_Mountain_car, self.batch_size)
        return choose
        
        
class NN_model:
    
    def Neural_Network(self):
        
        self.model = Sequential()
        self.model.add(Dense(units=16, activation='relu', input_dim=oservation_dimenstion ))
        self.model.add(Dense(units=32, activation='relu'))
        self.model.add(Dense(units=64, activation='relu'))
        self.model.add(Dense(units=64, activation='relu'))

        self.model.add(Dense(units=number_of_action, activation='linear'))
        #adam = keras.optimizers.Adam(learning_rate=0.001)

        self.model.compile(loss='mean_squared_error',
                  optimizer="adam")
    def predict(self,x):
        return self.model.predict(x)

    def fiting(self,x_train,lables):
        
        self.model.fit(x_train, lables, epochs=1, batch_size=batch_size,verbose=0)

    def extract_and_predict(self,one_batch):

        x_train=[]
        y_train=[]
        for state,action,reward,next_state,done in one_batch:
            
            if not done:
                next_Q_value = np.max(self.predict(next_state))
                update = reward+gamma*next_Q_value

                target = self.predict(state)
                target[0][action] = update
                
            elif done:
                update= reward
                target = self.model.predict(state)
                target[0][action] = update
                
            x_train.append(state)
            y_train.append(target[0])

            #print("len x_train= ",len(x_train))


        x_train = np.array(x_train)
        y_train = np.array(y_train)

        x_train=np.reshape(x_train,(batch_size,2))
        y_train=np.reshape(y_train,(batch_size,3))

        return x_train,y_train
        

    
env = gym.make("MountainCar-v0")
env.reset()
q_model = NN_model()
q_model.Neural_Network()
memory =Experience()

postion = interp1d([-1.2,0.6],[0,1])
velocity = interp1d([-0.07,0.07],[0,1])

def scale_normalize_reshape(state):
    
    #state[0] = float(postion(state[0]))
    #state[1] = float(velocity(state[1]))
    state=np.reshape(np.array(state),(1,oservation_dimenstion))

    return state


def choose_action(all_q_values):
    
    if random.uniform(0, 1)>epsilon:
        next_action = np.argmax(all_q_values)
    else:
        next_action = random.choice([i for i in range(0,NUMBER_OF_ACTIONS)])
        
    return next_action
    

for run in range(0,runs):
    
    state = (env.reset())
    state = scale_normalize_reshape(state)
    reward = -1

    x=0
    while True:
        
        all_q_values = q_model.predict(state)
        next_action  = choose_action(all_q_values[0])
        next_state,new_reward,end_epsoide,_ = env.step(next_action)
        next_state = scale_normalize_reshape(next_state)

            
        memory.experience_model(state,next_action,reward,next_state,end_epsoide)
        
        if end_epsoide and x<198:

            print("reached goal at = ",run)

            break
        
        elif end_epsoide and x>=200:
            break

        x=x+1
        if run >= 1:
            one_batch = memory.get_sample()
            x_train,y_train = q_model.extract_and_predict(one_batch)
            q_model.fiting(x_train,y_train)

        reward=new_reward
        state=next_state
        
        #if run%100000000 == 0:
            #env.render()
            #print(run)

    if run%100 == 0:
        #env.render()
        print(run)
    print(run)



                             
env.close()    

