from QAgent import QAgent
import numpy as np

# Initialize parameters
gamma = 0.75 # Discount factor
alpha = 0.9 # Learning rate
location_to_state = {
    'L1' : 0,
    'L2' : 1,
    'L3' : 2,
    'L4' : 3,
    'L5' : 4,
    'L6' : 5,
    'L7' : 6,
    'L8' : 7,
    'L9' : 8
}
actions = [0,1,2,3,4,5,6,7,8]
rewards = np.array([[0,1,0,0,0,0,0,0,0],
              [1,0,1,0,1,0,0,0,0],
              [0,1,0,0,0,1,0,0,0],
              [0,0,0,0,0,0,1,0,0],
              [0,1,0,0,0,0,0,1,0],
              [0,0,1,0,0,0,0,0,0],
              [0,0,0,1,0,0,0,1,0],
              [0,0,0,0,1,0,1,0,1],
              [0,0,0,0,0,0,0,1,0]])
state_to_location = dict((state,location) for location,state in location_to_state.items())
Q = np.array(np.zeros([9,9]))

qagent = QAgent(alpha, gamma, location_to_state, actions, rewards, state_to_location, Q)
qagent.training('L9', 'L1', 1000)