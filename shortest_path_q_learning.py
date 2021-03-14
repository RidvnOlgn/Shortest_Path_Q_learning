import numpy as np

#define the shape of the environment
env_rows = 11
env_collums = 11

#create a 3D numpy array to hold current Q values
q_values = np.zeros((env_rows,env_collums,4))

#define action
#numeric action codes: 0 = up , 1 = right , 2 = down , 3 = left
actions = ['up','right','down','left']

#Create 2D numpy array to hold rewards for each state
rewards = np.full((env_rows,env_collums),-100)
rewards[0,5] = 100 # set reward packaging area

#define aisle location
aisle = {}
aisle[1] = [i for i in range(1,10)]
aisle[2] = [1,7,9]
aisle[3] = [i for i in range(1,8)]
aisle[3].append(9)
aisle[4] = [3,7]
aisle[5] = [i for i in range(11)]
aisle[6] = [5]
aisle[7] = [i for i in range(1,10)]
aisle[8] = [4,7]
aisle[9] = [i for i in range(1,10)]

#set rewards for all aisle location
for row_index in range(1,10):
    for collumn_index in aisle[row_index]:
        rewards[row_index,collumn_index] = -1

#define a function that determines if the specified location is a terminal state
def is_terminal_state(current_row_index,current_collumn_index):
    if rewards[current_row_index,current_collumn_index] == -1:
        return False
    else:
        return True

#define a function that will choose a random, non-terminal startting location
def get_starting_location():
    #get a random row and collumn index
    current_row_index = np.random.randint(env_rows)
    current_collumn_index = np.random.randint(env_collums)
    while is_terminal_state(current_row_index,current_collumn_index):
        current_row_index = np.random.randint(env_rows)
        current_collumn_index = np.random.randint(env_collums)
    return current_collumn_index,current_collumn_index

#define an epsilon greedy algorithm that will chosee which action to take next
def get_next_action(current_row_index,current_collumn_index,epsilon):
    #if randomly chosen value between  0 and 1 in less than epsilon,
    #then choose the most promising value from the Q table for this state.
    if np.random.random() < epsilon:
        return np.argmax(q_values[current_row_index,current_collumn_index])
    else:#chose a random action
        return np.random.randint(4)

#define a function that will get the next location based on the chosen action
def get_next_location(current_row_index,current_collumn_index,action_index):
    new_row_index = current_row_index
    new_collumn_index = current_collumn_index
    if actions[action_index] == 'up' and current_row_index > 0:
        new_row_index -= 1
    elif actions[action_index] == 'right' and current_collumn_index < env_collums-1:
        new_collumn_index += 1
    elif actions[action_index] == 'down' and current_row_index < env_rows - 1 :
        new_row_index += 1
    elif actions[action_index] == 'left' and current_collumn_index > 0:
        new_collumn_index -=1
    return new_row_index, new_collumn_index

# define a function that will get the shortest path between any location within the warehouse that
# the robot is allowed to travel and the item packaging location
def get_shortest_path(start_row_index,start_collumn_index):
    if is_terminal_state(start_row_index,start_collumn_index):
        return[]
    else:
        current_row_index,current_collumn_index = start_row_index,start_collumn_index
        shortest_path = []
        shortest_path.append([current_row_index,current_collumn_index])
        while not is_terminal_state(current_row_index,current_collumn_index):
            actions_index = get_next_action(current_row_index,current_collumn_index,1.)
            current_row_index, current_collumn_index = get_next_location(current_row_index,current_collumn_index,actions_index)
            shortest_path.append([current_row_index,current_collumn_index])
    return shortest_path


#define train parameters
epsilon = 0.9 # the percentage of time when we should take the best action
discount_factor = 0.9 # discound factor for future rewards
learning_rate = 0.9 # the rate at which the AI agent should learn

#run through 10000 training episodes
for episode in range(10000):
    #get the start location for this episode
    row_index,collumn_index = get_starting_location()

    while not is_terminal_state(row_index,collumn_index):
        #choose which action to take
        action_index = get_next_action(row_index,collumn_index,epsilon)

        #perform the chosen action, and transition to the next state
        old_row_index, old_collumn_index = row_index, collumn_index
        row_index,collumn_index = get_next_location(row_index,collumn_index,action_index)

        #receive the reward for moving to the new state , and calcuate the temporal diference
        reward = rewards[row_index,collumn_index]
        old_q_value = q_values[old_row_index,old_collumn_index,action_index]
        temporal_difference = reward + (discount_factor * np.max(q_values[row_index,collumn_index])) - old_q_value

        #update the Q value for the previous state and action pair
        new_q_value = old_q_value + (learning_rate * temporal_difference)
        q_values[old_row_index,old_collumn_index,action_index] = new_q_value


print("training complete!")

#Test
print(get_shortest_path(9,8))
print("------------------")
print(get_shortest_path(5,3))
