import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def reward_movement(distance,max_distance,last_distance,target_radius=4):
    """
    Standard reward function, calculates reward based on
    distance - between agent and target position
    max_distance - possible with target still in agent field of view
    last_distance - between agent and previous target position
    target_radius
    """
    if distance < target_radius:
        # inside target radius and ahead of or on target (positive)
        reward = (target_radius - distance)/target_radius
        if last_distance < distance:
            # inside target radius but lagging behind
            reward = 0
    else:
        # outside target radius (negative)
        reward = -distance/(max_distance)
    return reward

def visualize_reward(fov_dims=[60,80],target_radius=4,movement=False):
    """
    Function to visualize reward wth respect to target position and movement.
    In the resulting image, each pixel value corresponds to the reward achieved if the agent was at this position.
    """
    frame = np.zeros(fov_dims)
    center = np.array(fov_dims)/2 - .5 # current position
    last_pos = np.array(fov_dims)/2 + .5 # last position
    max_distance = np.linalg.norm(center - np.array([0,0]))
    # calculate reward for each pixel in array
    for i in range(fov_dims[0]):
        for j in range(fov_dims[1]):
            distance = np.linalg.norm(center - np.array([i,j]))
            last_distance = np.linalg.norm(last_pos - np.array([i,j]))
            frame[i,j] = reward_movement(distance,max_distance,last_distance,target_radius)

    # plot image
    fig, ax = plt.subplots()
    cax = ax.imshow(frame, cmap=cm.coolwarm, interpolation='none', vmin=-1, vmax=1)
    ax.arrow(center[1],center[0],-4,-4,head_width=1,color='k')
    cbar = fig.colorbar(cax, ticks=[-1,-.5,0,.5,1])
    plt.grid('off')
    plt.show()

if __name__ == "__main__":
    visualize_reward()
