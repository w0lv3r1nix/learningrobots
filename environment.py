import numpy as np
import matplotlib.pyplot as plt
import imageio
from numpy.random import uniform,normal
import reward as rw

class Environment(object):

    def __init__(self, dimensions, agent, target, max_steps=64):
        """
        Initialize environment to "world"-size with maximum number of steps, register agent and target
        """
        self.frame_dims = dimensions
        self.agent = agent
        self.target = target
        self.max_steps = max_steps
        self.reset()

    def reset(self):
        """ Resets environment to start of episode """
        self.n_steps = 0
        self.agent.reset()
        self.target.reset()

        # agent location at center of world
        a_loc = np.array([self.frame_dims[0]/2,self.frame_dims[1]/2])
        self.agent.set_location(a_loc)
        a_fov = self.agent.get_field_of_view()

        # calculate maximum distance for reward function
        self.max_distance = np.linalg.norm(np.array([a_fov[0],a_fov[2]]) - np.array([a_fov[1],a_fov[3]])) / 2

        # set target location inside field of view with 5 pixel margin
        t_loc = np.array([uniform(a_fov[0]+5,a_fov[1]-5),uniform(a_fov[2]+5,a_fov[3]-5)])
        self.target.set_location(t_loc)

    def step(self, action):
        """
        Performs simulation step by updating agent and target position
        calculating reward and checking if episode is finished.
        Returns next observation, reward, whether episode is finished
        and various performance information.
        """
        # provided action is in [0,1,2], convert to [-1,0,1]
        action = action-1

        self.n_steps += 1

        # update agent location
        tmp = self.agent.get_location().copy()
        tmp += action
        self.agent.set_location(tmp)

        # calculate distance to last target position
        distance_to_last = self.get_distance()
        # perform target step
        self.target.step()
        # calculate distance to current target position
        curr_distance = self.get_distance()

        # determine whether target was hit,
        hit_on_target = curr_distance == 0
        if not hit_on_target:
            # was last target hit?
            hit_last_target = distance_to_last == 0
        else:
            hit_last_target = False

        if curr_distance <= self.target.radius:
            # metric "ahead" is 1 if closer to current than to last target position, when inside target radius
            if curr_distance <= distance_to_last:
                ahead = 1
            # negative if closer to last target position
            elif curr_distance > distance_to_last:
                ahead = -1
        else:
            # zero if not in target region
            ahead = 0

        # get next observation
        new_obs = self.get_observation()

        # reward as function of distance, max_distance, target radius and distances to current and last target distance
        reward = rw.reward_movement(self.get_distance(),self.max_distance,distance_to_last,self.target.radius)

        # done if maximum number of steps reached or target is not visible
        done = (self.n_steps == self.max_steps) or (np.sum(new_obs) <= 0)

        # collect metrics in dict
        info = {
            'hit_on_target': hit_on_target,
            'hit_last_target': hit_last_target,
            'ahead': ahead
        }

        return  new_obs, reward, done, info

    def get_state(self, step=-1):
        """ Returns "world-state" of environment at simulation step as rgb-image. """
        target_loc = np.round(self.target.get_location(step).copy()).astype(int)
        agent_loc = np.round(self.agent.get_location(step).copy()).astype(int)

        frame = np.zeros([self.frame_dims[0],self.frame_dims[1],3],dtype=np.float16)
        # target marker in red channel
        frame[target_loc[0]-self.target.radius:target_loc[0]+self.target.radius,target_loc[1]-self.target.radius:target_loc[1]+self.target.radius,0] = self.target.shape

        # agent field of view in blue channel
        a_fov = self.agent.get_field_of_view(step)
        frame[a_fov[0]:a_fov[1], a_fov[2]:a_fov[3], 2] = .5

        return frame

    def get_observation(self,step=-1):
        """
        Returns observation at simulation step
        (only agent field of view with target marker).
        """
        a_fov = self.agent.get_field_of_view(step)
        frame = self.get_state(step)[a_fov[0]:a_fov[1], a_fov[2]:a_fov[3],0]
        return np.expand_dims(frame,-1)

    def get_distance(self,step=-1):
        """ Returns euclidean distance between (rounded) agent and target locations. """
        d = np.linalg.norm(np.round(self.target.get_location(step)) - np.round(self.agent.get_location(step)))
        return d

    def generate_state_gif(self, episode_id='test'):
        """ Saves a visualization of the observations of last episode as gif. """
        images = []
        last_frame = np.zeros([self.agent.input_dims[0],self.agent.input_dims[1],3])
        for i in range(self.n_steps):
            frame = self.get_observation(i)
            frame = np.dstack([frame,frame,frame])
            frame = np.clip(frame*255,0,255).astype(np.uint8)
            images.append(np.clip(frame+last_frame,0,255).astype(np.uint8))
            last_frame = ((frame+last_frame)*.7)
        imageio.mimsave(episode_id+'_state.gif', images, duration=1/24)

    def generate_world_gif(self, episode_id='test'):
        """ Saves a visualization of the world-states of last episode as gif. """
        images = []
        last_frame = np.zeros([self.frame_dims[0],self.frame_dims[1],3])
        for i in range(self.n_steps):
            frame = self.get_state(i)
            frame = np.clip(frame*255,0,255).astype(np.uint8)
            images.append(np.clip(frame+last_frame,0,255).astype(np.uint8))
            last_frame = ((frame+last_frame)*.7)
        imageio.mimsave(episode_id+'_world.gif', images, duration=1/24)
