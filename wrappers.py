import gym
import numpy as np
import cv2
from gym import spaces
from PIL import Image

class ResizeWrapper(gym.ObservationWrapper):
    def __init__(self, env=None, shape=(120, 160)):
        super(ResizeWrapper, self).__init__(env)
        if isinstance(shape, str):
            self.shape = eval(shape) + (self.observation_space.shape[2],)
        else:
            self.shape = shape + (self.observation_space.shape[2],)
        self.observation_space.shape = shape
        self.observation_space = spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            self.shape,
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        resized = cv2.resize(observation, self.shape[:2][::-1], interpolation=cv2.INTER_AREA, )
        if len(resized.shape) == 2:
            resized = np.expand_dims(resized, 2)
        return resized

class ClipImageWrapper(gym.ObservationWrapper):
    def __init__(self, env=None, top_margin_divider=3):
        super(ClipImageWrapper, self).__init__(env)
        img_height, img_width, depth = self.observation_space.shape
        top_margin = img_height // top_margin_divider
        img_height = img_height - top_margin
        self.roi = [0, top_margin, img_width, img_height]

        self.observation_space = spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            (img_height, img_width, depth),
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        r = self.roi
        observation = observation[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
        return observation


class NormalizeWrapper(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(NormalizeWrapper, self).__init__(env)
        self.obs_lo = self.observation_space.low[0, 0, 0]
        self.obs_hi = self.observation_space.high[0, 0, 0]
        obs_shape = self.observation_space.shape
        self.observation_space = spaces.Box(0.0, 1.0, obs_shape, dtype=np.float32)

    def observation(self, obs):
        if self.obs_lo == 0.0 and self.obs_hi == 1.0:
            return obs
        else:
            return (obs - self.obs_lo) / (self.obs_hi - self.obs_lo)

class RGB2GrayscaleWrapper(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(RGB2GrayscaleWrapper, self).__init__(env)
        self.observation_space = spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            (self.observation_space.shape[0], self.observation_space.shape[1], 1),
            dtype=self.observation_space.dtype)

    def observation(self, obs):
        # cv2.imshow("Camera", cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))
        # cv2.waitKey(0)
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        # cv2.imshow("Camera2", gray)
        # cv2.waitKey(0)

        # Add an extra dimension, because conv lasers need an input as (batch, height, width, channels)
        gray = np.expand_dims(gray, 2)
        return gray

class DiscreteWrapper(gym.ActionWrapper):
    """
    Duckietown environment with discrete actions (left, right, forward)
    instead of continuous control
    """

    def __init__(self, env):
        gym.ActionWrapper.__init__(self, env)
        self.action_space = spaces.Discrete(3)
        # self.action_space = spaces.Box(low=0., high=1., shape=(3,))

    def action(self, action):
        if isinstance(action, tuple):
            action = action[0]
        # argmax_action = np.argmax(action)
        # sampled_action = np.random.sample([0, 1, 2, 3], 1, p=action)
        # Turn left
        if action == 0:
            vels = [0., 1.]
        #  Go forward
        elif action == 1:
            vels = [1., 1.]
        # Turn right
        elif action == 2:
            vels = [1., 0.]
        # # Stop
        # elif argmax_action == 3:
        #     vels = [0., 0.]
        else:
            assert False, "unknown action"
        return np.array(vels)

class DtRewardWrapperDistanceTravelled(gym.RewardWrapper):
    def __init__(self, env):
        super(DtRewardWrapperDistanceTravelled, self).__init__(env)
        # gym_duckietown.simulator.Simulator):
        self.prev_pos = None

    def reward(self, reward):
        # Baseline reward is a for each step
        my_reward = 0

        # Get current position and store it for the next step
        pos = self.unwrapped.cur_pos
        prev_pos = self.prev_pos
        self.prev_pos = pos
        if prev_pos is None:
            return 0

        # Get the closest point on the curve at the current and previous position
        angle = self.unwrapped.cur_angle
        curve_point, tangent = self.unwrapped.closest_curve_point(pos, angle)
        prev_curve_point, prev_tangent = self.unwrapped.closest_curve_point(prev_pos, angle)
        if curve_point is None or prev_curve_point is None:
            return my_reward
        # Calculate the distance between these points (chord of the curve), curve length would be more accurate
        diff = curve_point - prev_curve_point
        dist = np.linalg.norm(diff)

        try:
            lp = self.unwrapped.get_lane_pos2(pos, self.unwrapped.cur_angle)
            # print("Dist: {:3.2f} | DotDir: {:3.2f} | Angle_deg: {:3.2f}".format(lp.dist, lp.dot_dir, lp.angle_deg))
        except NotInLane:
            return my_reward

        # Dist is negative on the left side of the rignt lane center and is -0.1 on the lane center.
        # The robot is 0.13 (m) wide, to keep the whole vehicle in the right lane, dist should be > -0.1+0.13/2)=0.035
        # 0.05 is a little less conservative
        if lp.dist < -0.05:
            return my_reward
        # Check if the agent moved in the correct direction
        if np.dot(tangent, diff) < 0:
            return my_reward

        # Reward is proportional to the distance travelled at each step
        my_reward = 50 * dist
        if np.isnan(my_reward):
            my_reward = 0.
        return my_reward

class DtRewardCollisionAvoidance(gym.RewardWrapper):
    def __init__(self, env):
        if env is not None:
            super(DtRewardCollisionAvoidance, self).__init__(env)
            #gym_duckietown.simulator.Simulator
        self.prev_proximity_penalty = 0.
        self.proximity_reward = 0.

    def reward(self, reward):
        # Proximity reward is proportional to the change of proximity penalty. Range is ~ 0 - +1.5 (empirical)
        # Moving away from an obstacle is promoted, if the robot and the obstacle are close to each other.
        proximity_penalty = self.unwrapped._proximity_penalty2(self.unwrapped.cur_pos, self.unwrapped.cur_angle)
        self.proximity_reward = -(self.prev_proximity_penalty - proximity_penalty) * 50
        if self.proximity_reward < 0.:
            self.proximity_reward = 0.
        self.prev_proximity_penalty = proximity_penalty
        return reward + self.proximity_reward

    def reset(self, **kwargs):
        self.prev_proximity_penalty = 0.
        self.proximity_reward = 0.
        return self.env.reset(**kwargs)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        if 'custom_rewards' not in info.keys():
            info['custom_rewards'] = {}
        info['custom_rewards']['collision_avoidance'] = self.proximity_reward
        return observation, self.reward(reward), done, info