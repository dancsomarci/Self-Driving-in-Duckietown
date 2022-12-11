import numpy as np
import cv2
import gym
from gym import spaces
from matplotlib import pyplot as plt
import seaborn

import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
from gym_duckietown.simulator import NotInLane

class PreprocessObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env=None, top_margin_divider=3):
        super(PreprocessObservationWrapper, self).__init__(env)
        self.observation_space = spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            (61, 128),
            dtype=self.observation_space.dtype)

    def transform_img(self, img):
        img = cv2.resize(img, (480, 640))
        scale_precent = 0.2
        crop_amount = 35
        newHeight = int(img.shape[0] * scale_precent)
        newWidth = int(img.shape[1] * scale_precent)
        down_scaled_img = cv2.resize(img, (newHeight, newWidth))
        ds_size = down_scaled_img.shape
        cropped_img = down_scaled_img[crop_amount:,:]
        grayscale_img = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2GRAY)
        _, thresholded_img = cv2.threshold(grayscale_img, 150, 255, cv2.THRESH_BINARY)
        normalized_img = thresholded_img / 255
        return normalized_img

    def observation(self, observation):
        return self.transform_img(observation)


class PreprocessForCnnObservationWrapper(PreprocessObservationWrapper):
    def __init__(self, env=None, top_margin_divider=3):
        super(PreprocessObservationWrapper, self).__init__(env)
        self.observation_space = spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            (1, 61, 128),
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        transformed = self.transform_img(observation)
        transformed.reshape((1, 61, 128))
        return transformed


class DQNActionWrapperSimple(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        
    def action(self, act):
        if act == 0:
            vels = [0., 1.]
        #  Go forward
        elif act == 1:
            vels = [1., 1.]
        # Turn right
        elif act == 2:
            vels = [1., 0.]
        else:
            assert False, "unknown action"
        return np.array(vels)


class ResizeWrapper(gym.ObservationWrapper):
    def __init__(self, env=None, shape=(120, 160)):
        super(ResizeWrapper, self).__init__(env)
        if isinstance(shape, str):
            self.shape = eval(shape) + (self.observation_space.shape[2],)
        else:
            self.shape = shape + (self.observation_space.shape[2],)
        #self.observation_space.shape = shape
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
        # Region Of Interest
        # r = [margin_left, margin_top, width, height]
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
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        gray = np.expand_dims(gray, 2)
        return gray


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
            # logger.error("self.unwrapped.closest_curve_point(pos, angle) returned None!!!")
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
            #logger.error("Reward is nan!!!")
        return my_reward


class DtRewardPosAngle(gym.RewardWrapper):
    def __init__(self, env):
        if env is not None:
            super(DtRewardPosAngle, self).__init__(env)
            # gym_duckietown.simulator.Simulator

        self.max_lp_dist = 0.05
        self.max_dev_from_target_angle_deg_narrow = 10
        self.max_dev_from_target_angle_deg_wide = 50
        self.target_angle_deg_at_edge = 45
        self.scale = 1./2.
        self.orientation_reward = 0.

    def reward(self, reward):
        pos = self.unwrapped.cur_pos
        angle = self.unwrapped.cur_angle
        try:
            lp = self.unwrapped.get_lane_pos2(pos, angle)
            # print("Dist: {:3.2f} | DotDir: {:3.2f} | Angle_deg: {:3.2f}". format(lp.dist, lp.dot_dir, lp.angle_deg))
        except NotInLane:
            return -10.

        # print("Dist: {:3.2f} | Angle_deg: {:3.2f}".format(normed_lp_dist, normed_lp_angle))
        angle_narrow_reward, angle_wide_reward = self.calculate_pos_angle_reward(lp.dist, lp.angle_deg)
        #logger.debug("Angle Narrow: {:4.3f} | Angle Wide: {:4.3f} ".format(angle_narrow_reward, angle_wide_reward))
        self.orientation_reward = self.scale * (angle_narrow_reward + angle_wide_reward)

        early_termination_penalty = 0.
        # If the robot leaves the track or collides with an other object it receives a penalty
        # if reward <= -1000.:  # Gym Duckietown gives -1000 for this
        #     early_termination_penalty = -10.
        return self.orientation_reward + early_termination_penalty

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        if 'custom_rewards' not in info.keys():
            info['custom_rewards'] = {}
        info['custom_rewards']['orientation'] = self.orientation_reward
        return observation, self.reward(reward), done, info

    def reset(self, **kwargs):
        self.orientation_reward = 0.
        return self.env.reset(**kwargs)

    @staticmethod
    def leaky_cosine(x):
        slope = 0.05
        if np.abs(x) < np.pi:
            return np.cos(x)
        else:
            return -1. - slope * (np.abs(x)-np.pi)

    @staticmethod
    def gaussian(x, mu=0., sig=1.):
        return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

    def calculate_pos_angle_reward(self, lp_dist, lp_angle):
        normed_lp_dist = lp_dist / self.max_lp_dist
        target_angle = - np.clip(normed_lp_dist, -1., 1.) * self.target_angle_deg_at_edge
        reward_narrow = 0.5 + 0.5 * self.leaky_cosine(
            np.pi * (target_angle - lp_angle) / self.max_dev_from_target_angle_deg_narrow)
        reward_wide = 0.5 + 0.5 * self.leaky_cosine(
            np.pi * (target_angle - lp_angle) / self.max_dev_from_target_angle_deg_wide)
        return reward_narrow, reward_wide

    def plot_reward(self):
        from matplotlib import rcParams
        rcParams['font.family'] = 'serif'
        rcParams['font.size'] = 12

        x = np.linspace(-5, 5, 200)
        fx = np.vectorize(self.leaky_cosine)(x)
        plt.plot(x, 0.5 + 0.5 * fx)
        plt.plot(x, self.gaussian(x))
        plt.legend(["Leaky cosine", "Gaussian"])
        plt.show()

        xcount, ycount = (400, 400)
        x = np.linspace(-0.3, 0.1, xcount)
        y = np.linspace(-90, 90, ycount)
        vpos, vang = np.meshgrid(x, y)
        velocity_reward = 0.
        angle_narrow_reward, angle_wide_reward = np.vectorize(self.calculate_pos_angle_reward)(vpos, vang)
        reward = np.vectorize(self.scale_and_combine_rewards)(angle_narrow_reward, angle_wide_reward, velocity_reward)
        plt.imshow(reward)
        xtic_loc = np.floor(np.linspace(0, xcount - 1, 9)).astype(int)
        ytic_loc = np.floor(np.linspace(0, ycount - 1, 9)).astype(int)
        plt.xticks(xtic_loc, np.round(x[xtic_loc], 2))
        plt.yticks(ytic_loc, (y[ytic_loc]).astype(int))
        plt.colorbar()
        plt.xlabel("Position [m]")
        plt.ylabel("Robot position \n relative to the right lane center [m]")
        plt.grid()
        plt.tight_layout()
        plt.show()

        plt.plot(y, reward[:, 300])
        plt.plot(y, reward[:, 200])
        plt.plot(y, reward[:, 399])
        plt.legend(["At lane center", "At road center and in left lane", "At right road side"])
        plt.xlabel("Orientation")
        plt.ylabel("Reward")
        plt.tight_layout()
        plt.show()

        plt.figure()
        plt.plot(x, np.argmax(reward, axis=0))
        plt.xlabel("Position [m]")
        plt.ylabel("Preferred (maximal reward) orientation")
        plt.yticks(ytic_loc, (y[ytic_loc]).astype(int))
        plt.gca().invert_yaxis()
        seaborn.despine(ax=plt.gca(), offset=0)
        plt.gca().spines['bottom'].set_position('center')
        # plt.gca().spines['left'].set_position('zero')
        plt.grid()
        plt.tight_layout()
        plt.show()

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(vpos, vang, reward, antialiased=False,)
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()


class DtRewardTargetOrientation(DtRewardPosAngle):
    def __init__(self, env):
        super(DtRewardTargetOrientation, self).__init__(env)
        self.max_dev_from_target_angle_deg = 50
        self.scale = 1

    def calculate_pos_angle_reward(self, lp_dist, lp_angle):
        normed_lp_dist = lp_dist / self.max_lp_dist
        target_angle = - np.clip(normed_lp_dist, -1., 1.) * self.target_angle_deg_at_edge
        reward_narrow = 0.5 + 0.5 * self.leaky_cosine(
            np.pi * (target_angle - lp_angle) / self.max_dev_from_target_angle_deg)
        return reward_narrow, 0


class DtRewardClipperWrapper(gym.RewardWrapper):
    def __init__(self, env, clip_high=1000, clip_low=-100):
        super(DtRewardClipperWrapper, self).__init__(env)
        self.clip_high = clip_high
        self.clip_low = clip_low

    def reward(self, reward):
        if np.isnan(reward):
            reward = 0.
            #logger.error("Reward is nan!!!")
        return np.clip(reward, self.clip_low, self.clip_high)


class DtRewardVelocity(gym.RewardWrapper):
    def __init__(self, env):
        if env is not None:
            super(DtRewardVelocity, self).__init__(env)
        self.velocity_reward = 0.

    def reward(self, reward):
        self.velocity_reward = np.max(self.unwrapped.wheelVels) * 0.25
        if np.isnan(self.velocity_reward):
            self.velocity_reward = 0.
            #logger.error("Velocity reward is nan, likely because the action was [nan, nan]!")
        return reward + self.velocity_reward

    def reset(self, **kwargs):
        self.velocity_reward = 0.
        return self.env.reset(**kwargs)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        if 'custom_rewards' not in info.keys():
            info['custom_rewards'] = {}
        info['custom_rewards']['velocity'] = self.velocity_reward
        return observation, self.reward(reward), done, info


class DtRewardCollisionAvoidance(gym.RewardWrapper):
    def __init__(self, env):
        if env is not None:
            super(DtRewardCollisionAvoidance, self).__init__(env)
            # gym_duckietown.simulator.Simulator
        self.prev_proximity_penalty = 0.
        self.proximity_reward = 0.

    def reward(self, reward):
        # Proximity reward is proportional to the change of proximity penalty. Range is ~ 0 - +1.5 (empirical)
        # Moving away from an obstacle is promoted, if the robot and the obstacle are close to each other.
        proximity_penalty = self.unwrapped.proximity_penalty2(self.unwrapped.cur_pos, self.unwrapped.cur_angle)
        self.proximity_reward = -(self.prev_proximity_penalty - proximity_penalty) * 50
        if self.proximity_reward < 0.:
            self.proximity_reward = 0.
        # .debug("Proximity reward: {:.3f}".format(self.proximity_reward))
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
