"""
# @Time    : 2021/7/2 5:22 下午
# @Author  : hezhiqiang01
# @Email   : hezhiqiang01@baidu.com
# @File    : env.py
"""
#只需要编写这一部分的代码，就可以无缝衔接MAPPO。
import numpy as np

import math
import gym
from gym import spaces, logger
from gym.utils import seeding

class Env(object):
    """
    # 环境中的智能体
    """
    def __init__(self, i):
        # self.agent_num = 3  # 设置智能体(小飞机)的个数，这里设置为两个
        # self.obs_dim = 4  # 设置智能体的观测纬度14
        # self.action_dim = 2  # 设置智能体的动作纬度，这里假定为一个五个纬度的



       #CartPoleEnv

        # # Angle limit set to 2 * theta_threshold_radians so failing observation
        self.agent_num = 1  # 设置智能体的个数，这里设置为两个
        self.obs_dim = 4  # 设置智能体的观测纬度
        self.action_dim = 2  # 设置智能体的动作纬度，这里假定为一个五个纬度的

        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = 'euler'  # 运动学积分仪

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4
        # is still within bounds.
        high = np.array([self.x_threshold * 2,
                         np.finfo(np.float32).max,
                         self.theta_threshold_radians * 2,
                         np.finfo(np.float32).max],
                        dtype=np.float32)

        self.action_space = spaces.Discrete(2)  # env.step(0) ：小车向左，env.step(1) ：小车向右.定义了一个变量空间范围为[0,2) 之间的整数
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)  # ,义了一个取值范围在（-10，10）的变量 维度为1

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None



    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        """
        # self.agent_num设定为2个智能体时，返回值为一个list，每个list里面为一个shape = (self.obs_dim, )的观测数据
        """
        ##new
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        return np.array(self.state)


        sub_agent_obs = []
        for i in range(self.agent_num):
            sub_obs = np.random.random(size=(self.obs_dim, ))#14，obs_dim
            sub_agent_obs.append(sub_obs)
        return sub_agent_obs




        #new
        # sub_agent_obs = []
        # for i in range(self.agent_num):
        #     self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        #     sub_agent_obs.append(self.state)
        # return sub_agent_obs


# # if true, action is a number 0...N, otherwise action is a one-hot N-dimensional vector
        #self.discrete_action_input = False
    def step(self, actions):
        """
        # self.agent_num设定为2个智能体时，actions的输入为一个2纬的list，每个list里面为一个shape = (self.action_dim, )的动作数据
        # 默认参数情况下，输入为一个list，里面含有两个元素，因为动作纬度为5，所里每个元素shape = (5, )
        """
        # sub_agent_obs = []
        # sub_agent_reward = []
        # sub_agent_done = []
        # sub_agent_info = []
        # for i in range(self.agent_num):
        #     sub_agent_obs.append(np.random.random(size=(15,)))
        #     sub_agent_reward.append([np.random.rand()])
        #     sub_agent_done.append(False)
        #     sub_agent_info.append({})
        #
        # #print('sub_agent_obs',sub_agent_obs, 'sub_agent_reward', sub_agent_reward, 'sub_agent_done', sub_agent_done, 'sub_agent_info',sub_agent_info)
        # return [sub_agent_obs, sub_agent_reward, sub_agent_done, sub_agent_info]


    #new
        #err_msg = "%r (%s) invalid" % (actions, type(actions))
        #assert self.action_space.contains(actions), err_msg
        sub_agent_obs = []
        sub_agent_reward = []
        sub_agent_done = []
        sub_agent_info = []
        for i in range(self.agent_num):
            x, x_dot, theta, theta_dot = self.state
            #print("actions[i]",actions)
            force = self.force_mag if actions[i][0] == 1 else -self.force_mag
            # if action == 1:
            #     force = self.force_mag
            # elif action == 0:
            #     force = -self.force_mag
            # else:
            #     force = 0
            costheta = math.cos(theta)
            sintheta = math.sin(theta)

            # For the interested reader:
            # https://coneural.org/florian/papers/05_cart_pole.pdf
            temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
            thetaacc = (self.gravity * sintheta - costheta * temp) / (
                        self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass))
            xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

            if self.kinematics_integrator == 'euler':
                x = x + self.tau * x_dot
                x_dot = x_dot + self.tau * xacc
                theta = theta + self.tau * theta_dot
                theta_dot = theta_dot + self.tau * thetaacc
            else:  # semi-implicit euler
                x_dot = x_dot + self.tau * xacc
                x = x + self.tau * x_dot
                theta_dot = theta_dot + self.tau * thetaacc
                theta = theta + self.tau * theta_dot

            self.state = (x, x_dot, theta, theta_dot)  # (位置x，x加速度, 偏移角度theta, 角加速度)
            # 小车的世界，就一条x轴，
            # 变量env.x_threshold里存放着小车坐标的最大值（=2.4），
            # 超过这个数值，世界结束，每step()一次，就会奖励 1，直到上次done为True。
            done = bool(
                x < -self.x_threshold
                or x > self.x_threshold
                or theta < -self.theta_threshold_radians
                or theta > self.theta_threshold_radians
            )

            if not done:
                reward = 1.0
            elif self.steps_beyond_done is None:
                # Pole just fell!
                self.steps_beyond_done = 0
                reward = 1.0
            else:
                if self.steps_beyond_done == 0:
                    logger.warn(
                        "You are calling 'step()' even though this "
                        "environment has already returned done = True. You "
                        "should always call 'reset()' once you receive 'done = "
                        "True' -- any further steps are undefined behavior."
                    )
                self.steps_beyond_done += 1
                reward = 0.0
            sub_agent_obs.append(np.array(self.state))
            sub_agent_reward.append(reward)
            sub_agent_done.append(done)
            sub_agent_info.append({})

        return [sub_agent_obs, sub_agent_reward, sub_agent_done, sub_agent_info]
