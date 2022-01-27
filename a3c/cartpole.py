"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np


class CartPoleEnv(gym.Env):
    """
    Description:
        A pole is attached by an un-actuated joint to a cart, which moves along
        a frictionless track. The pendulum starts upright, and the goal is to
        prevent it from falling over by increasing and reducing the cart's
        velocity.

    Source:
        This environment corresponds to the version of the cart-pole problem
        described by Barto, Sutton, and Anderson

    Observation:
        Type: Box(4)
        Num     Observation               Min                     Max
        0       Cart Position             -4.8                    4.8
        1       Cart Velocity             -Inf                    Inf
        2       Pole Angle                -0.418 rad (-24 deg)    0.418 rad (24 deg)
        3       Pole Angular Velocity     -Inf                    Inf

    Actions:
        Type: Discrete(2)
        Num   Action
        0     Push cart to the left
        1     Push cart to the right


        Note: The amount the velocity that is reduced or increased is not
        fixed; it depends on the angle the pole is pointing. This is because
        the center of gravity of the pole increases the amount of energy needed
        to move the cart underneath it

    Reward:
        Reward is 1 for every step taken, including the termination step

    Starting State:
        All observations are assigned a uniform random value in [-0.05..0.05]

    Episode Termination:
        Pole Angle is more than 12 degrees.
        Cart Position is more than 2.4 (center of the cart reaches the edge of
        the display).
        Episode length is greater than 200.
        Solved Requirements:
        Considered solved when the average return is greater than or equal to
        195.0 over 100 consecutive trials.
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = 'euler'#运动学积分仪

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.array([self.x_threshold * 2,
                         np.finfo(np.float32).max,
                         self.theta_threshold_radians * 2,
                         np.finfo(np.float32).max],
                        dtype=np.float32)

        self.action_space = spaces.Discrete(2)#env.step(0) ：小车向左，env.step(1) ：小车向右.定义了一个变量空间范围为[0,2) 之间的整数
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)#,义了一个取值范围在（-10，10）的变量 维度为1

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        #初值值是4个[-0.05, 0.05)的随机数:
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
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
        thetaacc = (self.gravity * sintheta - costheta * temp) / (self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass))
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

        self.state = (x, x_dot, theta, theta_dot)#(位置x，x加速度, 偏移角度theta, 角加速度)
        #小车的世界，就一条x轴，
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

        return np.array(self.state), reward, done, {}

    def reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        return np.array(self.state)

    def render(self, mode='human'):#可以绘制当前场景。
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold * 2#有效世界的范围
        scale = screen_width/world_width#世界转屏幕系数
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(.8, .6, .4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth/2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5, .5, .8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

            self._pole_geom = pole

        if self.state is None:
            return None

        # Edit the pole polygon vertex
        pole = self._pole_geom
        l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
        pole.v = [(l, b), (l, t), (r, t), (r, b)]

        x = self.state
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


#--------------------------------------------------------------------------------------------------------------#
##pendulum self exercise
# GYM
# 环境构建
#
# 自我构建的环境为一个类。主要包含：变量、函数
# 必须的变量state,action
# 对应的两个空间为observation_space和action_space
# 这两个空间必须要用space
# 文件夹下的类在__init__中进行定义。其中state是一个object一般为一个np.array包含多个状态指示值。

    # def __init__(self):
    #     self.max_speed = 8
    #     self.max_torque = 2.
    #     self.dt = .05
    #     self.g = 10.0
    #     self.m = 1.
    #     self.l = 1.
    #     self.viewer = None
    #
    #     high = np.array([1., 1., self.max_speed], dtype=np.float32)
    #     self.action_space = spaces.Box(
    #         low=-self.max_torque,
    #         high=self.max_torque, shape=(1,),
    #         dtype=np.float32
    #     )
    #     self.observation_space = spaces.Box(
    #         low=-high,
    #         high=high,
    #         dtype=np.float32
    #     )
    #     self.seed()

# 必须存在的函数

# step:利用动作环境给出的一下步动作和环境给出的奖励（核心）
# 承担了最重要的功能，是所构建环境所实现功能的位置
# 输入为动作,输出为下一个状态值object,反馈float值,done（终结标志） 布尔值0或者1,info（对调试有用的任何信息） any

# reset
# 重置环境, 将状态设置为初始状态，返回： 状态值

# render
# 在图形界面上作出反应,可以没有，但是必须存在

# close
# 关闭图形界面

# seed
# 随机种子,可以没有，但是必须存在
#
#     def seed(self, seed=None):
#         self.np_random, seed = seeding.np_random(seed)
#         return [seed]
#
#     def step(self, u):
#         th, thdot = self.state  # th := theta
#
#         g = self.g
#         m = self.m
#         l = self.l
#         dt = self.dt
#
#         u = np.clip(u, -self.max_torque, self.max_torque)#a中的所有数限定到范围a_min和a_max
#         self.last_u = u  # for rendering
#         costs = angle_normalize(th) ** 2 + .1 * thdot ** 2 + .001 * (u ** 2)
#
#         newthdot = thdot + (-3 * g / (2 * l) * np.sin(th + np.pi) + 3. / (m * l ** 2) * u) * dt
#         newth = th + newthdot * dt
#         newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
#
#         self.state = np.array([newth, newthdot])
#         return self._get_obs(), -costs, False, {}
#
#     def reset(self):
#         high = np.array([np.pi, 1])
#         self.state = self.np_random.uniform(low=-high, high=high)
#         self.last_u = None
#         return self._get_obs()
#
#     def _get_obs(self):
#         theta, thetadot = self.state
#         return np.array([np.cos(theta), np.sin(theta), thetadot])
#
#     def render(self, mode='human'):
#         if self.viewer is None:
#             from gym.envs.classic_control import rendering
#             self.viewer = rendering.Viewer(500, 500)
#             self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
#             rod = rendering.make_capsule(1, .2)
#             rod.set_color(.8, .3, .3)
#             self.pole_transform = rendering.Transform()
#             rod.add_attr(self.pole_transform)
#             self.viewer.add_geom(rod)
#             axle = rendering.make_circle(.05)
#             axle.set_color(0, 0, 0)
#             self.viewer.add_geom(axle)
#             fname = path.join(path.dirname(__file__), "assets/clockwise.png")
#             self.img = rendering.Image(fname, 1., 1.)
#             self.imgtrans = rendering.Transform()
#             self.img.add_attr(self.imgtrans)
#
#         self.viewer.add_onetime(self.img)
#         self.pole_transform.set_rotation(self.state[0] + np.pi / 2)
#         if self.last_u:
#             self.imgtrans.scale = (-self.last_u / 2, np.abs(self.last_u) / 2)
#
#         return self.viewer.render(return_rgb_array=mode == 'rgb_array')
#
#     def close(self):
#         if self.viewer:
#             self.viewer.close()
#             self.viewer = None
#
# def angle_normalize(x):
#         return (((x + np.pi) % (2 * np.pi)) - np.pi)
