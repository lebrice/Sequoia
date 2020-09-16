"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import math
import random
from abc import ABC
from typing import ClassVar, Dict, List, Optional, Sequence, Tuple

import gym
import numpy as np
from gym import logger, spaces
from gym.envs.classic_control import CartPoleEnv
from gym.envs.registration import register
from gym.utils import seeding

class MultiTaskCartPole(CartPoleEnv):
    # Same as the regular CartPoleEnv, but the task parameters (physics, size of
    # the cart, lengths, masses, etc) can change.

    def __init__(self, noise_std: float = 0.1):
        super().__init__()
        self.noise_std: float = noise_std
        self.default_task = np.array([
            self.gravity,
            self.masscart,
            self.masspole,
            self.length,
            self.force_mag,
            self.tau,
        ])
        self.current_task = self.default_task.copy() 

    def step(self, action):
        observation, reward, done, info = super().step(action)
        # TODO: Do we really want to concatenate the task to the state? 
        # observation = np.concatenate([self.state, self.task], axis=0)
        return observation, reward, done, info

    def random_task(self) -> np.ndarray:
        mult = np.random.normal(
            loc=1,
            scale=self.noise_std,
            size=self.default_task.shape,
        )
        # Only allow values from 0 to 3 times the default task.
        mult = mult.clip(0.1, 3.0)
        task = mult * self.default_task
        return task

    def set_new_task(self) -> None:
        self.gravity = 4.9 + random.random() * 3.0 * 4.9 
        self.masscart = 0.5 + random.random() * 3.0 * 0.5 
        self.masspole = 0.05 + random.random() * 3.0 * 0.05 
        self.length = 0.25 + random.random() * 3.0 * 0.25 
        self.force_mag = 5.0 + random.random() * 3.0 * 5.0
        self.tau = 0.01 + random.random() * 3.0 * 0.01

        self.current_task = np.array([
            self.gravity,
            self.masscart,
            self.masspole,
            self.length,
            self.force_mag,
            self.tau,
        ])

    def reset(self, new_task: bool = False):
        if new_task:
            self.set_new_task()
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        observation = np.concatenate([self.state, self.current_task], axis=0)
        return observation

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold*2
        scale = screen_width/world_width
        carty = 100 # TOP OF CART
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l,r,t,b = -cartwidth/2, cartwidth/2, cartheight/2, -cartheight/2
            axleoffset =cartheight/4.0
            cart = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
            pole = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            pole.set_color(.8,.6,.4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth/2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5,.5,.8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0,carty), (screen_width,carty))
            self.track.set_color(0,0,0)
            self.viewer.add_geom(self.track)

            self._pole_geom = pole

        if self.state is None: return None

        # Edit the pole polygon vertex
        pole = self._pole_geom
        l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
        pole.v = [(l,b), (l,t), (r,t), (r,b)]

        x = self.state
        cartx = x[0]*scale+screen_width/2.0 # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

# try:
#     register(
#         id='MultiTaskCartPole-v0',
#         entry_point='settings.active.rl.mtcartpole:MultiTaskCartPole',
#     )
# except gym.error.Error:
#     pass
