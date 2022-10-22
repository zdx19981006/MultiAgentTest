#!/usr/bin/env python

# Copyright (c) 2021 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import math

import cv2
import gym.spaces
import numpy as np
from gym.spaces import Box, Discrete

import carla

from dqn_example.dqn_experiment import DQNExperiment
from rllib_integration.helper import post_process_image


class LaneFollowAgent(DQNExperiment):

    def __init__(self, config={}):
        super().__init__(config)
        self.allowed_types = [carla.LaneType.Driving, carla.LaneType.Parking]
        self.last_action = None
        self.distance_travelled = None

    def get_observation(self, sensor_data):
        if sensor_data is not None:
            self.diff_lane = 'lane_invasion' in sensor_data.keys()
            self.collision = 'collision' in sensor_data.keys()
        else:
            self.diff_lane = False
            self.collision = False
        return super().get_observation(sensor_data)


    def get_done_status(self, observation, core):
        """Returns whether or not the experiment has to end"""
        hero = core.hero
        self.done_time_idle = self.max_time_idle < self.time_idle
        if self.get_speed(hero) > 1.0:
            self.time_idle = 0
        else:
            self.time_idle += 1
        self.time_episode += 1
        self.done_dist = self.distance_travelled > 2000
        self.done_falling = hero.get_location().z < -0.5
        return self.done_time_idle or self.done_falling or self.done_dist or self.diff_lane or self.collision

    def compute_reward(self, observation, core):
        hero = core.hero

        # Hero-related variables
        hero_location = hero.get_location()
        hero_velocity = self.get_speed(hero)

        # Initialize last location
        if self.last_location is None:
            self.last_location = hero_location

        # Compute deltas
        delta_distance = float(np.sqrt(np.square(hero_location.x - self.last_location.x) +
                                       np.square(hero_location.y - self.last_location.y)))
        self.distance_travelled += delta_distance

        # Update variables
        self.last_location = hero_location
        self.last_velocity = hero_velocity

        # Reward if going forward
        reward = delta_distance

        if self.done_falling:
            reward += -1.0
        if self.done_dist:
            print("Max dist travelled")
            reward += 1.0
        if self.done_time_idle:
            print("Done idle")
            reward += -1.0
        if self.collision:
            print('collision')
            reward += -1.0
        if self.diff_lane:
            reward += -1.0

        return reward
