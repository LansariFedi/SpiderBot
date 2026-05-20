import pybullet as p
import pybullet_data
import time
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from pprint import pprint


class spider_env(gym.Env):
    def __init__(self):
        super().__init__()
        self.SLIDER_IDS = [
            "L1_J1",
            "L1_J2",
            "L1_J3",
            "L2_J1",
            "L2_J2",
            "L2_J3",
            "L3_J1",
            "L3_J2",
            "L3_J3",
            "L4_J1",
            "L4_J2",
            "L4_J3",
        ]
        self.j_vel = []
        self.j_pose = []
        self.client = p.connect(p.GUI)
        self.FRICTION_VALUE = 1.0
        self.DOWN_LIMIT = 0.03
        self.MAX_STEPS = 1000
        self.current_steps = 0
        self.action_space = spaces.Box(
            low=-0.785, high=0.785, shape=(12,), dtype=np.float32
        )
        obs_high = np.ones(37) * np.inf
        self.observation_space = spaces.Box(
            low=-obs_high, high=obs_high, dtype=np.float32
        )
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf", physicsClientId=self.client)
        self.spider = p.loadURDF(
            "urdf/Spider_URDF.urdf",
            [0, 0, 0.1],
            physicsClientId=self.client,
        )
        for i in [2, 5, 8, 11]:
            p.changeDynamics(
                self.spider,
                i,
                lateralFriction=self.FRICTION_VALUE,
                physicsClientId=self.client,
            )

    def _get_observations(self):
        obs = []
        pos, orn = p.getBasePositionAndOrientation(
            self.spider, physicsClientId=self.client
        )
        for i in pos:
            obs.append(i)
        for i in orn:
            obs.append(i)
        lin_vel, ang_vel = p.getBaseVelocity(self.spider, physicsClientId=self.client)
        for i in lin_vel:
            obs.append(i)
        for i in ang_vel:
            obs.append(i)
        for i in range(p.getNumJoints(self.spider, physicsClientId=self.client) - 1):
            j_pos, j_vel, _, _ = p.getJointState(
                self.spider, i + 1, physicsClientId=self.client
            )
            obs.append(j_pos)
            obs.append(j_vel)
        return np.array(obs, dtype=np.float32)

    def _calculate_reward(self, obs, old_x, action):
        distance_moved = obs[0] - old_x
        base_height = obs[2]
        alive_bonus = 0.1 if distance_moved > 0.005 else 0.0
        energy_pen = -0.01 * np.sum(np.abs(action))
        fall_pen = -5.0 if base_height < self.DOWN_LIMIT else 0.0
        reward = (50.0 * distance_moved) + alive_bonus + energy_pen + fall_pen
        return reward

    def _episode_finished_yet(self, obs):
        truncated = self.current_steps >= self.MAX_STEPS
        terminated = obs[2] < self.DOWN_LIMIT
        return truncated, terminated

    def reset(self, seed=None, options=None):
        self.current_steps = 0
        p.resetSimulation(physicsClientId=self.client)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        p.loadURDF("plane.urdf", physicsClientId=self.client)
        self.spider = p.loadURDF(
            "urdf/Spider_URDF.urdf",
            [0, 0, 0.1],
            physicsClientId=self.client,
        )
        for i in [2, 5, 8, 11]:  # indexes of the lower tips of the spider legs
            p.changeDynamics(
                self.spider,
                i,
                lateralFriction=self.FRICTION_VALUE,
                physicsClientId=self.client,
            )
        for i in range(p.getNumJoints(self.spider) - 1):
            p.resetJointState(
                self.spider,
                i + 1,
                targetValue=0.0,
                targetVelocity=0.0,
                physicsClientId=self.client,
            )
        for _ in range(20):
            p.stepSimulation(physicsClientId=self.client)
        p.resetBaseVelocity(self.spider, [0, 0, 0], [0, 0, 0])
        obs = self._get_observations()
        return obs, {}

    def step(self, action):
        old_x = p.getBasePositionAndOrientation(
            self.spider, physicsClientId=self.client
        )[0][0]
        for i, a in enumerate(action):
            curr_pos = p.getJointState(
                self.spider,
                i + 1,
                physicsClientId=self.client,
            )[0]
            target = curr_pos + (0.1 * a)
            target = np.clip(target, -0.785, 0.785)
            p.setJointMotorControl2(
                self.spider,
                i + 1,
                p.POSITION_CONTROL,
                targetPosition=target,
                force=0.5,
                maxVelocity=1.5,
                physicsClientId=self.client,
            )
        for _ in range(8):
            p.stepSimulation(physicsClientId=self.client)
        self.current_steps += 1
        obs = self._get_observations()
        reward = self._calculate_reward(obs, old_x, action)
        truncated, terminated = self._episode_finished_yet(obs)
        return obs, reward, terminated, truncated, {}

    def close(self):
        p.disconnect(physicsClientId=self.client)


def _make_one_env():
    env = spider_env()
    env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    return env


def make_envs(nb_envs):
    envs = [_make_one_env for _ in range(nb_envs)]
    return envs
