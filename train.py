from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback
from env import make_envs, spider_env
import gymnasium as gym
import tensorboard
import os
import time


def Agent(env):
    return PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-5,
        n_steps=2048,
        batch_size=512,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.15,
        ent_coef=0.003,
        verbose=1,
        device="cpu",
    )


def test(model_path="ppo_spider"):
    folder_path = "old_model/"
    eval_env = SubprocVecEnv(make_envs(1))
    model_path = folder_path + model_path
    vec_path = folder_path + "vec_normalize.pkl"
    if os.path.exists(vec_path):
        eval_env = VecNormalize.load(vec_path, eval_env)
        eval_env.training = False
        eval_env.norm_reward = False

    model = PPO.load(model_path, device="cpu")
    obs = eval_env.reset()
    while True:
        try:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
            eval_env.render(mode="human")
            time.sleep(0.05)
        except KeyboardInterrupt:
            break
    eval_env.close()


def train():
    env = SubprocVecEnv(env_fns=make_envs(4))
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    eval_env = SubprocVecEnv(env_fns=make_envs(1))
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    model = Agent(env)
    save_dir = "best_models"
    os.makedirs(save_dir, exist_ok=True)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_dir,
        log_path=save_dir,
        eval_freq=50_000,
        deterministic=True,
        render=False,
    )
    model.learn(
        total_timesteps=5_000_000,
        progress_bar=False,
        callback=eval_callback,
        log_interval=100,
    )
    model.save("ppo_spider")
    env.save("vec_normalize.pkl")
    env.close()
    eval_env.close()


if __name__ == "__main__":
    # train()
    test()
