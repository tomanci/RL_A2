#!/venv/bin/ python3

from dataclasses import asdict
import numpy as np
import wandb

from config import Config, EpsilonConfig, TempConfig
import main as main


def wandb_run():
    config=Config() # default config

    wandb.init(
        # set the wandb project where this run will be logged
        project="RL-A2-nn_test_torch",
        
        # track hyperparameters and run metadata
        config=asdict(config)
    )

    try:
        # recreate the config object we use for the run with the latest parameters
        # the parametes may have been changed by wandb for a hyperparameter run
        epslion_config = EpsilonConfig(**wandb.config["epsilon"])
        temp_config = TempConfig(**wandb.config["temp"])
        top_level_config = {**wandb.config}
        del(top_level_config["epsilon"])
        del(top_level_config["temp"])
        config = Config(epsilon=epslion_config, temp=temp_config, **top_level_config)
       
        run_rewards = main.run(config, experience_replay=True, target_network=True)

        # log all metrics to wandb
        for step,episode_reward in enumerate(run_rewards):
            avg_reward_last_100 = np.mean(run_rewards[:step+1][-100:])
            metrics={"episode_reward": episode_reward, "avg_reward_last_100": avg_reward_last_100}
            wandb.log(metrics, step=step)

        # TODO: Log Pytorch model to wandb
        # wandb.watch(model, log="all", log_freq=100, log_graph=True)
    except Exception:
        wandb.finish(exit_code=1)
        raise
    else:
        wandb.finish()
    finally:
        print("Finished run")


if __name__ == "__main__":
    wandb_run()
