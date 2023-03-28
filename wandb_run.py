#!/venv/bin/ python3

from dataclasses import asdict
import wandb
from agent_wandb import WandbDQNAgent

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
       
        _ = main.run(config, experience_replay=True, target_network=True, agent_class=WandbDQNAgent)

    except Exception:
        wandb.finish(exit_code=1)
        raise
    else:
        wandb.finish()
    finally:
        print("Finished run")


if __name__ == "__main__":
    wandb_run()
