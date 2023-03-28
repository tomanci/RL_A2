import subprocess
import signal
import sys
import time

DEFAULT_NUM_AGENTS = 4
SCREEN_NAME_TEMPLATE = "wandb-agent-{}"

def start_agents(num_agents=DEFAULT_NUM_AGENTS, sweep_agent_command=None):
    if not sweep_agent_command:
        print("Error: sweep agent command is required")
        print("Example command: python wandb_agent_launcher.py 'wandb agent rl-leiden/RL-A2-nn_test_torch/vj00dewt'")
        sys.exit(1)

    processes = []
    for i in range(num_agents):
        screen_name = SCREEN_NAME_TEMPLATE.format(i)
        command = "screen -S {} -dm {}".format(screen_name, sweep_agent_command)
        process = subprocess.Popen(command.split())
        processes.append(process)
        print("Started agent {} in screen session {}".format(i+1, screen_name))
    return processes

def kill_processes(processes):
    for process in processes:
        process.kill()

def cleanup_screens(num_agents=DEFAULT_NUM_AGENTS):
    for i in range(num_agents):
        screen_name = SCREEN_NAME_TEMPLATE.format(i)
        command = "screen -X -S {} quit".format(screen_name)
        subprocess.run(command.split())
        print("Terminated screen session {}".format(screen_name))

def wait_for_screens(num_agents=DEFAULT_NUM_AGENTS):
    while True:
        output = subprocess.check_output(["screen", "-ls"]).decode("utf-8")
        running = False
        for i in range(num_agents):
            screen_name = SCREEN_NAME_TEMPLATE.format(i)
            if screen_name in output:
                running = True
                break
        if not running:
            break
        time.sleep(1)

if __name__ == '__main__':
    num_agents = DEFAULT_NUM_AGENTS
    sweep_agent_command = None

    # Check for command-line arguments
    if len(sys.argv) < 2:
        print("Error: sweep agent command is required")
        print("Example command: python wandb_agent_launcher.py 'wandb agent rl-leiden/RL-A2-nn_test_torch/vj00dewt'")
        sys.exit(1)

    sweep_agent_command = sys.argv[1]

    for arg in sys.argv[2:]:
        num_agents = int(arg)

    processes = start_agents(num_agents, sweep_agent_command)
    print("Started {} agents".format(num_agents))

    def signal_handler(sig, frame):
        print("Terminating processes...")
        kill_processes(processes)
        cleanup_screens(num_agents)
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)

    wait_for_screens(num_agents)
    print("All agents have terminated")

    cleanup_screens(num_agents)
