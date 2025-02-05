import argparse
import os 
import subprocess
def main():
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="Submit a hyperparameter tuning job")

    # Adding arguments
    parser.add_argument("-job_name", type=str, help="Name of the job", required=True)  
    parser.add_argument("-env_name", type=str, help="Name of the conda environment to run the job") 
    parser.add_argument("-dataset_name", type=str, help="the name of the dataset folder to use, in the data folder")
    parser.add_argument("-model_type", type=str, help="the model to tune, either RNN or SAN",  choices= ['RNN', 'SAN']) 
    parser.add_argument("-epochs", type=int, help="Number of epochs to train for", default=1000)  
    parser.add_argument("-hyper_config_path", type=str, help="Path to the hyperparameter config yaml file, in the hyper_config folder, enter the absolute path to avoid unexpected errors")

    # Parse arguments
    args = parser.parse_args()
    tmux_session_name = args.job_name
    tmux_session_name = tmux_session_name.replace(" ", "_")


    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Command to run inside tmux
    command = f"""
    cd {script_dir} && \
    conda activate {args.env_name} && \
    python -m main -mode tune -dataset {args.dataset_name} \
    -hyper_config_path {args.hyper_config_path} \
    -model_type {args.model_type} -noiseless_validation -epochs {args.epochs}
    """

    # Create a new tmux session and run the command
    tmux_command = f'tmux new-session -d -s {tmux_session_name} "{command}"'

    # Execute tmux command
    subprocess.run(tmux_command, shell=True, check=True)

    print(f"Hyperparameter tuning job '{args.job_name}' started in tmux session '{tmux_session_name}'.")

if __name__ == "__main__":
    main()

# python3 submit_job.py -job_name tune_lstm_sparse_parity_10_4_bf10 -env_name evan_env -dataset_name sparse_parity_k4_nbits10_n5000_bf10_seed1234 -model_type RNN -epochs 1000 -hyper_config_path /u/a34deng/ResearchDocuments/MindReadingAutobot/mindreadingautobots/hyper_config/rnn_hyper_config.yaml
