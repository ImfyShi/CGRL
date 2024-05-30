#!/usr/bin/zsh

# script_dir=$(cd $(dirname $0);pwd)
# echo ${script_dir}

rm -rf ../data/generated_results
mkdir ../data/generated_results
mkdir ../data/generated_results/deterministic
## Create a Tmux session "mywork" in a window "window0" started in the background.
tmux new-session -d -s collect_results -n window0

## Split the window to 4 panes.
tmux split-window -h -t collect_results:window0
tmux split-window -v -t collect_results:window0.0
tmux split-window -v -t collect_results:window0.2

tmux send -t collect_results:window0.0 "conda activate research_lab" ENTER
tmux send -t collect_results:window0.1 "conda activate research_lab" ENTER
tmux send -t collect_results:window0.2 "conda activate research_lab" ENTER
tmux send -t collect_results:window0.3 "conda activate research_lab" ENTER

tmux send -t collect_results:window0.0 "python eval_model.py --instance_set_group ai --policy_mode deterministic" ENTER
tmux send -t collect_results:window0.1 "python eval_model.py --instance_set_group ani --policy_mode deterministic" ENTER
tmux send -t collect_results:window0.2 "python eval_model.py --instance_set_group irinch --policy_mode deterministic" ENTER
tmux send -t collect_results:window0.3 "python eval_model.py --instance_set_group random --policy_mode deterministic" ENTER
