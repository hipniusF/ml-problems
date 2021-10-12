if [ -z "$1" ]; then
	echo "Path needed"
	exit 1
fi

DIR="$PWD/$1"
tmux has-session -t ml 2>/dev/null
if [ $? != 0 ]; then
	tmux new-session -s ml -d -c $DIR
	tmux rename-window -t ml:1 notebook

	tmux send-keys -t ml:1 "jupyter notebook" Enter

	tmux new-window -n monitor -t ml
	tmux send-keys -t ml:2 "watch -n1 nvidia-smi" Enter
	tmux split-window -t ml:2
	tmux send-keys -t ml:2 "htop" Enter
fi
tmux attach-session -t ml
