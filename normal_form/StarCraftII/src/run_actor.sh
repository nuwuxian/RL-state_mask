for i in $(seq 0 50); do
	python train_ppo_masknet_bot.py --job_name=actor --learner_ip localhost &
done;
