#!/bin/bash
nohup python -u replay.py > log/replay.log 2>&1 &
nohup python -u rand_replace.py > log/rand_replace.log 2>&1 &
nohup python -u replay_none.py > log/replay_none.log 2>&1 &
nohup python -u rand_replace_non.py > log/rand_replace_non.log 2>&1 &