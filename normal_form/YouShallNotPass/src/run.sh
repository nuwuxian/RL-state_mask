for i in 0 1 2; do
    nohup python -u mask_train.py > console_$i.txt 2>&1 &
    sleep 10
done
