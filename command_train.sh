


python train.py \
    --gpu 1 \
    --model model_concat_upsa \
    --data ../data \
    --model_path  ../checkpoint/log_train/model.ckpt \
    --log_dir ../checkpoint/test \
    --num_point 2048 \
    --max_epoch 151 \
    --learning_rate 0.001 \
    --batch_size 2 \
    --lamda 0.5 \
    > test.txt 2>&1 &
