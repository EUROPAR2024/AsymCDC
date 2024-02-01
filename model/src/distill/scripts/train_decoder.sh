# kd
CUDA_VISIBLE_DEVICES=2 python train_decoder.py --path_t ./save/models/iRevNet18/speech_irevnet_t.t7 \
                        --dataset speech \
                        --distill kd \
                        --model_s iRevNet20x320 \
                        --learning_rate 0.01 \
                        -r 0.1 \
                        -a 0.9 \
                        -b 0 \
                        --trial 1 \
                        --irev \
                        --epochs 210 \
                        --lr_decay_epochs 120,150,180 \
                        --ec_k 6 \
                        --batch_size 120 \
                        --et