# kd
CUDA_VISIBLE_DEVICES=2 python train_student.py --path_t ./save/models/iRevNet18/cifar10_irevnet_t.t7 \
                        --dataset cifar10 \
                        --distill kd \
                        --model_s iRevNet4 \
                        -r 0.1 \
                        -a 0.9 \
                        -b 0 \
                        --trial 1 \
                        --irev \
                        --epochs 210 \
                        --lr_decay_epochs 120,150,180