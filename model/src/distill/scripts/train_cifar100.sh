# kd
CUDA_VISIBLE_DEVICES=4 python train_student.py --path_t ./save/models/iRevNet18/cifar100_irevnet_t.t7 \
                        --dataset cifar100 \
                        --distill kd \
                        --model_s iRevNet1 \
                        -r 0.1 \
                        -a 0.9 \
                        -b 0 \
                        --trial 1 \
                        --irev \
                        --epochs 210 \
                        --lr_decay_epochs 120,150,180