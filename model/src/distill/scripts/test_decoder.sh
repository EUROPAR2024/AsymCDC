# kd
CUDA_VISIBLE_DEVICES=0 python test_decoder.py \
                        --path_t ./save/models/iRevNet18/speech_irevnet_t.t7 \
                        --path_s ./save/models/iRevNet18/S:iRevNet20x320_T:iRevNet18_speech_kd_r:0.1_a:0.9_b:0.0_1/iRevNet20x320_best.pth \
                        --dataset speech \
                        --model_s iRevNet20x320 \
                        --irev \
                        --ec_k 4 \
                        --batch_size 32 \
                        --et \
                        --es \
                        --en