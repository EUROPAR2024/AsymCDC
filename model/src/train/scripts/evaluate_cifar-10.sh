python CIFAR_wl_main.py --nBlocks 18 18 18 \
                    --nStrides 1 2 2 \
                    --nChannels 16 64 256 \
                    -e \
                    --resume ./checkpoint/cifar10/iRevNet18_32_8_best.pth \
                    --time \
                    -c
