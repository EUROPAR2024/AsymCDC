python ILSVRC_main.py --data /home/mzq/wl/dataset/imagefolder --nBlocks 6 16 72 6 --nStrides 2 2 2 2 --nChannels 24 96 384 1536 -e \
                      --init_ds 2 --resume checkpoint/ilsvrc2012/pre-trained/ILSVRC_trained_irevnet.pth.tar
