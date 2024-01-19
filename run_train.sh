CUDA_VISIBLE_DEVICES=0 python train.py  --model resnet18 \
                                        --data_loc /mnt/ssd2/dataset/CIFAR10 \
                                        --epochs 50 \
                                        --lr 0.02 \
