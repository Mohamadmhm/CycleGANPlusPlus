bash ./datasets/download_cyclegan_dataset.sh horse2zebra

python train.py --dataroot ./datasets/horse2zebra/ --name horse2zebra --model cycle_gan_plusplus

python test.py --dataroot ./datasets/horse2zebra/ --name horse2zebra --model cycle_gan_plusplus
