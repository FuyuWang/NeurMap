cd ./src/TPU3
CUDA_VISIBLE_DEVICES=0 python main.py --fitness EDP --num_pe 65536 --l1_size 4194304 --l2_size 25165824 --NocBW 81920000 --slevel_min 2 --slevel_max 3 --epochs 20 --model mobilenet_v2
cd ../..
