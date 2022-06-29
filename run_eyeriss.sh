cd ./src/Eyeriss
CUDA_VISIBLE_DEVICES=1 python main.py --fitness EDP --num_pe 168 --l1_size 86016 --l2_size 108000 --NocBW 81920000 --slevel_min 2 --slevel_max 3 --epochs 20 --model resnet50
cd ../..
