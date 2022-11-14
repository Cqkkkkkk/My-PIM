# Train from sctrach
# python main.py --c ./configs/CUB200_SwinT.yaml

# Train with pretrained Swin-T
# python main.py --c ./configs/CUB200_SwinTPretrained.yaml

# Train from sctrach with positive adj
# python main.py --c ./configs/positive_adj.yaml

python main.py --cfg configs/CUB200_SwinT.yaml

# python infer.py --cfg 