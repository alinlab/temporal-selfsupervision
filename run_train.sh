python main.py --cfg configs/timesformer/ssv2/train.yaml \
ERM_TRAIN.BATCH_SIZE 16 \
SOLVER.MAX_EPOCH 1

python main.py --cfg configs/motionformer/ssv2/train.yaml \
ERM_TRAIN.BATCH_SIZE 16 \
SOLVER.MAX_EPOCH 1

python main.py --cfg configs/xvit/ssv2/train.yaml \
ERM_TRAIN.BATCH_SIZE 16 \
SOLVER.MAX_EPOCH 1
