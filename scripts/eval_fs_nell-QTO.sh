DATADIR=../KG_data
MODELDIR=../FuzzSys/models
ADJ_PATH=./QTO/neural_adj/NELL_10_0.0002_2.5.pt

BACKBONE=QTO

CUDA_VISIBLE_DEVICES=2 python3 main.py --do_test --data_path ${DATADIR}/NELL-betae -n 1 -b 1000 -d 1000 --cpu_num 0 --geo cqd --print_on_screen --test_batch_size 1 --checkpoint_path ${MODELDIR}/nell-betae --cqd discrete --cuda --adj_path ${ADJ_PATH} --dataname NELL --backbone_type ${BACKBONE}