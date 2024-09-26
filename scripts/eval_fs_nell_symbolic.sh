DATADIR=../KG_data
MODELDIR=../FuzzSys/models

CUDA_VISIBLE_DEVICES=2 python3 main.py --do_test --data_path ${DATADIR}/NELL-betae -n 1 -b 1000 -d 1000 --cpu_num 0 --geo cqd --tasks 1p --print_on_screen --test_batch_size 1 --checkpoint_path ${MODELDIR}/nell-betae --cqd discrete --cuda --dataname NELL --backbone_type CQD --memb_type symbolic

CUDA_VISIBLE_DEVICES=2 python3 main.py --do_test --data_path ${DATADIR}/NELL-betae -n 1 -b 1000 -d 1000 --cpu_num 0 --geo cqd --tasks 2p --print_on_screen --test_batch_size 1 --checkpoint_path ${MODELDIR}/nell-betae --cqd discrete --cqd-t-norm prod --cuda --dataname NELL --backbone_type CQD --memb_type symbolic

CUDA_VISIBLE_DEVICES=2 python3 main.py --do_test --data_path ${DATADIR}/NELL-betae -n 1 -b 1000 -d 1000 --cpu_num 0 --geo cqd --tasks 3p --print_on_screen --test_batch_size 1 --checkpoint_path ${MODELDIR}/nell-betae --cqd discrete --cqd-t-norm prod --cqd-sigmoid --cqd-k 2 --cuda --dataname NELL --backbone_type CQD --memb_type symbolic

CUDA_VISIBLE_DEVICES=2 python3 main.py --do_test --data_path ${DATADIR}/NELL-betae -n 1 -b 1000 -d 1000 --cpu_num 0 --geo cqd --tasks 2i --print_on_screen --test_batch_size 1 --checkpoint_path ${MODELDIR}/nell-betae --cqd discrete --cqd-t-norm prod --cuda --dataname NELL --backbone_type CQD --memb_type symbolic

CUDA_VISIBLE_DEVICES=2 python3 main.py --do_test --data_path ${DATADIR}/NELL-betae -n 1 -b 1000 -d 1000 --cpu_num 0 --geo cqd --tasks 3i --print_on_screen --test_batch_size 1 --checkpoint_path ${MODELDIR}/nell-betae --cqd discrete --cqd-t-norm prod --cuda --dataname NELL --backbone_type CQD --memb_type symbolic

CUDA_VISIBLE_DEVICES=2 python3 main.py --do_test --data_path ${DATADIR}/NELL-betae -n 1 -b 1000 -d 1000 --cpu_num 0 --geo cqd --tasks ip --print_on_screen --test_batch_size 1 --checkpoint_path ${MODELDIR}/nell-betae --cqd discrete --cqd-t-norm prod --cuda --dataname NELL --backbone_type CQD --memb_type symbolic

CUDA_VISIBLE_DEVICES=2 python3 main.py --do_test --data_path ${DATADIR}/NELL-betae -n 1 -b 1000 -d 1000 --cpu_num 0 --geo cqd --tasks pi --print_on_screen --test_batch_size 1 --checkpoint_path ${MODELDIR}/nell-betae --cqd discrete --cqd-t-norm prod --cuda --dataname NELL --backbone_type CQD --memb_type symbolic

CUDA_VISIBLE_DEVICES=2 python3 main.py --do_test --data_path ${DATADIR}/NELL-betae -n 1 -b 1000 -d 1000 --cpu_num 0 --geo cqd --tasks 2u --print_on_screen --test_batch_size 1 --checkpoint_path ${MODELDIR}/nell-betae --cqd discrete --cqd-t-norm prod --cuda --dataname NELL --backbone_type CQD --memb_type symbolic

CUDA_VISIBLE_DEVICES=2 python3 main.py --do_test --data_path ${DATADIR}/NELL-betae -n 1 -b 1000 -d 1000 --cpu_num 0 --geo cqd --tasks up --print_on_screen --test_batch_size 1 --checkpoint_path ${MODELDIR}/nell-betae --cqd discrete --cqd-t-norm prod --cuda --dataname NELL --backbone_type CQD --memb_type symbolic

CUDA_VISIBLE_DEVICES=1 python3 main.py --do_test --data_path ${DATADIR}/NELL-betae -n 1 -b 1000 -d 1000 --cpu_num 0 --geo cqd --tasks 2in --print_on_screen --test_batch_size 1 --checkpoint_path ${MODELDIR}/nell-betae --cqd discrete --cqd-t-norm prod --cuda --dataname NELL --backbone_type CQD --memb_type symbolic

CUDA_VISIBLE_DEVICES=1 python3 main.py --do_test --data_path ${DATADIR}/NELL-betae -n 1 -b 1000 -d 1000 --cpu_num 0 --geo cqd --tasks 3in --print_on_screen --test_batch_size 1 --checkpoint_path ${MODELDIR}/nell-betae --cqd discrete --cqd-t-norm prod --cuda --dataname NELL --backbone_type CQD --memb_type symbolic

CUDA_VISIBLE_DEVICES=1 python3 main.py --do_test --data_path ${DATADIR}/NELL-betae -n 1 -b 1000 -d 1000 --cpu_num 0 --geo cqd --tasks inp --print_on_screen --test_batch_size 1 --checkpoint_path ${MODELDIR}/nell-betae --cqd discrete --cqd-t-norm prod --cuda --dataname NELL --backbone_type CQD --memb_type symbolic

CUDA_VISIBLE_DEVICES=1 python3 main.py --do_test --data_path ${DATADIR}/NELL-betae -n 1 -b 1000 -d 1000 --cpu_num 0 --geo cqd --tasks pin --print_on_screen --test_batch_size 1 --checkpoint_path ${MODELDIR}/nell-betae --cqd discrete --cqd-t-norm prod --cuda --dataname NELL --backbone_type CQD --memb_type symbolic

CUDA_VISIBLE_DEVICES=1 python3 main.py --do_test --data_path ${DATADIR}/NELL-betae -n 1 -b 1000 -d 1000 --cpu_num 0 --geo cqd --tasks pni --print_on_screen --test_batch_size 1 --checkpoint_path ${MODELDIR}/nell-betae --cqd discrete --cqd-t-norm prod --cuda --dataname NELL --backbone_type CQD --memb_type symbolic