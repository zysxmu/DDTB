

############edsr

edsr_x2_eval() {
python3 main_limitrange_incremental.py --scale 2 --model EDSR \
--k_bits 4 --save_results --test_only \
--data_test Set5+Set14+B100+Urban100  \
--save "../experiment/output/edsrx2/4bit" --dir_data ./datasets
}

edsr_x4_eval() {
python3 main_limitrange_incremental.py --scale 4 --model EDSR \
--k_bits 4 --save_results --test_only \
--data_test Set5+Set14+B100+Urban100  \
--save "../experiment/output/edsrx4/4bit" --dir_data ./datasets
}


##############srresnet

bnsrresnet_x2_eval() {
python3 main_limitrange_incremental.py --scale 2 --model bnsrresnet \
--k_bits 4 --save_results --test_only \
--data_test Set5+Set14+B100+Urban100  \
--save "../experiment/output_tmp/bnsrresnetx2/4bit" --dir_data ./datasets
}

bnsrresnet_x4_eval() {
python3 main_limitrange_incremental.py --scale 4 --model bnsrresnet \
--k_bits 4 --save_results --test_only \
--data_test Set5+Set14+B100+Urban100  \
--save "../experiment/output_tmp/bnsrresnetx4/4bit" --dir_data ./datasets
}

##############rdn

rdn_x4_eval() {
python3 main_limitrange_incremental.py --scale 4 --model RDN \
--k_bits 4 --save_results --test_only \
--data_test Set5+Set14+B100+Urban100  \
--save "../experiment/output/rdnx4/4bit" --dir_data ./datasets
}

rdn_x2_eval() {
python3 main_limitrange_incremental.py --scale 2 --model RDN \
--k_bits 4 --save_results --test_only \
--data_test Set5+Set14+B100+Urban100  \
--save "../experiment/output/rdnx2/4bit" --dir_data ./datasets
}