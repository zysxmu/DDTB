############edsr

edsr_x4() {
python main_limitrange_incremental.py --scale 4 \
--k_bits 4 --model EDSR \
--pre_train ../edsr_baseline_x4.pt --patch_size 192 \
--data_test Set14 \
--dynamic_ratio 0.3 \
--save "output/edsrx4/4bit" --dir_data ./datasets --print_every 100
}


edsr_x2() {
python main_limitrange_incremental.py --scale 2 \
--k_bits 4 --model EDSR \
--pre_train ../edsr_baseline_x2.pt --patch_size 96 \
--data_test Set14 \
--dynamic_ratio 0.3 \
--save "output/edsrx2/4bit" --dir_data ./datasets --print_every 100
}

##############srresnet

bnsrresnet_x2() {
python main_limitrange_incremental.py --scale 2 \
--k_bits 4 --model bnsrresnet \
--pre_train ../bnsrresnet_baseline_x2.pt --patch_size 96 \
--data_test Set14 \
--dynamic_ratio 0.1 \
--save "output/bnsrresnetx2/4bit" --dir_data ./datasets --print_every 100
}

bnsrresnet_x4() {
python main_limitrange_incremental.py --scale 4 \
--k_bits 4 --model bnsrresnet \
--pre_train ../bnsrresnet_baseline_x4.pt --patch_size 192 \
--data_test Set14 \
--dynamic_ratio 0.1 \
--save "output/bnsrresnetx4/4bit" --dir_data ./datasets --print_every 100
}

##############rdn


rdn_x4() {
python main_limitrange_incremental.py --scale 4 \
--k_bits 4 --model RDN \
--pre_train ../rdn_baseline_x4.pt --patch_size 96 \
--data_test Set14 \
--dynamic_ratio 0.5 \
--save "output/rdnx4/4bit" --dir_data ./datasets --print_every 100
}

rdn_x2() {
python main_limitrange_incremental.py --scale 2 \
--k_bits 4 --model RDN \
--pre_train ../rdn_baseline_x2.pt --patch_size 64 \
--data_test Set14 \
--dynamic_ratio 0.5 \
--save "output/rdnx2/4bit" --dir_data ./datasets --print_every 100
}