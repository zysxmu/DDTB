##############edsr

edsr_baseline_x2() {
python main_ori.py --model edsr --scale 2 \
--save edsr_baseline_x2 \
--patch_size 96 \
--epochs 300 \
--decay 200 \
--gclip 0 \
--dir_data ./datasets
}

edsr_baseline_x4() {
python main_ori.py --model edsr --scale 4 \
--save edsr_baseline_x4 \
--patch_size 192 \
--epochs 300 \
--decay 200 \
--gclip 0 \
--dir_data ./datasets
}

##############srresnet

srresnet_baseline_x2() {
python main_ori.py --model bnsrresnet --scale 2 \
--save srresnet_baseline_x2 \
--patch_size 96 \
--epochs 500 \
--decay 200 \
--gclip 0 \
--dir_data ./datasets
}

srresnet_baseline_x4() {
python main_ori.py --model bnsrresnet --scale 4 \
--save srresnet_baseline_x4 \
--patch_size 192 \
--epochs 500 \
--decay 200 \
--gclip 0 \
--dir_data ./datasets
}

###############rdn

rdn_baseline_x2() {
python main_ori.py --model rdn --scale 2 \
--save rdn_baseline_x4 \
--patch_size 64 \
--epochs 300 \
--decay 200 \
--gclip 0 \
--dir_data ./datasets
}


rdn_baseline_x4() {
python main_ori.py --model rdn --scale 4 \
--save rdn_baseline_x4 \
--patch_size 192 \
--epochs 300 \
--decay 200 \
--gclip 0 \
--dir_data ./datasets
}


################## eval trained fp models!
srresnet_baseline_x2_eval() {
python3 main_ori.py --scale 2 --model bnsrresnet \
--save_results --test_only \
--data_test Set5+Set14+B100+Urban100  \
--refine 'path_to_pretrained' --dir_data ./datasets \
--save "bnsrresnet_baseline_x2" 
}
å
srresnet_baseline_x4_eval() {
python3 main_ori.py --scale 4 --model bnsrresnet \
--save_results --test_only \
--data_test Set5+Set14+B100+Urban100  \
--refine 'path_to_pretrained' --dir_data ./datasets \
--save "bnsrresnet_baseline_x4"
}

edsr_baseline_x2_eval() {
python3 main_ori.py --scale 2 --model edsr \
--save_results --test_only \
--data_test Set5+Set14+B100+Urban100  \
--refine 'path_to_pretrained' --dir_data ./datasets \
--save "edsr_baseline_x2"
}

edsr_baseline_x4_eval() {
python3 main_ori.py --scale 4 --model edsr \
--save_results --test_only \
--data_test Set5+Set14+B100+Urban100  \
--refine 'path_to_pretrained' --dir_data ./datasets \
--save "edsr_baseline_x4"
}

rdn_baseline_x2_eval() {
python3 main_ori.py --scale 2 --model rdn \
--save_results --test_only \
--data_test Set5+Set14+B100+Urban100  \
--refine 'path_to_pretrained' --dir_data ./datasets \
--save "rdn_baseline_x2"
}

rdn_baseline_x4_eval() {
python3 main_ori.py --scale 4 --model rdn \
--save_results --test_only \
--data_test Set5+Set14+B100+Urban100  \
--refine 'path_to_pretrained' --dir_data ./datasets \
--save "rdn_baseline_x4"
}
