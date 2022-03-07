# Code for our paper: Dynamic Dual Trainable Bounds for Ultra-low Precision Super-Resolution Networks [paper]()

## Dependence
* Python 3.6
* PyTorch >= 1.7.0

## Datasets
Please download DIV2K datasets.

Then, create a directory 'datasets' and re-organise the downloaded dataset directory as follows:

```
...
option.py
main_limitrange_incremental.py
datasets
  benchmark
  DIV2K
```


## Usage

### 1: train full-precision models:

An example:
```buildoutcfg
python main_ori.py --model edsr --scale 4 \
--save edsr_baseline_x4 \
--patch_size 192 \
--epochs 300 \
--decay 200 \
--gclip 0 \
--dir_data ./datasets
```

Please refer to 'baseline.sh' for more commands.

### 2: train quantized models:

An example:
```
python main_limitrange_incremental.py --scale 4 \
--k_bits 4 --model EDSR \
--pre_train ./pretrained/edsr_baseline_x4.pt --patch_size 192 \
--data_test Set14 \
--dynamic_ratio 0.3 \
--save "output/edsrx4/4bit" --dir_data ./datasets --print_every 10
```

Please refer to 'run.sh' for more commands.

### 3: test quantized models

An example:
```
python3 main_limitrange_incremental.py --scale 4 --model EDSR \
--k_bits 4 --save_results --test_only \
--data_test Set5+Set14+B100+Urban100  \
--save "../experiment/output/edsrx4/4bit" --dir_data ./datasets
```

Please refer to 'test.sh' for more commands.

### calculate PSNR/SSIM

After saving the images, modify path in`metrics/calculate_PSNR_SSIM.m` to generate results.

```
matlab -nodesktop -nosplash -r "calculate_PSNR_SSIM('$dataset',$scale,$bit);quit"
```

refer to `metrics/run.sh` for more details.

##  Trained FP models and quantized models: [here](https://drive.google.com/drive/folders/1NIn8utKaLv9607fdA_pHGUsw9mwZgeXl?usp=sharing)

Download these model. Then use the commands above to obtain the reported results of the paper.



| Model                                            | Bit                                                                     
|--------------------------------------------------|-----------------------------------------------------------------------------
| EDSRx4 |                     [2](https://drive.google.com/file/d/1VveOu1t6XwPI4SxA3qoxVqXokRaUfnCY/view?usp=sharing)          
|               |      [3](https://drive.google.com/file/d/1N68cmRryRuv02ya7s7iU0xKMGZHapuAP/view?usp=sharing)                                                                     
|          |       [4](https://drive.google.com/file/d/1AEQb2-FDXGN_cJTi9lIzm-ZwMi420MtI/view?usp=sharing) 
| EDSRx2 |                     [2](https://drive.google.com/file/d/1QigliZlQZHqe8hL2Z61Zk7Eihk6lTLm3/view?usp=sharing)          
|               |      [3](https://drive.google.com/file/d/1isFtKm1g-s2ngClKDwAGEmKmv_K2haCT/view?usp=sharing)                                                                     
|          |       [4](https://drive.google.com/file/d/1zKHjKhBfwshY3ic_fUSGdSio5eqx3kgp/view?usp=sharing)  
| RDNx4 |                     [2](https://drive.google.com/file/d/1Dd94TPJOxNk6z9R6LX6di0rlWxIqNAjo/view?usp=sharing)          
|               |      [3](https://drive.google.com/file/d/1qZON0gbk_zA3hBIx9XlG54S32Yx9breu/view?usp=sharing)                                                                     
|          |       [4](https://drive.google.com/file/d/1TtPQ3GNsl73J51EvewUlP-OVbH6Ykb6f/view?usp=sharing) 
| RDNx2 |                     [2](https://drive.google.com/file/d/1IejVXhBG9A-uM3Uo3_HBh0EeUzt8KwKh/view?usp=sharing)          
|               |      [3](https://drive.google.com/file/d/17oKk2PIGWZhuJYGjSPPgLUvUnpnq1G2W/view?usp=sharing)                                                                     
|          |       [4](https://drive.google.com/file/d/1ajCa0TXTQnSMBmUE050d_Hnf13EOs941/view?usp=sharing) 
| SRResNetx4 |                     [2](https://drive.google.com/file/d/1Lo1BWkf3zSqVfZsMuMA5Bq4XFBPsUeO2/view?usp=sharing)          
|               |      [3](https://drive.google.com/file/d/1yr-r2oYLF0LWwYwkQtgXAoOaXTX3G-5M/view?usp=sharing)                                                                     
|          |       [4](https://drive.google.com/file/d/1_VXmCJkmPj9lehfYMI5gaiTesgTgFRg8/view?usp=sharing) 
| SRResNetx2 |                     [2](https://drive.google.com/file/d/1K66QWvXypbIJOUYKxyqLY59y5cSt5lsA/view?usp=sharing)          
|               |      [3](https://drive.google.com/file/d/1NLWqRw8qDbIYzs2_hBOTtwdXmtNAtFEr/view?usp=sharing)                                                                     
|          |       [4](https://drive.google.com/file/d/1ldS1MYa16kRpaOFR94U2pNmIDiXzBZYE/view?usp=sharing)   



Results of pre-trained models are shown below:

### EDSR
| Model                                            | Dataset                                                                     | Bit |  DDTB(Ours)    |
|--------------------------------------------------|-----------------------------------------------------------------------------|-----|----------------------|
| EDSRx4 |                     Set5                                                        |  2  |             30.97/0.876         |               
|                                                  |                                                                             | [3](https://drive.google.com/file/d/1N68cmRryRuv02ya7s7iU0xKMGZHapuAP/view?usp=sharing)   |  31.52/0.883   |
|                                                  |                                                                             | [4](https://drive.google.com/file/d/1AEQb2-FDXGN_cJTi9lIzm-ZwMi420MtI/view?usp=sharing)   |  31.85/0.889   |
|                                                  | Set14      |  2   |                  27.87/0.764    |
|                                                  |                                                                             | 3   |  28.18/0.771   |
|                                                  |                                                                             | 4   |  28.39/0.777   |
|                                                  | BSD100 |   2  |                   27.09/0.719   |
|                                                  |                                                                             | 3   |  27.30/0.727   |
|                                                  |                                                                             | 4   |  27.44/0.732   |
|                                                  | Urban100  |  2   |               24.82/0.742       |
|                                                  |                                                                             | 3   |  25.33/0.761   |
|                                                  |                                                                             | 4   |  25.69/0.774   |
| EDSRx2 |               Set5                                                               |  2   |      37.25/0.958                |
|                                                  |                                                                             | 3   |  37.51/0.958   |
|                                                  |                                                                             | 4   |  37.72/0.959   |
|                                                  | Set14      |  2   |             32.87/0.911         |
|                                                  |                                                                             | 3   |  33.17/0.914   |
|                                                  |                                                                             | 4   |  33.35/0.916   |
|                                                  | BSD100 |   2  |           31.67/0.893           |
|                                                  |                                                                             | 3   |  31.89/0.896   |
|                                                  |                                                                             | 4   |  32.01/0.898   |
|                                                  | Urban100  |   2  |             30.34/0.910         |
|                                                  |                                                                             | 3   |  31.01/0.919   |
|                                                  |                                                                             | 4   |  31.39/0.922   |

### RDN
| Model                                           | Dataset                                                                     | Bit |   DDTB(Ours)    |
|-------------------------------------------------|-----------------------------------------------------------------------------|-----|----------------------|
| RDNx4 |                                                Set5                             |  2   |           30.57/0.867           |
|                                                 |                                                                             | 3   |   31.49/0.883   |
|                                                 |                                                                             | 4   |   31.97/0.891   |
|                                                 | Set14      |  2   |         27.56/0.757             |
|                                                 |                                                                             | 3   |   28.17/0.772   |
|                                                 |                                                                             | 4   |   28.49/0.780   |
|                                                 | BDS100 |  2   |    26.91/0.714                  |
|                                                 |                                                                             | 3   |   27.30/0.728   |
|                                                 |                                                                             | 4   |   27.49/0.735   |
|                                                 |   Urban100    |   2  |      24.50/0.728                |
|                                                 |                                                                             | 3   |   25.35/0.764   |
|                                                 |                                                                             | 4   |   25.90/0.783   |
| RDNx2 |                                       Set5                                      |  2   |    36.76/0.955                  |
|                                                 |                                                                             | 3   |   37.61/0.959   |
|                                                 |                                                                             | 4   |   37.88/0.960   |
|                                                 |   Set14        |  2   |         32.54/0.908             |
|                                                 |                                                                             | 3   |   33.26/0.915   |
|                                                 |                                                                             | 4   |   33.51/0.917   |
|                                                 |   BSD100   |  2   |         31.44/0.890             |
|                                                 |                                                                             | 3   |   31.91/0.897   |
|                                                 |                                                                             | 4   |   32.12/0.899   |
|                                                 |   Urban100    |   2  |     29.77/0.903                 |
|                                                 |                                                                             | 3   |   31.10/0.920   |
|                                                 |                                                                             | 4   |   31.76/0.926   |

### SRResNet
| Model                                                | Dataset                                                                     | Bit |   DDTB(Ours)    |
|------------------------------------------------------|-----------------------------------------------------------------------------|-----|----------------------|
|   SRResNetx4 |                           Set5                                                  |  2   |     31.51/0.887                 |
|                                                      |                                                                             | 3   |   31.85/0.890   |
|                                                      |                                                                             | 4   |   31.97/0.892   |
|                                                      |   Set14        |  2   |      28.23/0.773                |
|                                                      |                                                                             | 3   |   28.39/0.776   |
|                                                      |                                                                             | 4   |   28.46/0.778   |
|                                                      |   BSD100   |   2  |        27.33/0.728              |
|                                                      |                                                                             | 3   |   27.44/0.731   |
|                                                      |                                                                             | 4   |   27.48/0.733   |
|                                                      |   Urban100    | 2    |     25.37/0.762                 |
|                                                      |                                                                             | 3   |   25.64/0.770   |
|                                                      |                                                                             | 4   |   25.77/0.776   |
|   SRResNetx2 |                                       Set5                                      |   2  |    37.46/0.958                  |
|                                                      |                                                                             | 3   |   37.67/0.959   |
|                                                      |                                                                             | 4   |   37.78/0.960   |
|                                                      |   Set14        |  2   |         33.02/0.913             |
|                                                      |                                                                             | 3   |   33.24/0.915   |
|                                                      |                                                                             | 4   |   33.32/0.916   |
|                                                      |   BSD100   |  2   |           31.78/0.895           |
|                                                      |                                                                             | 3   |   31.95/0.897   |
|                                                      |                                                                             | 4   |   32.03/0.898   |
|                                                      |   Urban100    |   2  |       30.57/0.913               |
|                                                      |                                                                             | 3   |   31.15/0.919   |
|                                                      |                                                                             | 4   |   31.40/0.921   |


## Contact

For any question, be free to contact: viper.zhong@gmail.com. The github issue is also welcome.