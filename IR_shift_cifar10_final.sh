
module load Anaconda3/2020.11
source activate base
source activate graphot

echo "Generate OTP metric matrix"

source activate graphot
ulimit -c unlimited

# run this script to generate OTP metric matrix, with noise from 0.0 to 1.0, step 0.1, write shell script to run on cluster
# shell for loop
metric_scaler=10
n=50
noise_type=whiteout
delta=0.001
noise=0.0
k=25
shift_pixel=20
range_noise=0

echo "IR, shifting, cifar10, parameter n = $n, noise_type = $noise_type, metric_scaler = $metric_scaler"

python3 gen_w1w2_metric_matrix_IR.py --n $n --data_name cifar10 --shift_pixel $shift_pixel --noise $noise --noise_type $noise_type --k $k --range_noise $range_noise
python3 gen_OTP_metric_matrix_IR.py --n $n --delta $delta --data_name cifar10 --metric_scaler $metric_scaler --shift_pixel $shift_pixel --noise $noise --noise_type $noise_type --k $k --range_noise $range_noise
python3 gen_ROBOT_metric_matrix_IR.py --n $n --data_name mnist --shift_pixel $shift_pixel --noise $noise --noise_type $noise_type --k $k --range_noise $range_noise

python exp_image_retrieval_topk_vs_acc.py --n $n --delta $delta --data_name cifar10 --noise $noise --shift $shift_pixel --metric_scaler $metric_scaler --noise_type $noise_type --range_noise $range_noise
exit