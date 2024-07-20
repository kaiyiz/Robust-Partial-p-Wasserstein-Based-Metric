
metric_scaler=1
n=50
noise_type=rand1pxl
delta=0.0001
noise=0.0
k=30
shift_pixel=2
range_noise=1

echo "IR, shifting, noise, mnist, parameter n = $n, noise_type = $noise_type, metric_scaler = $metric_scaler"

python3 gen_w1w2_metric_matrix_IR.py --n $n --data_name mnist --shift_pixel $shift_pixel --noise $noise --noise_type $noise_type --k $k --range_noise $range_noise
python3 gen_OTP_metric_matrix_IR.py --n $n --delta $delta --data_name mnist --metric_scaler $metric_scaler --shift_pixel $shift_pixel --noise $noise --noise_type $noise_type --k $k --range_noise $range_noise
python3 gen_ROBOT_metric_matrix_IR.py --n $n --data_name mnist --shift_pixel $shift_pixel --noise $noise --noise_type $noise_type --k $k --range_noise $range_noise

python exp_image_retrieval_topk_vs_acc.py --n $n --delta $delta --data_name mnist --noise $noise --shift $shift_pixel --metric_scaler $metric_scaler --noise_type $noise_type --range_noise $range_noise

exit