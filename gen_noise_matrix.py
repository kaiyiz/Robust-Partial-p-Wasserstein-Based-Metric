import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--n', type=int, default=200)
parser.add_argument('--delta', type=float, default=0.01)
parser.add_argument('--data_name', type=str, default='mnist')
parser.add_argument('--metric_scaler', type=float, default=1.0)
args = parser.parse_args()
print(args)

n = int(args.n)
delta = args.delta
data_name = args.data_name
metric_scaler = args.metric_scaler
argparse = "n_{}_delta_{}_data_{}_noise_{}_ms_{}".format(n, delta, data_name, noise, metric_scaler)

# Define the noise rates to use
noise_rates = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

# Call gen_OTP_metric_matrix.py for each noise rate
for noise_rate in noise_rates:
    # Define the command-line arguments to use
    args = ['python', 'gen_OTP_metric_matrix.py', '--noise', str(noise_rate), '--n', str(n), '--delta', str(n), '--data_name', data_name, '--metric_scaler', str(metric_scaler)]

    # Run the script with the command-line arguments
    subprocess.run(args)