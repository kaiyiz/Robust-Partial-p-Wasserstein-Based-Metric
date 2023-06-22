import subprocess

# Define the noise rates to use
noise_rates = [0 0.1 0.2 0.3 0.4 0.5, 0.6, 0.7, 0.8, 0.9, 1]

# Call gen_OTP_metric_matrix.py for each noise rate
for noise_rate in noise_rates:
    # Define the command-line arguments to use
    args = ['python', 'gen_OTP_metric_matrix.py', '--noise', str(noise_rate)]

    # Run the script with the command-line arguments
    subprocess.run(args)