import matplotlib.pyplot as plt
import numpy as np

with open('cpu.txt', 'r') as f:
    cpu_data = [float(line.strip().replace('ms', '')) for line in f]
with open('gpu.txt', 'r') as f:
    gpu_data = [float(line.strip().replace('ms', '')) for line in f]
with open('gpu_int2.txt', 'r') as f:
    gpu_int2_data = [float(line.strip().replace('ms', '')) for line in f]
with open('gpu_kernel.txt', 'r') as f:
    gpu_kernel_data = [float(line.strip().replace('ms', '')) for line in f]
with open('mega_big_kernel.txt') as f:
    mega_big_kernel_data = [float(line.strip().replace('ms', '')) for line in f]
with open('mega_big_int2.txt') as f:
    mega_big_int2_data = [float(line.strip().replace('ms', '')) for line in f]

# Create x-axis values
x = np.arange(len(cpu_data)) * 685
cpu_data = np.array(cpu_data) / 1e6
gpu_data = np.array(gpu_data) / 1e6

# Plot data
plt.plot(x, cpu_data, label='CPU')
plt.plot(x, gpu_data, label='GPU')

# Add labels and title
plt.xlabel('Number of points')
plt.ylabel('Time (ms)')
plt.title('CPU vs GPU Timing')

plt.yscale('log')

# Add legend
plt.legend()

# Show plot
# plt.show()
plt.savefig('plots/CPUvsGPU.png')

# SPEEDUP ----------------------------------------------------

plt.clf()

speedup = cpu_data / gpu_data
plt.plot(x, speedup, label='Speedup Factor', color='red')

# Add labels and title
plt.xlabel('Number of points')
plt.ylabel('Speedup')
plt.title('Speedup of GPU vs CPU')

# Add legend
plt.legend()

# Show plot
plt.savefig('plots/Speedup.png')

# INT2 ----------------------------------------------------

plt.clf()

gpu_int2_data = np.array(gpu_int2_data) / 1e6

speedup = gpu_data / gpu_int2_data
plt.plot(x, speedup, label='Speedup Factor', color='red')

# Add labels and title
plt.xlabel('Number of points')
plt.ylabel('Speedup')
plt.title('Speedup of Base vs int2 vectorisation')

plt.legend()

# Show plot
plt.savefig('plots/int2.png')

# KERNEL ----------------------------------------------------

plt.clf()

gpu_kernel_data = np.array(gpu_kernel_data) / 1e6

speedup = gpu_int2_data / gpu_kernel_data
plt.plot(x, speedup, label='Speedup Factor', color='red')

# Add labels and title
plt.xlabel('Number of points')
plt.ylabel('Speedup')
plt.title('Speedup of int2 vs patch optimisation')

plt.legend()

# Show plot
plt.savefig('plots/kernel.png')

# ALL_TIMES ----------------------------------------------------

plt.clf()

plt.plot(x, cpu_data, label='CPU')
plt.plot(x, gpu_data, label='GPU')
plt.plot(x, gpu_int2_data, label='GPU int2')
plt.plot(x, gpu_kernel_data, label='GPU patch')

# Add labels and title
plt.xlabel('Number of points')
plt.ylabel('Time (ms)')
plt.title('All Timings')

plt.yscale('log')

# Add legend
plt.legend()

# Show plot
# plt.show()
plt.savefig('plots/all_times.png')

# BIG ----------------------------------------------------

x = np.arange(len(mega_big_kernel_data)) * 3000
mega_big_kernel_data = np.array(mega_big_kernel_data) / 1e6
mega_big_int2_data = np.array(mega_big_int2_data) / 1e6

plt.clf()

plt.plot(x, mega_big_kernel_data, label='GPU patch')
plt.plot(x, mega_big_int2_data, label='GPU int2')

# Add labels and title
plt.xlabel('Number of points')
plt.ylabel('Time (ms)')
plt.title('int2 vs patch Timing')

plt.yscale('log')

# Add legend
plt.legend()

# Show plot
# plt.show()
plt.savefig('plots/big.png')


