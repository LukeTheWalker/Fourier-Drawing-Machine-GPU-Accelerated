# script to load and plot data from the txt file

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib.pyplot as plt

#lws = 32; lws <= 1024; lws *= 2
lws = [32, 64, 128, 256, 512, 1024]

#for (int ngroups = 64; ngroups <= 8192; ngroups *= 2){
ngroups = [64, 128, 256, 512, 1024, 2048, 4096, 8192]

# steps = ["filter_contour_by_hand", "merge_close_contours", "filter_vector_by_min", "remove_all_duplicate_points", "biggest_contour_first"]

nsteps = 5
steps = []

# Read data from file
with open('timings.txt', 'r') as f:
    data = f.readlines()

# Extract x, y, z values
x = []
y = []
z = []
name = ""
for line in data:
    # if line contains non digit characters, skip it
    if line[0] == '-':
        z = np.array(z)

        min_point = np.argmin(z)
        print("min point: ", min_point)
        print("min value: ", z[min_point])
        print("ngroups: ", x[min_point])
        print("lws: ", y[min_point])
        print()

        
        # Create 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x, y, z)

        ax.set_xlabel('ngroups')
        ax.set_ylabel('lws')
        ax.set_zlabel('time (ms)')

        plt.savefig("plots/" + name + ".png")

        x = []
        y = []
        z = []
        continue
    if not line[0].isdigit():
        name = line.strip()
        print(line)
        continue

    values = line.split()
    x.append(float(values[0]))
    y.append(float(values[1]))
    z.append(float(values[2]))

