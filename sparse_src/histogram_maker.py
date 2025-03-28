#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np

# Initialize lists to store parsed data
bin_edges = []  # Store the left edges of bins
bin_counts = []  # Store the frequencies

# Read the file
file_path = "histogram_data.txt"  # Change this to your actual file path
with open(file_path, "r") as file:
    for line in file:
        # Extract the bin range and count
        range_part, count_part = line.strip().split(":")
        lower, upper = map(float, range_part.strip("[]").split(","))
        count = int(count_part.strip())

        # Store bin edges and frequencies
        bin_edges.append(lower)  # Store only the lower edge
        bin_counts.append(count)

# Ensure we have the last upper bound to define the full range
bin_edges.append(upper)  # Add the last upper bound

# Plot the histogram using bars
plt.figure(figsize=(10, 5))
plt.bar(bin_edges[:-1], bin_counts, width=np.diff(bin_edges), align="edge", edgecolor="black")

# Labels and title
plt.xlabel("Value Range")
plt.ylabel("Frequency")
plt.title("Histogram Distribution of Dot Products")
plt.savefig("dot_products.jpg")
