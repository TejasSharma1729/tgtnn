#!/usr/bin/env python3
import sys
import csv
import os

if __name__ == "__main__":
    # Read the data from the standard input
    if (len(sys.argv) < 2):
        print("Usage: collate_data.py <directory>")
        sys.exit(1)
    directory = sys.argv[1]
    with open(directory + "/results.csv", "w") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["algorithm", "data_depth", "query_depth", "time", "num_dot_products"])
        naive_time = 0.0
        num_files = 0
        naive_num_dot_products = int(sys.argv[1].split("_")[1]) * int(sys.argv[1].split("_")[2])

        file_list = os.listdir(directory)
        file_list.sort()
        for file in file_list:
            if file.endswith(".out"):
                file_split = file[:-4].split("_")
                ddepth = file_split[-2]
                qdepth = file_split[-1]
                algorithm = "_".join(file_split[:-2])
                with open(directory + "/" + file, "r") as input_file:
                    input_file.readline()
                    time = float(input_file.readline().split(" ")[-1])
                    num_dot_products = int(input_file.readline().split(" ")[-1])
                    writer.writerow([algorithm, ddepth, qdepth, time, num_dot_products])
                    input_file.readline()
                    input_file.readline()
                    time = float(input_file.readline().split(" ")[-1])
                    num_dot_products = int(input_file.readline().split(" ")[-1])
                    writer.writerow([algorithm + "_DOUBLE", ddepth, qdepth, time, num_dot_products])
                    input_file.readline()
                    input_file.readline()
                    naive_time += float(input_file.readline().split(" ")[-1])
                    num_dot_products = int(input_file.readline().split(" ")[-1])
                    assert num_dot_products == naive_num_dot_products
                    num_files += 1
        writer.writerow(["NAIVE", "N/A", "N/A", naive_time / num_files, naive_num_dot_products])
