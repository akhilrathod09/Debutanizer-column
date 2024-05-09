import numpy as np
import pandas as pd
import argparse

#this script decypts the given txt file to csv file which was initially tab seperated

parser =argparse.ArgumentParser(description='Convert text data to CSV format')

#adding the argument for input file
parser.add_argument('input_file', metavar='input_file', type=str,
                    help='input text file containing the data')

#adding the argument for output file
parser.add_argument('output_file', metavar='output_file', type=str,
                    help='output CSV file name')

args=parser.parse_args()

with open(args.input_file,'r') as file:
    lines=file.readlines()

with open(args.output_file,'w') as file:
    #uncomment the below line or edit it accordingly
    #file.write('u1,u2,u3,u4,u5,u6,u7,y\n')

    for line in lines:
        values = line.split()
        file.write(','.join(values) + '\n')

print("Conversion complete. CSV file saved as", args.output_file)