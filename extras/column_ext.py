import csv

def extract_column(csv_file, column_index, output_file):
    with open(csv_file, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        column_values = [row[column_index] for row in reader]
        column_values = list(set(column_values))
    with open(output_file, 'w') as outfile:
        for value in column_values:
            outfile.write('- '+value + '\n')

# Usage example:
csv_file = 'data.csv'
column_index = 1  # Change this to the index of the column you want to extract (0-based)
output_file = 'data/medical_condition.yml'
extract_column(csv_file, column_index, output_file)
