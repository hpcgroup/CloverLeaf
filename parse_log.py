import re
import csv
import sys

def parse_clover_log(input_file, output_csv):
    #recompile the patterns
    pattern = re.compile(
        r"\[CLOVER_.*DEBUG\] Transfer \((?P<dir>H2D|D2H)\):.*\| "
        r"Type: (?P<dtype>[a-zA-Z/]+)(?: \(\d+ bytes\))? \| "
        r"Elements: (?P<elements>\d+) \| Total: (?P<size>\d+) bytes"
    )

    data_rows = []
    index = 1

    try:
        with open(input_file, 'r') as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    data_rows.append({
                        'index': index,
                        'type': match.group('dir').lower(),
                        'size': int(match.group('size')),
                        'datatype': match.group('dtype'),
                        'number of elements': int(match.group('elements'))
                    })
                    index += 1

        headers = ['index', 'type', 'size', 'datatype', 'number of elements']
        with open(output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(data_rows)

        print(f"Success! {len(data_rows)} transfers saved to '{output_csv}'")

    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # usage: python parse_clover_log.py <input_log_file>
    input_log = "clover.out"  # default
    output_csv = "data_transfers.csv"
    
    if len(sys.argv) > 1:
        input_log = sys.argv[1]
        
    parse_clover_log(input_log, output_csv)