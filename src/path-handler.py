import os
import csv

def collect_file_paths(directory, output_file_path):
    """
    Collects all file paths within a directory and its subdirectories,
    saving them to a CSV file.

    Args:
        directory (str): Path to the directory.
        output_file_path (str): Path to the output CSV file.
    """

    filepaths = []
    for root, dirs, files in os.walk(directory):
        for filename in files:
            filepath = os.path.join(root, filename)
            label = filepath.split('\\')[-2].lower()
            filepaths.append(
                {
                    'path': '\\'.join(filepath.split('\\')[6:]),
                    'label': label
                }
            )

    with open(output_file_path, 'w', newline='') as csvfile:
        fieldnames = ['File Path', 'Label']  # Create appropriate headers if needed
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()  # Write the header row
        for filepath in filepaths:
            writer.writerow({'File Path': filepath['path'], 'Label': filepath['label']})

if __name__ == '__main__':
    directory = input("Enter the directory path: ")
    output_file_path = input("Enter the output CSV file path: ")

    try:
        collect_file_paths(directory, output_file_path)
        print(f"File paths collected successfully and saved to: {output_file_path}")
    except FileNotFoundError:
        print(f"Error: Directory '{directory}' not found.")
    except PermissionError:
        print(f"Error: Insufficient permissions to access directory or file.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
