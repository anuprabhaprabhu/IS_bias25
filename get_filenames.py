import os, sys

def get_all_files(parent_folder):
    file_paths = []
    for root, dirs, files in os.walk(parent_folder):
        for file in files:
            file_paths.append(os.path.join(root, file))  # Get full path
    return file_paths

def save_to_file(file_paths, output_file):
    with open(output_file, 'w') as f:
        for file in file_paths:
            f.write(file + '\n')

# Example usage
parent_folder = sys.argv[1]
output_file = "filename.txt"
files = get_all_files(parent_folder)
save_to_file(files, output_file)

print(f"Filenames have been saved to {output_file}")