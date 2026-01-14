import os
import subprocess

home_folder = os.path.join("/home")
# list files and directories in home folder
files = os.listdir(home_folder)

for file in files:
    file_path = os.path.join(home_folder, file)
    # execute a linux command and return output
    output = subprocess.run(["du", "-sh", file_path], capture_output=True, text=True, check=False).stdout

    print(f"{file} {output}")
