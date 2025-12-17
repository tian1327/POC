import os


def cleanup_downloads(path):
    """Cleans up the downloaded images in the specified directory by:
    1. Removing files without a valid image suffix.
    2. Renaming files to have a consistent .jpg suffix.
    """

    # loop through each folder in the train directory
    # find all the possible suffixes and any file that does not have a valid suffix

    suffixes = {}
    total_ct = 0
    no_suffix_ct = 0
    invalid_suffix_ct = 0
    removed_ct = 0
    renamed_ct = 0
    valid_suffixes = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}

    for folder in os.listdir(path):
        # print(folder)
        folder_path = os.path.join(path, folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                # print(file)
                total_ct += 1
                file_path = os.path.join(folder_path, file)
                if os.path.isfile(file_path):
                    # get the suffix, print an error if no suffix
                    _, suffix = os.path.splitext(file)
                    if not suffix:
                        # print(f"Error: {file} has no suffix")
                        no_suffix_ct += 1
                        # remove the file
                        os.remove(file_path)
                        removed_ct += 1
                    else: # has suffix
                    
                        if suffix not in suffixes:
                            suffixes[suffix] = 1
                        else:
                            suffixes[suffix] += 1
                        
                        if suffix not in valid_suffixes:
                            invalid_suffix_ct += 1
                            # remove the file
                            os.remove(file_path)
                            removed_ct += 1
                        else:
                            # replace the suffix with .jpg
                            if suffix != '.jpg':
                                new_file = file.replace(suffix, '.jpg')
                                new_file_path = os.path.join(folder_path, new_file)
                                os.rename(file_path, new_file_path)
                                # print(f"Renamed {file} to {new_file}")
                                renamed_ct += 1
                        
    print(f"Total files: {total_ct}")
    print(f"Files with no suffix: {no_suffix_ct}")
    print(f"Invalid suffix files: {invalid_suffix_ct}")
    print(f"Removed files: {removed_ct}")
    print(f"Renamed files: {renamed_ct}")
    print(suffixes)

if __name__ == "__main__":

    path_train = '../species196/train/'
    path_val = '../species196/val/'
    print("Cleaning up train directory...")
    cleanup_downloads(path_train)
    print("\nCleaning up val directory...")
    cleanup_downloads(path_val)
    print("\nCleanup complete.")
