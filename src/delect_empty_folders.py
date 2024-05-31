import os

def remove_empty_folders(root_folder):
    root_folder = os.path.abspath(os.path.join(os.path.expanduser('~'), root_folder))
    for root, dirs, files in os.walk(root_folder, topdown=False):
        for folder in dirs:
            folder_path = os.path.join(root, folder)
            if not os.listdir(folder_path):
                os.rmdir(folder_path)
    if not os.listdir(root_folder):
        os.rmdir(root_folder)

# Only save the active bugs
remove_empty_folders('defects4j/tmp')