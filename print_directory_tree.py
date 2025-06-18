import os

# Folders to skip
<<<<<<< HEAD
SKIP_FOLDERS = {'.github', '.venv', 'food_segmentation_pipeline_backup.git', '.git'}
# Extensions to ignore
IGNORE_EXTENSIONS = ('.jpg', '.png', '.json')
=======
SKIP_FOLDERS = {'.github', '.venv', 'food_segmentation_pipeline_backup.git', '.git', '.trunk'}
# Extensions to ignore
IGNORE_EXTENSIONS = ('.jpg', '.png', '.json', '.txt')
>>>>>>> 82a126b (Complete Meal or Portion integration)

def print_directory_tree(root_dir, indent='', is_last=True):
    items = sorted(os.listdir(root_dir))
    items = [item for item in items if item not in SKIP_FOLDERS]

    for index, item in enumerate(items):
        path = os.path.join(root_dir, item)
        is_last_item = (index == len(items) - 1)
        connector = '└── ' if is_last_item else '├── '

        # Skip files with unwanted extensions
<<<<<<< HEAD
        # if os.path.isfile(path) and item.lower().endswith(IGNORE_EXTENSIONS):
        #     continue
=======
        if os.path.isfile(path) and item.lower().endswith(IGNORE_EXTENSIONS):
            continue
>>>>>>> 82a126b (Complete Meal or Portion integration)

        print(indent + connector + item)

        if os.path.isdir(path):
            new_indent = indent + ('    ' if is_last_item else '│   ')
            print_directory_tree(path, new_indent, is_last_item)

if __name__ == "__main__":
    repo_path = r'E:\food_segmentation_pipeline'
    print(f'[FOLDER] {repo_path}')
    print_directory_tree(repo_path)
