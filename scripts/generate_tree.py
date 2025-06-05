import os

def generate_tree(startpath, output_file):
    with open(output_file, 'w') as f:
        for root, dirs, files in os.walk(startpath):
            # Exclude .git folder
            if '.git' in dirs:
                dirs.remove('.git')
            level = root.replace(startpath, '').count(os.sep)
            indent = ' ' * 4 * level
            f.write(f'{indent}- {os.path.basename(root)}/\n')
            subindent = ' ' * 4 * (level + 1)
            for file in files:
                f.write(f'{subindent}- {file}\n')

if __name__ == "__main__":
    generate_tree('.', 'directory-tree.md')
