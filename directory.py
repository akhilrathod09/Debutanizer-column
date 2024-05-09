def print_directory_structure(directory, indent=''):
    import os

    items = os.listdir(directory)
    items.sort()

    for item in items:
        if item != '.git':
            path = os.path.join(directory, item)
            if os.path.isdir(path):
                print(indent + '├── ' + item + '/')
                print_directory_structure(path, indent + '│   ')
            else:
                print(indent + '├── ' + item)


if __name__ == "__main__":
    print_directory_structure('.')
