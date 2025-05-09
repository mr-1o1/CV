def get_image_directory():
    """
    Returns the directory path of the images.
    """
    import os

    # Get the current file's directory (where utils.py is located)
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Move up one level and join with 'images'
    image_directory = os.path.normpath(os.path.join(current_dir, '..', 'images'))

    print(f"Image directory: {image_directory}")

    # Check if the directory exists
    if not os.path.exists(image_directory):
        print(f"Directory does not exist: {image_directory}")
        os.makedirs(image_directory, exist_ok=True)
        print(f"Created directory: {image_directory}")

    return image_directory