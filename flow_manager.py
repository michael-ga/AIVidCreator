import os

def continue_create_movement(path, images_paths):
    """
    Check if the movement folder exists and if all images have been processed.
    If not, it will continue processing the images.

    Parameters:
    - path: The path to the directory containing the images.
    """
    movement_folder = os.path.join(path, "movement")
    if not os.path.exists(movement_folder):
        return True
    existing_files = os.listdir(movement_folder)
    return len(existing_files) == len(images_paths) or len(existing_files) == 0
