# import cv2
# import os

# def add_strobe_effect(img, duration, strobe_fps):
#     strobe_frames = int(duration * strobe_fps)
#     for _ in range(strobe_frames):
#         yield img
#         yield 255 - img  # Invert the colors to create a strobe effect

# def images_to_video(image_folder, video_name, output_directory, fps=24, strobe_effect=False, strobe_fps=10):
#     # Get the list of image files in the specified folder
#     images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
#     # Sort the list of image files alphabetically
#     images.sort()

#     # Read the first image to get its dimensions
#     frame = cv2.imread(os.path.join(image_folder, images[0]))
#     height, width, layers = frame.shape

#     # Create a video writer object
#     output_path = os.path.join(output_directory, video_name)
#     video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

#     # Iterate through the sorted images and add them to the video
#     for image in images:
#         img_path = os.path.join(image_folder, image)
#         img = cv2.imread(img_path)

#         if strobe_effect:
#             # Add strobe effect frames
#             for strobe_frame in add_strobe_effect(img, duration=0.2, strobe_fps=strobe_fps):
#                 video.write(strobe_frame)
#         else:
#             # Add normal frames without strobe effect
#             for _ in range(int(fps * 0.5)):
#                 video.write(img)

#     # Release the video writer and close any open windows
#     cv2.destroyAllWindows()
#     video.release()

# Specify the path to the folder containing the images
image_folder = "/Users/borjan/code/python/AnesthesiaProjectEmergence/results/preopt/notitle/png"
# Specify the name of the output video file
#video_name = "output_video_with_strobe.mp4"
# Specify the path to the output directory
output_directory = "/Users/borjan/code/python/AnesthesiaProjectEmergence/results"
# Enable strobe effect
#strobe_effect_enabled = True
# Call the function to create and save the video with strobe effect
#images_to_video(image_folder, video_name, output_directory, strobe_effect=strobe_effect_enabled)


# %%

import cv2
import os
import numpy as np

def add_strobe_effect(img, duration, strobe_fps):
    strobe_frames = int(duration * strobe_fps)
    for _ in range(strobe_frames):
        yield img
        yield 255 - img  # Invert the colors to create a strobe effect

def apply_random_transform(frame):
    # Randomly scale the image (zoom in or out)
    scale_factor = np.random.uniform(0.9, 1.1)
    frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor)

    # Randomly translate the image (move in x and y direction)
    rows, cols, _ = frame.shape
    max_translate = 20
    dx = np.random.randint(-max_translate, max_translate)
    dy = np.random.randint(-max_translate, max_translate)
    translation_matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    frame = cv2.warpAffine(frame, translation_matrix, (cols, rows))

    return frame

def images_to_video(image_folder, video_name, output_directory, fps=24, strobe_effect=False, strobe_fps=10, compression=True, random_transform=True):
    # Get the list of image files in the specified folder
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    # Sort the list of image files alphabetically
    images.sort()

    # Read the first image to get its dimensions
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    # Create a video writer object with compression settings
    output_path = os.path.join(output_directory, video_name)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') if compression else cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Iterate through the sorted images and add them to the video
    for image in images:
        img_path = os.path.join(image_folder, image)
        img = cv2.imread(img_path)

        if random_transform:
            img = apply_random_transform(img)

        if strobe_effect:
            # Add strobe effect frames
            for strobe_frame in add_strobe_effect(img, duration=0.5, strobe_fps=strobe_fps):
                video.write(strobe_frame)
        else:
            # Add normal frames without strobe effect
            for _ in range(int(fps * 0.5)):
                video.write(img)

    # Release the video writer and close any open windows
    cv2.destroyAllWindows()
    video.release()


# Specify the name of the output video file
video_name = "output_video_transformed.mp4"
# Enable strobe effect
strobe_effect_enabled = True
# Enable video compression
compression_enabled = True
# Enable random transformations
random_transform_enabled = True
# Call the function to create and save the video with random transformations and strobe effect
images_to_video(image_folder, video_name, output_directory, strobe_effect=strobe_effect_enabled, compression=compression_enabled, random_transform=random_transform_enabled)
