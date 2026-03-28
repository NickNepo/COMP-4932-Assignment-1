import cv2
import numpy as np
import os

def generate_morph_video():
    # 1. Configuration
    folder_name = 'images'
    total_steps = 9 
    fps = 10
    output_filename = 'assignment1_morph.avi'
    
    # Check if the folder exists
    if not os.path.exists(folder_name):
        print(f"Error: The folder '{folder_name}' does not exist.")
        return

    # Helper function to get the correct path
    get_path = lambda f: os.path.join(folder_name, f)

    # Load first image to get dimensions from the 'images' folder
    first_frame = cv2.imread(get_path('W0.t1.jpg'))
    if first_frame is None:
        print(f"Error: Could not find W0.t1.jpg in the '{folder_name}' folder.")
        return
    
    height, width, _ = first_frame.shape
    size = (width, height)

    # 2. Define Video Writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter(output_filename, fourcc, fps, size)

    # 3. Add Source Image 0 (t0) from 'images' folder
    img_i0 = cv2.imread(get_path('I0.jpg'))
    if img_i0 is not None:
        video.write(cv2.resize(img_i0, size))

    # 4. Generate, Save, and Add Blended Frames (t1 to t8)
    for i in range(1, total_steps):
        img0_path = get_path(f'W0.t{i}.jpg')
        img1_path = get_path(f'W1.t{i}.jpg')
        
        img0 = cv2.imread(img0_path).astype(np.float32)
        img1 = cv2.imread(img1_path).astype(np.float32)
        
        # Linear Interpolation (Cross-Dissolve)
        weight = i / total_steps
        morphed = cv2.addWeighted(img0, 1.0 - weight, img1, weight, 0)
        morphed_uint8 = morphed.astype(np.uint8)
        
        # Save frame back into the 'images' folder and write to video
        cv2.imwrite(get_path(f'Morphed_t{i}.jpg'), morphed_uint8)
        video.write(morphed_uint8)
        print(f"Processed frame t{i}")

    # 5. Add Source Image 1 (t9) from 'images' folder
    img_i1 = cv2.imread(get_path('I1.jpg'))
    if img_i1 is not None:
        video.write(cv2.resize(img_i1, size))

    # 6. Cleanup
    video.release()
    print(f"Successfully created {output_filename}")

if __name__ == "__main__":
    generate_morph_video()