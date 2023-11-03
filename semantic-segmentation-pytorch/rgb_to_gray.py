import os
import cv2

# Define the input folder containing color images
input_folder = "data/first_floor/annotations/validation"

# Define the output folder to save grayscale images
output_folder = "data/first_floor/annotations/validation_2"

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# List all files in the input folder
files = os.listdir(input_folder)

# Loop through the files and convert color images to grayscale
for file in files:
    input_path = os.path.join(input_folder, file)
    output_path = os.path.join(output_folder, file)

    # Check if the file is an image (you can add more image extensions if needed)
    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')):
        # Read the color image
        color_image = cv2.imread(input_path)

        if color_image is not None:
            # Convert the color image to grayscale
            grayscale_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

            # Save the grayscale image to the output folder
            cv2.imwrite(output_path, grayscale_image)
        else:
            print(f"Error: Unable to read {input_path}")
    else:
        print(f"Skipping non-image file: {file}")

print("Conversion complete. Grayscale images saved in the output folder.")
