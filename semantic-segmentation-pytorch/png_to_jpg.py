from PIL import Image
import os

# input directory and output directory
input_folder = 'data/first_floor/images/validation'  # png images path
output_folder = 'data/first_floor/images/validation_jpg' # jpg images path

# if the output folder isn't exist, create it 
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# go thorough the files in the folder
for filename in os.listdir(input_folder):
    if filename.endswith('.png'):
        # create the full path of the file 
        input_path = os.path.join(input_folder, filename)
        
        # png to jpg
        img = Image.open(input_path)
        output_filename = os.path.splitext(filename)[0] + '.jpg'
        print("Output Image:"+output_filename)
        output_path = os.path.join(output_folder, output_filename)
        img.save(output_path, 'JPEG')

print("Process FInish")
