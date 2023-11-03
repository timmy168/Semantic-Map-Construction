from PIL import Image
import os

# 3 folders
input_dir_original = 'data/second_floor/images/validation'
input_dir_result = 'data/second_floor/result_model_2'
input_dir_gt = 'data/second_floor/gt_result'

# output diretory
output_dir = 'data/second_floor/compare_result_model_2'

# get the filename in the directory
original_images = os.listdir(input_dir_original)
result_images = os.listdir(input_dir_result)
semantic_images = os.listdir(input_dir_gt)

# ensure the pattern is the same
original_images.sort()
result_images.sort()
semantic_images.sort()

# create the output folder
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# combine the image
for i in range(len(original_images)):
    original_image = Image.open(os.path.join(input_dir_original, original_images[i]))
    print("Dealing with:"+ os.path.join(input_dir_original, original_images[i]))
    result_image = Image.open(os.path.join(input_dir_result, result_images[i]))
    semantic_image = Image.open(os.path.join(input_dir_gt, semantic_images[i]))

    # check the size of the image is match or not
    if original_image.size == result_image.size == semantic_image.size:
        # create a new image
        combined_image = Image.new('RGB', (original_image.width * 3, original_image.height))
        combined_image.paste(original_image, (0, 0))
        combined_image.paste(semantic_image, (original_image.width, 0))
        combined_image.paste(result_image, (original_image.width * 2, 0))

        # save the conbine image
        combined_image.save(os.path.join(output_dir, original_images[i].replace('.jpg', '_combined.jpg')))

# Finish
print('Processed Finish!')
