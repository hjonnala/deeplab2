from PIL import Image
import os

image_dir_path = '/usr/local/google/home/deeplab_edge/cityscapes/leftImg8bit/test'
out_dir = '/usr/local/google/home/deeplab_edge/representative_data/img_size_256'
img_resize = (256, 256)
image_dirs  = os.listdir(image_dir_path)

for dir in image_dirs:
    print(dir)
    for img in os.listdir(os.path.join(image_dir_path, dir)):
        img_path = os.path.join(image_dir_path, dir, img)
        im = Image.open(img_path)
        resized_im = im.resize(img_resize)
        resized_im.save(os.path.join(out_dir, img.replace('.png', f"{img_resize[0]}x{img_resize[1]}.png")))
