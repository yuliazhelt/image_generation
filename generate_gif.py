import glob
from PIL import Image


def make_gif(folder, run_name):
    frames = [Image.open(image) for image in sorted(glob.glob(f"{folder}/{run_name}/*.png"))]
    frame_one = frames[0]
    frame_one.save(f"{run_name}.gif", format="GIF", append_images=frames,
               save_all=True, duration=100, loop=0)
    

if __name__ == "__main__":
    make_gif(folder="/home/ubuntu/image_generation/saved/images", run_name="run_18.12-17:51:14")