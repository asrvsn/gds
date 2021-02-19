''' Script for saving to GIF '''

if __name__ == '__main__':

	import os
	import sys
	import glob
	from PIL import Image

	path = os.path.abspath(sys.argv[1]) 
	fullpath = path + '/full.gif'
	fps = int(sys.argv[2])
	imgs = [Image.open(f) for f in sorted(glob.glob(path + '/*.png'), key=lambda f: int(f.split('/')[-1][:-4]))]
	duration = int(1000 / fps)
	imgs[0].save(fp=fullpath, format='GIF', append_images=imgs[1:], save_all=True, duration=duration, loop=0)
	os.system(f'gifsicle -b -O2 "{fullpath}"')