''' Script for saving to mp4 '''

if __name__ == '__main__':

	import os
	import sys
	import ffmpeg

	path = os.path.abspath(sys.argv[1]) 
	fps = int(sys.argv[2])
	print('Converting...')
	ffmpeg.input(path + '/*.png', pattern_type='glob', framerate=fps).output(path + '/full.mp4').run()
