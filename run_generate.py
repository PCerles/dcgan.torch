import os
#checkpoints = [10, 20, 30, 40, 50, 60, 70]
checkpoints =[25,50,75,100,110,120,130,140]
for c in checkpoints:
	check = "checkpoints_normal/experiment1_" + str(c) + '_net_G.t7'
	im_name = "images_normal/check" + str(c)
	os.system('gpu=1 net=' + check +  ' ' + "name=" + str(im_name) + " " + 'input_caption=bus' + ' th generate_copy.lua')

