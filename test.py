pixel_counter=-1
gaze_data= [[] for i in range(9)]
pixel_values=[]
with open("gaze.txt", "r") as f:
	for gaze in f.readlines():		
		# gaze = f.readline()
		if gaze.startswith('('):
			pixel_counter+=1
			list_gaze = eval(gaze)
			pixel_values.append(list_gaze)
		else:
			gaze_data[pixel_counter].append(eval(gaze))
				# if type(list(gaze))== list:
			

print(pixel_counter)
# print(gaze_data)
theta_list = [[] for i in range(9)]
phi_list = [[] for i in range(9)]

for x in range(9):
	for y in range(len(gaze_data[x])):
		theta_list[x].append(gaze_data[x][y][0])
		phi_list[x].append(gaze_data[x][y][1])


print(theta_list)

import numpy as np
means_theta = []
means_phi = []
for x in range(9):
	means_theta.append(np.mean(theta_list[x]))
	means_phi.append(np.mean(phi_list[x]))	

for x in range(9):
	print(means_theta[x],means_phi[x])