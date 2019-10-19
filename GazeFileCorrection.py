import datetime
outputFile = open("gaze_points.csv", "r")
finalOutput = open("FinalGaze.csv", "w+")
result = []
for lines in outputFile:
	if lines == "Calibration ended\n":
		result = []
		continue
	result.append(lines)

# finalOutput.write("time,x_gaze,y_gaze\n")
for points in result:
	finalOutput.write(points)
outputFile.close()
finalOutput.close()
