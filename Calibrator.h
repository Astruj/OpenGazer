#pragma once

#include "ImageWindow.h"
#include "Component.h"

class Calibrator: public Component {
public:
	Calibrator(const std::vector<cv::Point> &points);
	~Calibrator();
	
	// Main processing function
	void process();
	
	// Calibration API functions (for future use)
	void start();
	void pointStart();
	void pointEnd();
	void abortCalibration();
	void calibrationEnded();
	void draw();
	
	bool isActive();
	cv::Point getActivePoint();
	int getPointNumber();
	int getPointFrameNo();
	bool isLastFrame();
	bool isLastPoint();
	bool isFinished();
	bool shouldStartNextPoint();
	
	bool needRecalibration;
	
private:
	int _frameNumber;
	std::vector<cv::Point> _points;
	ImageWindow _window;
	cv::Mat _screenImage;
	cv::Mat _targetImage;
};


