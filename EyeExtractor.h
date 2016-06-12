#pragma once

#include "BlinkDetector.h"
#include "FeatureDetector.h"
#include "Component.h"
#include "PointTracker.h"
#include "FacePoseEstimator.h"

class EyeExtractor: public Component {
public:
	static const int eyeDX;
	static const int eyeDY;
	static const cv::Size eyeSize;

	boost::scoped_ptr<FeatureDetector> averageEye;
	boost::scoped_ptr<FeatureDetector> averageEyeLeft;

	cv::Mat eyeGrey, eyeFloat, eyeImage;
	cv::Mat eyeGreyLeft, eyeFloatLeft, eyeImageLeft;

	EyeExtractor(bool fromGroundTruth=false);
	~EyeExtractor();
	void process();
	bool isBlinking();
	bool hasValidSample();
	void draw();

	void start();
	void pointStart();
	void pointEnd();
	void abortCalibration();
	void calibrationEnded();

private:
	BlinkDetector _blinkDetector;
	BlinkDetector _blinkDetectorLeft;
	bool _isBlinking;
	bool _fromGroundTruth;
    PointTracker *_pointTracker;
    FacePoseEstimator *_facePoseEstimator;

	void extractEyes(const cv::Mat originalImage);
	void extractRegion(const cv::Mat originalImage, cv::Point2f imageCoords[3], cv::Point2f extractedCoords[3],
						cv::Mat &extractedColor, cv::Mat &extractedGrey, cv::Mat &extractedFloat);
	//void extractEye(const cv::Mat originalImage);
	//void extractEyeLeft(const cv::Mat originalImage);
};
