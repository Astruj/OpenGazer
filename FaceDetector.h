#pragma once

#include <opencv2/objdetect/objdetect.hpp>

class FaceDetector {
public:
	static FaceDetector faceDetector;

	FaceDetector();
	~FaceDetector();
	cv::Rect detect(const cv::Mat img);

private:
	cv::CascadeClassifier _cascade;
};


