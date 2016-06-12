#include <opencv2/highgui/highgui.hpp>
#include <fstream>
#include <boost/lexical_cast.hpp>


#include "EyeExtractor.h"
#include "Application.h"
#include "utils.h"

const int EyeExtractor::eyeDX = 64;
const int EyeExtractor::eyeDY = 32;
const cv::Size EyeExtractor::eyeSize = cv::Size(eyeDX * 2, eyeDY * 2);

EyeExtractor::EyeExtractor(bool fromGroundTruth):
	_pointTracker(NULL),
	_facePoseEstimator(NULL),

	eyeGrey(eyeSize, CV_8UC1),
	eyeFloat(eyeSize, CV_32FC1),

 	// ONUR DUPLICATED CODE FOR LEFT EYE
	eyeGreyLeft(eyeSize, CV_8UC1),
	eyeFloatLeft(eyeSize, CV_32FC1),

	eyeImage(eyeSize, CV_8UC3),
	eyeImageLeft(eyeSize, CV_8UC3),

	_isBlinking(false),
	_fromGroundTruth(fromGroundTruth)
{
}

EyeExtractor::~EyeExtractor() {}

void EyeExtractor::process() {
    static int imageNo = 1;

    // Initialize components
    if(_pointTracker == NULL) {
        _pointTracker = (PointTracker*) Application::getComponent("PointTracker");
        _facePoseEstimator = (FacePoseEstimator*) Application::getComponent("FacePoseEstimator");
    }

    if(Application::Signals::initiateCalibrationFrameNo == Application::Components::videoInput->frameCount) {
		start();
	}

    if (Application::Data::isTrackingSuccessful) {
    	cv::Mat frame = Application::Components::videoInput->frame;

		if(_fromGroundTruth) {
    		char file_detect_eye[100];
			sprintf(file_detect_eye, "/media/onur/DATA/1 Data/ground_truth/frames_gt/img_%d.png", Application::Components::videoInput->frameCount-1);
			cv::Mat image = cv::imread(file_detect_eye);
			frame = image;
		}

        
    	// Extract eye images using point tracker results
		extractEyes(frame);
        
        //cv::imwrite("eye_" + boost::lexical_cast<std::string>(imageNo) + ".png", eyeImage);
        imageNo++;
        
		// Blink detection
		_blinkDetector.update(eyeFloat);
		_blinkDetectorLeft.update(eyeFloatLeft);

		_isBlinking = (_blinkDetector.isBlinking || _blinkDetectorLeft.isBlinking);
                
        //if(_isBlinking) {
        //    std::cout << "BLINK!!!" << std::endl;
        //}
        
		// If calibration is active, collect eye image samples, calculate averages and train the system
		if(Application::Components::calibrator->isActive()) {
			if (Application::Components::calibrator->shouldStartNextPoint()) {
				// Switch to next calibration point
				pointStart();
			} else if (hasValidSample()) {
				// Add the valid training samples
				averageEye->addSample(&eyeFloat);
				averageEyeLeft->addSample(&eyeFloatLeft);

				// TODO MOVE ADD SAMPLES TO NN CODE
				//Application::Components::gazeTracker->addSampleToNN(Application::Components::calibrator->getActivePoint(), eyeFloat.get(), eyeGrey.get());
				//Application::Components::gazeTracker->addSampleToNNLeft(Application::Components::calibrator->getActivePoint(), eyeFloatLeft.get(), eyeGreyLeft.get());

				if (Application::Components::calibrator->getPointFrameNo() == Application::dwelltimeParameter - 1) {
					pointEnd();
				}
			}
		}
	}
}

bool EyeExtractor::isBlinking() {
	return _isBlinking;
}

void EyeExtractor::extractEyes(const cv::Mat originalImage) {
	static cv::Point2f imageCoords[4], imageCoordsLeft[4], extractedCoords[4];
	
	// If the face pose estimator is active, get the image coordinates from there
    if(_facePoseEstimator->isActive()) {
		imageCoords[0] = _facePoseEstimator->eyeRectangleCorners[0];
		imageCoords[1] = _facePoseEstimator->eyeRectangleCorners[1];
		imageCoords[2] = _facePoseEstimator->eyeRectangleCorners[2];
		imageCoords[3] = _facePoseEstimator->eyeRectangleCorners[3];
		
		imageCoordsLeft[0] = _facePoseEstimator->eyeRectangleCornersLeft[0];
		imageCoordsLeft[1] = _facePoseEstimator->eyeRectangleCornersLeft[1];
		imageCoordsLeft[2] = _facePoseEstimator->eyeRectangleCornersLeft[2];
		imageCoordsLeft[3] = _facePoseEstimator->eyeRectangleCornersLeft[3];
	}
	// Else, calculate them using the point tracker output
	else {
        cv::Point2f rightEyeCorner = _pointTracker->currentPoints[PointTracker::eyePoint1];
        cv::Point2f leftEyeCorner  = _pointTracker->currentPoints[PointTracker::eyePoint2];
		cv::Point2f rightToLeftVector = cv::Point2f(leftEyeCorner.x - rightEyeCorner.x, leftEyeCorner.y - rightEyeCorner.y);
		
		double dx = fabs(rightEyeCorner.x - leftEyeCorner.x);	// Horizontal distance between eye corners in image
		double imageEyeHeight = 0.17 * dx;	// %17 percent of horizontal distance (used as height of extracted eyes)
		double rollAngle = atan((leftEyeCorner.y-rightEyeCorner.y)/(leftEyeCorner.x - rightEyeCorner.x));
		
		// Calculate the first two points (rightmost)
		imageCoords[0] = cv::Point2f(rightEyeCorner.x + sin(rollAngle)*imageEyeHeight/2, rightEyeCorner.y - cos(rollAngle)*imageEyeHeight/2);	// Top right
		imageCoords[1] = cv::Point2f(rightEyeCorner.x - sin(rollAngle)*imageEyeHeight/2, rightEyeCorner.y + cos(rollAngle)*imageEyeHeight/2);	// Bottom right
		
		// The rest 6 can be calculated using the rightToLeft vector and previous points
		imageCoords[2] = cv::Point2f(imageCoords[1].x + rightToLeftVector.x*0.34, imageCoords[1].y + rightToLeftVector.y*0.34);	// Bottom left
		imageCoords[3] = cv::Point2f(imageCoords[0].x + rightToLeftVector.x*0.34, imageCoords[0].y + rightToLeftVector.y*0.34);	// Top left
		
		imageCoordsLeft[0] = cv::Point2f(imageCoords[0].x + rightToLeftVector.x*0.66, imageCoords[0].y + rightToLeftVector.y*0.66);	// Top right
		imageCoordsLeft[1] = cv::Point2f(imageCoords[1].x + rightToLeftVector.x*0.66, imageCoords[1].y + rightToLeftVector.y*0.66);	// Bottom right
		imageCoordsLeft[2] = cv::Point2f(imageCoords[1].x + rightToLeftVector.x*1.00, imageCoords[1].y + rightToLeftVector.y*1.00);	// Bottom left
		imageCoordsLeft[3] = cv::Point2f(imageCoords[0].x + rightToLeftVector.x*1.00, imageCoords[0].y + rightToLeftVector.y*1.00);	// Top left
	}
	
	// Calculate the corresponding pixel positions in the extracted images
	extractedCoords[0] = cv::Point2f(0, 0);
	extractedCoords[1] = cv::Point2f(0, eyeSize.height-1);
	extractedCoords[2] = cv::Point2f(eyeSize.width-1, eyeSize.height-1);
	extractedCoords[3] = cv::Point2f(eyeSize.width-1, 0);
	
	// Extract both eyes
	extractRegion(originalImage, imageCoords, extractedCoords, eyeImage, eyeGrey, eyeFloat);
	extractRegion(originalImage, imageCoordsLeft, extractedCoords, eyeImageLeft, eyeGreyLeft, eyeFloatLeft);
}

// Calculates the affine transformation between the imageCoords and extractedCoords,
// and uses this transformation to extract the eye regions from the given image
// Saves the color, grey and float type representations of the extracted image in the passed parameters
void EyeExtractor::extractRegion(const cv::Mat originalImage, cv::Point2f imageCoords[3], cv::Point2f extractedCoords[3],
									cv::Mat &extractedColor, cv::Mat &extractedGrey, cv::Mat &extractedFloat) {
	// 
	// Calculate the transformation matrix between two sets of points, and use it to extract eye image
	cv::Mat transform = cv::getAffineTransform(imageCoords, extractedCoords);
    cv::warpAffine(originalImage, extractedColor, transform, eyeSize);
    cv::cvtColor(extractedColor, extractedGrey, CV_BGR2GRAY);

	// Apply blurring and normalization
	//Utils::normalizeGrayScaleImage(extractedGrey.get(), 127, 50);	// TODO ONUR UNCOMMENT
	extractedGrey.convertTo(extractedFloat, CV_32FC1);
    //cv::GaussianBlur(extractedFloat, extractedFloat, cv::Size(3,3), 0);
	cv::equalizeHist(extractedGrey, extractedGrey);
}

void EyeExtractor::draw() {
    if (!Application::Data::isTrackingSuccessful)
		return;

	cv::Mat image = Application::Components::videoInput->debugFrame;
	int eyeDX = eyeSize.width;
	int eyeDY = eyeSize.height;

	int baseX = 0;
    int baseY = 0;

	// Draw the extacted eye images on top left of debug screen
    cv::cvtColor(eyeGrey, image(cv::Rect(baseX, baseY, eyeDX, eyeDY)), CV_GRAY2BGR);
    cv::cvtColor(eyeGreyLeft, image(cv::Rect(baseX + 140, baseY, eyeDX, eyeDY)), CV_GRAY2BGR);
}

// Prepares the eye extractor for calibration
void EyeExtractor::start() {
	pointStart();
}

// When a new calibration point is starting, prepare the average eye accumulators
void EyeExtractor::pointStart() {
	averageEye.reset(new FeatureDetector(EyeExtractor::eyeSize));
	averageEyeLeft.reset(new FeatureDetector(EyeExtractor::eyeSize));
}

void EyeExtractor::pointEnd() {}
void EyeExtractor::abortCalibration() {}
void EyeExtractor::calibrationEnded() {}

bool EyeExtractor::hasValidSample() {
	return Application::Components::calibrator->getPointFrameNo() >= 11 && !isBlinking();
}
