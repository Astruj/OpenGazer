/*
 * Face pose estimation algorithm by Kazemi, V. and Sullivan, J.
 *
 *      One Millisecond Face Alignment with an Ensemble of Regression Trees. Vahid Kazemi and Josephine Sullivan, CVPR 2014
 *
 * Implementation adapted from dlib examples
 *
 *      https://github.com/davisking/dlib/blob/master/examples/face_landmark_detection_ex.cpp
 */
#include "FacePoseEstimator.h"

#include "Application.h"
#include "utils.h"
#include "HiResTimer.h"

#include <iostream>
#include <boost/lexical_cast.hpp>

#include <opencv2/video/tracking.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>

// TODO UPDATE FOCAL LENGTH (FROM CAMERA CALIBRATION)
#define FOCAL_LENGTH 685

FacePoseEstimator::FacePoseEstimator() :
    isFaceInitialized(false),
    _isActive(false)
{
    // A multiplier to correct for the generic head model used below
    // With this, the distances are calculated more accurately (validated using the dataset from
    // our previous work: http://mv.cvc.uab.es/projects/eye-tracker/cvceyetrackerdb )
    float focalLengthMultiplier = 0.87;

    // Load the pretrained shape predictor
    // The file can be downloaded from
    // http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
    // TODO Download file in the configuration script
    dlib::deserialize("./data/shape_predictor_68_face_landmarks.dat") >> _faceShapePredictor;
    
    _faceDetector = dlib::get_frontal_face_detector();

    // Prepare parameters used for head pose angle calculations
    cv::Mat frame = Application::Components::videoInput->frame;

    // Prepare the projection matrix
    cv::Mat projectionMat = cv::Mat::zeros(3,3,CV_32F);
    _projection = projectionMat;
    _projection(0,0) = FOCAL_LENGTH*focalLengthMultiplier;
    _projection(1,1) = FOCAL_LENGTH*focalLengthMultiplier;
    _projection(0,2) = frame.size().width/2;
    _projection(1,2) = frame.size().height/2;
    _projection(2,2) = 1;

    std::cout << "Frame size is " << frame.size().width << "x" << frame.size().height << std::endl;

    // Prepare the array with points facial feature positions relative to "sellion"
    // Numbers based on anthropometric properties for male adult (see .h for more info)
    _headPoints.push_back(P3D_SELLION);
    _headPoints.push_back(P3D_RIGHT_EYE);
    _headPoints.push_back(P3D_LEFT_EYE);
    _headPoints.push_back(P3D_RIGHT_EAR);
    _headPoints.push_back(P3D_LEFT_EAR);
    _headPoints.push_back(P3D_MENTON);
    _headPoints.push_back(P3D_NOSE);
    _headPoints.push_back(P3D_STOMMION);

    _eyeRegionPoints.push_back(P3D_RIGHT_EYE);
    _eyeRegionPoints.push_back(P3D_LEFT_EYE);
    _eyeRegionPoints.push_back(P3D_EYE_REGION_TOP_RIGHT);
    _eyeRegionPoints.push_back(P3D_EYE_REGION_TOP_LEFT);
    _eyeRegionPoints.push_back(P3D_EYE_REGION_BOTTOM_RIGHT);
    _eyeRegionPoints.push_back(P3D_EYE_REGION_BOTTOM_LEFT);

    // Allocate the image structures used in the face tracker
    allocateFaceTracker();
}


void FacePoseEstimator::process() {
    static int counter = 0;
    _isActive = true;
    
    // Create a wrapper dlib structure around the camera framepredictor
    dlib::assign_image(_reducedSizeVideoFrame, dlib::cv_image<dlib::bgr_pixel>(Application::Components::videoInput->reducedSizeFrame));
    dlib::assign_image(_videoFrame, dlib::cv_image<dlib::bgr_pixel>(Application::Components::videoInput->frame));

    /*
    // Every N=10 frames, refresh the face detection
    if(counter % 10 == 0) {
        // Try to detect the face
        if(!detectFace()) {
            // If face is not detected succesfully
            // decrement counter to try detection in the next frame
            //counter--;
            
            // Track the face using the previous detections
            trackFace();
            
            std::cout << "NO faces found!" << std::endl;
        }
    }
    else {
        trackFace();
    }
    */
    
    //counter++;
    detectFace();

    if(isFaceInitialized) {
        // Use the face detection (or last face tracking results) to predict face shape (landmark positions)
        dlib::full_object_detection shape = _faceShapePredictor(_videoFrame, dlib::rectangle(faceRectangle.x,
                                                                                           faceRectangle.y,
                                                                                           faceRectangle.x+faceRectangle.width-1,
                                                                                           faceRectangle.y+faceRectangle.height-1));

        // Fill the facialLandmarks structure with landmark positions
        facialLandmarks.clear();

        for(int i=0; i<shape.num_parts(); i++) {
            facialLandmarks.push_back(cv::Point(shape.part(i).x(), shape.part(i).y()));
        }

        Application::Data::isTrackingSuccessful = true;
        calculatePose();
    }
}

void FacePoseEstimator::calculatePose() {
    // Prepare the array with corresponding point positions detected on the image
    std::vector<cv::Point2f> detectedPoints;

    detectedPoints.push_back(facialLandmarks[SELLION]);
    detectedPoints.push_back(facialLandmarks[RIGHT_EYE]);
    detectedPoints.push_back(facialLandmarks[LEFT_EYE]);
    detectedPoints.push_back(facialLandmarks[RIGHT_SIDE]);
    detectedPoints.push_back(facialLandmarks[LEFT_SIDE]);
    detectedPoints.push_back(facialLandmarks[MENTON]);
    detectedPoints.push_back(facialLandmarks[NOSE]);


    cv::Point2f stomion = (facialLandmarks[MOUTH_CENTER_TOP] + facialLandmarks[MOUTH_CENTER_BOTTOM]) * 0.5;
    detectedPoints.push_back(stomion);

    // Find the 3D pose of our headprojection
    cv::solvePnP(_headPoints, detectedPoints,
            _projection, cv::noArray(),
            _rvec, _tvec, false,
            0);    // hardcoded value for cv::SOLVEPNP_ITERATIVE or cv::ITERATIVE (to fix problems with different versions of OpenCV)

    cv::Matx33d rotation;
    cv::Rodrigues(_rvec, rotation);

    headPose(0,0) = rotation(0,0);
    headPose(0,1) = rotation(0,1);
    headPose(0,2) = rotation(0,2);
    headPose(0,3) = _tvec.at<double>(0)/1000;
    
    headPose(1,0) = rotation(1,0);
    headPose(1,1) = rotation(1,1);
    headPose(1,2) = rotation(1,2);
    headPose(1,3) = _tvec.at<double>(1)/1000;
    
    headPose(2,0) = rotation(2,0);
    headPose(2,1) = rotation(2,1);
    headPose(2,2) = rotation(2,2);
    headPose(2,3) = _tvec.at<double>(2)/1000;
    
    headPose(3,0) = 0;
    headPose(3,1) = 0;
    headPose(3,2) = 0;
    headPose(3,3) = 1;
    
    /*
    headPose = {
        rotation(0,0),    rotation(0,1),    rotation(0,2),    _tvec.at<double>(0)/1000,
        rotation(1,0),    rotation(1,1),    rotation(1,2),    _tvec.at<double>(1)/1000,
        rotation(2,0),    rotation(2,1),    rotation(2,2),    _tvec.at<double>(2)/1000,
                    0,                0,                0,                     1
    };*/
}

bool FacePoseEstimator::detectFace() {    
    static HiResTimer timer;
    
    // Detect the faces in the image
    //timer.start();
    std::vector<dlib::rectangle> faceDetections = _faceDetector(_reducedSizeVideoFrame);
    //timer.stop();
    //std::cout << timer.getElapsedTime() << " ms passed for face detection" << std::endl;
    
    if(faceDetections.size() == 0) {
        return false;
    }
    
    isFaceInitialized = true;
    
    // Use the first detection to reset the Camshift tracker
    dlib::rectangle detection = faceDetections[0];
    
    resetFaceTracking(cv::Rect(detection.left()*2, detection.top()*2, detection.width()*2, detection.height()*2));
    
    return true;
}

void FacePoseEstimator::allocateFaceTracker() {
    cv::Mat frame = Application::Components::videoInput->frame;
    
    // Initialize the image structures
    _faceTracker.hsv.create(frame.size(), CV_8UC3);
    _faceTracker.hue.create(frame.size(), CV_8UC1);
    _faceTracker.mask.create(frame.size(), CV_8UC1);
    _faceTracker.prob.create(frame.size(), CV_8UC1);
}

void FacePoseEstimator::extractHueAndMask() {
    cv::Mat frame = Application::Components::videoInput->frame;
    int vmin = 65, vmax = 256, smin = 55;
    int ch[] = {0, 0};
    
    // Convert color to HSV
    cv::cvtColor(frame, _faceTracker.hsv, CV_BGR2HSV);
    
    // Threshold the HSV image and create the mask
    cv::inRange(_faceTracker.hsv, cv::Scalar(0, smin, vmin), cv::Scalar(180, 256, vmax), _faceTracker.mask);
    
    // Extract the first channel of HSV and save in hueprojection
    cv::mixChannels(&_faceTracker.hsv, 1, &_faceTracker.hue, 1, ch, 1);
}

void FacePoseEstimator::resetFaceTracking(cv::Rect faceDetection) {
    cv::Mat frame = Application::Components::videoInput->frame;
    int hsize = 30;
    float hranges[] = {0,180};
    const float* phranges = hranges;
    
    extractHueAndMask();
    
    // Calculate the histogram features that describe the face
    cv::Mat roi(_faceTracker.hue, faceDetection), maskroi(_faceTracker.mask, faceDetection);
    cv::calcHist(&roi, 1, 0, maskroi, _faceTracker.hist, 1, &hsize, &phranges);
    cv::normalize(_faceTracker.hist, _faceTracker.hist, 0, 255, cv::NORM_MINMAX);
    
    faceRectangle = faceDetection;
}

void FacePoseEstimator::trackFace() {
	if (!isFaceInitialized)
		return;
    
    float hranges[] = {0,180};
    const float* phranges = hranges;
    
    extractHueAndMask();
    
    // Track the face using the Camshift algorithm
    cv::calcBackProject(&_faceTracker.hue, 1, 0, _faceTracker.hist, _faceTracker.prob, &phranges);
    _faceTracker.prob &= _faceTracker.mask;
    faceRotatedRectangle = cv::CamShift(_faceTracker.prob,
                                        faceRectangle,
                                        cv::TermCriteria( cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 10, 1 ));
    
    // The rotated rectangle results is stored in faceRotatedRectangle
    // faceRectangle contains its bounding box for easy use
    //faceRectangle = faceRotatedRectangle.boundingRect();
    
    /*
    int cols = faceTracker.prob.cols, 
        rows = faceTracker.prob.rows, 
        r = (MIN(cols, rows) + 5)/6;
    
    faceRectangle = cv::Rect(faceRectangle.x - r, faceRectangle.y - r,
                       faceRectangle.x + r, faceRectangle.y + r) &
                    cv::Rect(0, 0, cols, rows);
                    */
}

bool FacePoseEstimator::isActive() {
    return _isActive;
}

void FacePoseEstimator::draw() {
	if (!isFaceInitialized)
		return;

    cv::Mat image = Application::Components::videoInput->debugFrame;
/*
    std::vector<cv::Point2f> detectedPoints;

    detectedPoints.push_back(facialLandmarks[SELLION]);
    detectedPoints.push_back(facialLandmarks[RIGHT_EYE]);
    detectedPoints.push_back(facialLandmarks[LEFT_EYE]);
    detectedPoints.push_back(facialLandmarks[RIGHT_SIDE]);
    detectedPoints.push_back(facialLandmarks[LEFT_SIDE]);
    detectedPoints.push_back(facialLandmarks[MENTON]);
    detectedPoints.push_back(facialLandmarks[NOSE]);


    // Reproject the eye region boundary points from the generic head model points to the image, and draw these on the debug image
    std::vector<cv::Point2f> reprojectedPoints;

    cv::projectPoints(headPoints, rvec, tvec, projection, cv::noArray(), reprojectedPoints);

    for (int i=0; i<reprojectedPoints.size(); i++) {

        cv::circle(image, Utils::mapFromCameraToDebugFrameCoordinates(reprojectedPoints[i]), 2, cv::Scalar(0,255,255), 2);
        cv::circle(image, Utils::mapFromCameraToDebugFrameCoordinates(detectedPoints[i]), 2, cv::Scalar(255,255,255), 2);
    }
*/

    // Trying to find eye corner points
    cv::circle(image, Utils::mapFromCameraToDebugFrameCoordinates(facialLandmarks[RIGHT_EYE]), 3, cv::Scalar(255,255,255), -1, 8, 0);
    cv::circle(image, Utils::mapFromCameraToDebugFrameCoordinates(facialLandmarks[LEFT_EYE]), 3, cv::Scalar(255,255,255), -1, 8, 0);


    /*
    cv::rectangle(image,
                  Utils::mapFromCameraToDebugFrameCoordinates(cv::Point(faceRectangle.x, faceRectangle.y)),
                  Utils::mapFromCameraToDebugFrameCoordinates(cv::Point(faceRectangle.x+faceRectangle.width, faceRectangle.y+faceRectangle.height)),
                  CV_RGB(0, 255, 0), 2, 8, 0);
*/
    // Prepare the axes for the head pose, project it to the image and draw on the debug image
    std::vector<cv::Point3f> axes;
    axes.push_back(cv::Point3f(0,0,0));
    axes.push_back(cv::Point3f(50,0,0));
    axes.push_back(cv::Point3f(0,50,0));
    axes.push_back(cv::Point3f(0,0,50));
    std::vector<cv::Point2f> projectedAxes;

    projectPoints(axes, _rvec, _tvec, _projection, cv::noArray(), projectedAxes);

    cv::line(image, Utils::mapFromCameraToDebugFrameCoordinates(projectedAxes[0]), Utils::mapFromCameraToDebugFrameCoordinates(projectedAxes[3]), cv::Scalar(255,0,0),2,CV_AA);
    cv::line(image, Utils::mapFromCameraToDebugFrameCoordinates(projectedAxes[0]), Utils::mapFromCameraToDebugFrameCoordinates(projectedAxes[2]), cv::Scalar(0,255,0),2,CV_AA);
    cv::line(image, Utils::mapFromCameraToDebugFrameCoordinates(projectedAxes[0]), Utils::mapFromCameraToDebugFrameCoordinates(projectedAxes[1]), cv::Scalar(0,0,255),2,CV_AA);

    cv::putText(image, "(" + boost::lexical_cast<std::string>(int(headPose(0,3) * 100)) + "cm, " + boost::lexical_cast<std::string>(int(headPose(1,3) * 100)) + "cm, " + boost::lexical_cast<std::string>(int(headPose(2,3) * 100)) + "cm)", Utils::mapFromCameraToDebugFrameCoordinates(facialLandmarks[SELLION]), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0,0,255),2, 8);


}
