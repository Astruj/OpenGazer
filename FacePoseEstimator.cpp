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

#include <fstream>
#include <iostream>
#include <cmath>
#include <boost/lexical_cast.hpp>
#include <boost/filesystem.hpp>

// Serialization related libraries
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/vector.hpp>

#include <opencv2/video/tracking.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>

// TODO UPDATE FOCAL LENGTH (FROM CAMERA CALIBRATION)
#define FOCAL_LENGTH 685
#define PIXEL_WIDTH 0.2768  // Pixel width in mm, used in correcting for the headpose change 

FacePoseEstimator::FacePoseEstimator() :
    isFaceFound(false),
    _isActive(false),
    _lastSampleFrame(50)
{
    // Load the pretrained shape predictor
    // The file can be downloaded from
    // http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
    // and the configure.sh script should download it automatically to the correct folder
    dlib::deserialize("./data/shape_predictor_68_face_landmarks.dat") >> _faceShapePredictor;
    
    _faceDetector = dlib::get_frontal_face_detector();

    // Prepare parameters used for head pose angle calculations
    cv::Mat frame = Application::Components::videoInput->frame;

    // Prepare the projection matrix
    cv::Mat projectionMat = cv::Mat::zeros(3,3,CV_32F);
    _projection = projectionMat;
    _projection(0,0) = FOCAL_LENGTH;
    _projection(1,1) = FOCAL_LENGTH;
    _projection(0,2) = frame.size().width/2;
    _projection(1,2) = frame.size().height/2;
    _projection(2,2) = 1;

    //std::cout << "Frame size is " << frame.size().width << "x" << frame.size().height << std::endl;

    // Fill the personal parameters related to face shape with default values (filled with 1s) 
    for(int i=0; i<NUM_PERSONAL_PARAMETERS; i++)
        _parameters.push_back(1.0);

    // Try to load (overwriting) the personal parameters if a previously saved file exists
    loadParameters();
    
    // Mark the allowed max. deviation (in percent) for each parameter
    _parameterAllowedDeviations.push_back(0.5);    // Eye depth ??
    _parameterAllowedDeviations.push_back(0.1);    // Eye separation
    _parameterAllowedDeviations.push_back(0.5);    // Nose depth
    _parameterAllowedDeviations.push_back(0.2);    // Nose length
    _parameterAllowedDeviations.push_back(0.1);    // Menton length
    
    _parameterAllowedDeviations.push_back(0.5);    // Ear depth
    _parameterAllowedDeviations.push_back(0.2);    // Ear separation
    _parameterAllowedDeviations.push_back(0.2);    // Stommion depth
    _parameterAllowedDeviations.push_back(0.2);    // Stommion length
    
    std::cout << "LOADED PARAMETERS: " << std::endl;
    for(int i=0; i<NUM_PERSONAL_PARAMETERS; i++)
        std::cout << i << " " << _parameters[i] << std::endl;
    
    // Calculate the head model with the default parameters
    calculateHeadModel(_parameters, _headModel);
    calculateHeadModel(_parameters, _genericHeadModel);
    
    // Calculate the corner points defining the eye region rectangles
    calculateEyeRectangleCorners();
    
    _sampleCount = 0;
    _iterationNumber = 0;
    
    rightEye = cv::Point2f(-1, -1);
    leftEye = cv::Point2f(-1, -1);
}

void FacePoseEstimator::process() {
    static int counter = 0;
    _isActive = true;
    
    // Create a wrapper dlib structure around the camera framepredictor
    dlib::assign_image(_reducedSizeVideoFrame, dlib::cv_image<dlib::bgr_pixel>(Application::Components::videoInput->reducedSizeFrame));
    dlib::assign_image(_videoFrame, dlib::cv_image<dlib::bgr_pixel>(Application::Components::videoInput->frame));
    //dlib::pyramid_up(_reducedSizeVideoFrame);
    
    // If calibration is started, cancel learning
    if(Application::status == Application::STATUS_CALIBRATING) {
        Application::Settings::faceStructureLearning = false;
    }
    
    isFaceFound = detectFace();

    if(isFaceFound) {
        // Use the face detection to predict face shape (landmark positions)
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
        estimateFacePose();
        
        if(Application::Settings::faceStructureLearning) {
            // If there was a mouse click on the debug window,
            // or if the current head pose is different enough (compared to previously added samples),
            // Add another sample for the calculation of the face structure
            if(Application::Signals::addFaceSample || shouldAddFaceSample()) {
                addFaceSample();
                Application::Signals::addFaceSample = false;
            }
            
            // If there were enough samples, do 5 iterations of coordinate descent 
            if(_sampleCount > 4) {
                HiResTimer timer;
                
                timer.start();
                for(int iteration=0; iteration<5; iteration++)
                    coordinateDescentIteration();
                timer.stop();
                
                // Save the updated personal parameters
                saveParameters();
            }
        }
          
        // Calculate the corner points defining the eye region rectangles
        calculateEyeRectangleCorners();
        
        // Update the estimations for left and right eye corners
        std::vector<cv::Point2f> eyeCornerProjections;
        projectPoints(getUsedHeadModel(), _rvec, _tvec, eyeCornerProjections);
        
        // Update the 2D eye rectangle corners
        eyeRectangleCorners.clear();
        eyeRectangleCornersLeft.clear();
        projectPoints(_eyeRectangleCorners3d, _rvec, _tvec, eyeRectangleCorners);
        projectPoints(_eyeRectangleCorners3dLeft, _rvec, _tvec, eyeRectangleCornersLeft);
        
        // Smooth the estimation a little bit
        double alpha = 0.3;//    DISABLED SMOOTHING FOR NOW
        rightEye.x = (alpha)*rightEye.x + (1-alpha)*eyeCornerProjections[INDEX_RIGHT_EYE].x;
        rightEye.y = (alpha)*rightEye.y + (1-alpha)*eyeCornerProjections[INDEX_RIGHT_EYE].y;
        
        leftEye.x = (alpha)*leftEye.x + (1-alpha)*eyeCornerProjections[INDEX_LEFT_EYE].x;
        leftEye.y = (alpha)*leftEye.y + (1-alpha)*eyeCornerProjections[INDEX_LEFT_EYE].y;
        //std::cout << "Updating eye corner positions. Right eye position: (" << rightEye.x << ", " << rightEye.y << ")" << std::endl;
    }
}

void FacePoseEstimator::estimateFacePose() {
    // Prepare the array with corresponding point positions detected on the image
    std::vector<cv::Point2f> point2d;

    point2d.push_back(facialLandmarks[SELLION]);
    point2d.push_back(facialLandmarks[RIGHT_EYE]);
    point2d.push_back(facialLandmarks[LEFT_EYE]);
    point2d.push_back(facialLandmarks[NOSE]);
    point2d.push_back(facialLandmarks[MENTON]);
    
#ifdef EXTENDED_FACE_MODEL
    point2d.push_back(facialLandmarks[RIGHT_SIDE]);
    point2d.push_back(facialLandmarks[LEFT_SIDE]);

    cv::Point2f stommion = (facialLandmarks[MOUTH_CENTER_TOP] + facialLandmarks[MOUTH_CENTER_BOTTOM]) * 0.5;
    point2d.push_back(stommion);
#endif

    // Find the 3D pose of our headprojection
    estimateFacePoseFrom2DPoints(point2d, _rvec, _tvec, false);
}

bool FacePoseEstimator::detectFace() {    
    static HiResTimer timer;
    
    // Detect the faces in the image
    std::vector<dlib::rectangle> faceDetections = _faceDetector(_reducedSizeVideoFrame); //_faceDetector(_videoFrame);
    
    if(faceDetections.size() == 0) {
        return false;
    }
    
    // Save the first detection as the face rectangle (multiply coords. by 2 because of scaled image)
    dlib::rectangle detection = faceDetections[0];
    faceRectangle = cv::Rect(detection.left()*2, detection.top()*2, detection.width()*2, detection.height()*2); //cv::Rect(detection.left(), detection.top(), detection.width(), detection.height());
    
    return true;
}

bool FacePoseEstimator::shouldAddFaceSample() {
    double minDifference = 10000;
    
    // If there a sample was added recently, wait a little (10 frames)
    if(Application::Components::videoInput->frameCount < _lastSampleFrame+10) {
        return false;
    }
    
    // If we already have enough samples, do not add more
    if(_sampleHeadPoseAngles.size() >= 40) {
        return false;
    }
    
    cv::Vec3d currentPoseAngles = getEulerAngles(_rvec);
    //std::cout << "Current pose: " << currentPoseAngles << std::endl << "--------------" << std::endl;
        
    for(int i=0; i<_sampleHeadPoseAngles.size(); i++) {
        double difference = cv::norm(currentPoseAngles - _sampleHeadPoseAngles[i]);
        //std::cout << "Comparing to " << _sampleHeadPoseAngles[i] << ", norm is: " << difference << std::endl;
        
        minDifference = std::min(difference, minDifference);
    }
    
    //std::cout << "Min norm is: " << minDifference << std::endl << std::endl;
    
    // If the current head pose angle has large enough difference compared to previous samples, 
    // add a new sample
    if(minDifference > 0.15) {
        return true;
    }
    
    return false;
}

void FacePoseEstimator::addFaceSample() {
    _lastSampleFrame = Application::Components::videoInput->frameCount;
    // Add the facial landmark positions, rvec and tvec vectors to the corresponding vectors
    std::vector<cv::Point2f> *facePoints = new std::vector<cv::Point2f>();
    facePoints->push_back(cv::Point2f(facialLandmarks[SELLION].x,  facialLandmarks[SELLION].y));
    facePoints->push_back(cv::Point2f(facialLandmarks[RIGHT_EYE].x, facialLandmarks[RIGHT_EYE].y));
    facePoints->push_back(cv::Point2f(facialLandmarks[LEFT_EYE].x, facialLandmarks[LEFT_EYE].y));
    facePoints->push_back(cv::Point2f(facialLandmarks[NOSE].x,     facialLandmarks[NOSE].y));
    facePoints->push_back(cv::Point2f(facialLandmarks[MENTON].x,   facialLandmarks[MENTON].y));

#ifdef EXTENDED_FACE_MODEL    
    facePoints->push_back(cv::Point2f(facialLandmarks[RIGHT_SIDE].x,   facialLandmarks[RIGHT_SIDE].y));
    facePoints->push_back(cv::Point2f(facialLandmarks[LEFT_SIDE].x,    facialLandmarks[LEFT_SIDE].y));
    cv::Point2f stommion = (facialLandmarks[MOUTH_CENTER_TOP] + facialLandmarks[MOUTH_CENTER_BOTTOM]) * 0.5;
    facePoints->push_back(cv::Point2f(stommion.x, stommion.y));
#endif

    _sampleFacePoints.push_back(*facePoints);
    
    cv::Mat *tempRvec = new cv::Mat();
    _rvec.copyTo(*tempRvec);
    _sampleRvecs.push_back(*tempRvec);
    
    cv::Mat *tempTvec = new cv::Mat();
    _tvec.copyTo(*tempTvec);
    _sampleTvecs.push_back(*tempTvec);
    
    // Calculate the Euler angles for face pose (yaw, pitch, roll) and store in the _sampleHeadPoseAngles vector
    _sampleHeadPoseAngles.push_back(getEulerAngles(_rvec));
    
    std::cout << "Added face sample!!" << std::endl;
    
    _sampleCount++;
    
    // Reset coordinate descent iteration number
    _iterationNumber = 0;
}

bool FacePoseEstimator::isActive() {
    return _isActive;
}

void FacePoseEstimator::draw() {
	if (!isFaceFound)
		return;

    cv::Mat image = Application::Components::videoInput->debugFrame;

    std::vector<cv::Point2f> detectedPoints;

    detectedPoints.push_back(facialLandmarks[SELLION]);
    detectedPoints.push_back(facialLandmarks[RIGHT_EYE]);
    detectedPoints.push_back(facialLandmarks[LEFT_EYE]);
    detectedPoints.push_back(facialLandmarks[NOSE]);
    detectedPoints.push_back(facialLandmarks[MENTON]);

#ifdef EXTENDED_FACE_MODEL  
    detectedPoints.push_back(facialLandmarks[RIGHT_SIDE]);
    detectedPoints.push_back(facialLandmarks[LEFT_SIDE]);
    cv::Point2f stommion = (facialLandmarks[MOUTH_CENTER_TOP] + facialLandmarks[MOUTH_CENTER_BOTTOM]) * 0.5;
    detectedPoints.push_back(cv::Point2f(stommion.x, stommion.y));
#endif

    /*
    // Reproject the eye region boundary points from the generic head model points to the image, and draw these on the debug image
    std::vector<cv::Point2f> reprojectedPoints;
    projectPoints(getUsedHeadModel(), _rvec, _tvec, reprojectedPoints);

    for (int i=0; i<reprojectedPoints.size(); i++) {

        cv::circle(image, Utils::mapFromCameraToDebugFrameCoordinates(reprojectedPoints[i]), 2, cv::Scalar(0,255,255), 2);
        cv::circle(image, Utils::mapFromCameraToDebugFrameCoordinates(detectedPoints[i]), 2, cv::Scalar(255,255,255), 2);
    }
    */

    // Trying to find eye corner points
    //cv::circle(image, Utils::mapFromCameraToDebugFrameCoordinates(facialLandmarks[RIGHT_EYE]), 3, cv::Scalar(255,255,255), -1, 8, 0);
    //cv::circle(image, Utils::mapFromCameraToDebugFrameCoordinates(facialLandmarks[LEFT_EYE]), 3, cv::Scalar(255,255,255), -1, 8, 0);
    
    cv::circle(image, Utils::mapFromCameraToDebugFrameCoordinates(rightEye), 3, cv::Scalar(0,0,255), -1, 8, 0);
    cv::circle(image, Utils::mapFromCameraToDebugFrameCoordinates(leftEye), 3, cv::Scalar(0,0,255), -1, 8, 0);


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

    projectPoints(axes, _rvec, _tvec, projectedAxes);

    cv::line(image, Utils::mapFromCameraToDebugFrameCoordinates(projectedAxes[0]), Utils::mapFromCameraToDebugFrameCoordinates(projectedAxes[3]), cv::Scalar(255,0,0),2,CV_AA);
    cv::line(image, Utils::mapFromCameraToDebugFrameCoordinates(projectedAxes[0]), Utils::mapFromCameraToDebugFrameCoordinates(projectedAxes[2]), cv::Scalar(0,255,0),2,CV_AA);
    cv::line(image, Utils::mapFromCameraToDebugFrameCoordinates(projectedAxes[0]), Utils::mapFromCameraToDebugFrameCoordinates(projectedAxes[1]), cv::Scalar(0,0,255),2,CV_AA);

    cv::putText(image, "("  + boost::lexical_cast<std::string>(int(_tvec.at<double>(0)/10)) + "cm, " 
                            + boost::lexical_cast<std::string>(int(_tvec.at<double>(1)/10)) + "cm, " 
                            + boost::lexical_cast<std::string>(int(_tvec.at<double>(2)/10)) + "cm)", 
                Utils::mapFromCameraToDebugFrameCoordinates(facialLandmarks[SELLION]), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0,0,255),2, 8);

    if(Application::Signals::useGenericFaceModel) {
        cv::putText(image, "GENERIC MODEL", cv::Point(100, 450), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0,0,255),2, 8);
    }
    else {
        cv::putText(image, "CUSTOMIZED MODEL", cv::Point(100, 450), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0,255,0),2, 8);
    }
    
    // Write the number of samples collected so far on the debug frame
    cv::putText(image, "# SAMPLES: " + boost::lexical_cast<std::string>(_sampleHeadPoseAngles.size()), cv::Point(100, 500), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0,255,0),2, 8);
        
    cv::Vec3d rotationAngles = getEulerAngles(_rvec);
    
    cv::putText(image, "ROTATIONS: ("  + boost::lexical_cast<std::string>(int((180/M_PI)*rotationAngles[0])) + ", " 
                            + boost::lexical_cast<std::string>(int((180/M_PI)*rotationAngles[1])) + ", " 
                            + boost::lexical_cast<std::string>(int((180/M_PI)*rotationAngles[2])) + ")",
            cv::Point(100, 550), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0,255,0),2, 8);
    
    
    cv::Rect* geometry = Utils::getSecondMonitorGeometry();
    int monitorCenterX = geometry->width/2;
    int monitorCenterY = geometry->height/2;
    
    // Correct for pitch and yaw of head pose change 
    Application::Data::headPoseCorrection.x = tan(rotationAngles[0])*_tvec.at<double>(2)/PIXEL_WIDTH;
    Application::Data::headPoseCorrection.y = tan(rotationAngles[1])*_tvec.at<double>(2)/PIXEL_WIDTH;
    
    // Correct for head position change (in the horizontal and vertical directions)
    Application::Data::headPoseCorrection.x += _tvec.at<double>(0)/PIXEL_WIDTH;
    Application::Data::headPoseCorrection.y += _tvec.at<double>(1)/PIXEL_WIDTH;
    
    // TODO CHECK
    // Disable the correction in Y axis (seems it's doing a worse job)
    Application::Data::headPoseCorrection.y = 0;
    
    // Draw the corrected estimation on the debug window (WHITE)
	cv::Point correctedEstimation = Application::Data::gazePoints[0] + Application::Data::headPoseCorrection;
    
	Utils::boundToScreenArea(correctedEstimation);
    
    cv::circle(image,
        Utils::mapFromSecondMonitorToDebugFrameCoordinates(correctedEstimation),
        12, cv::Scalar(255, 255, 255), -1, 12, 0);
    
    // Draw the correction (w.r.t. center of monitor) too (BLUE)
    cv::circle(image,
        Utils::mapFromSecondMonitorToDebugFrameCoordinates(cv::Point(monitorCenterX + Application::Data::headPoseCorrection.x, monitorCenterY + Application::Data::headPoseCorrection.y)),
        4, cv::Scalar(255, 0, 0), -1, 4, 0);
        
        /*
    // Draw the projected eye region corner points
    for (int i=0; i<eyeRectangleCorners.size(); i++) {
        cv::circle(image, Utils::mapFromCameraToDebugFrameCoordinates(eyeRectangleCorners[i]), 2, cv::Scalar(0,255,255), 2);
        cv::circle(image, Utils::mapFromCameraToDebugFrameCoordinates(eyeRectangleCornersLeft[i]), 2, cv::Scalar(0,255,255), 2);
    }
    */
}

// Return the used head model (either the generic one or the personalized one)
std::vector<cv::Point3f> FacePoseEstimator::getUsedHeadModel() {
    if(Application::Signals::useGenericFaceModel)
        return _genericHeadModel;
    else
        return _headModel;
}

// One iteration of coordinate descent
void FacePoseEstimator::coordinateDescentIteration() {
    double stepSize = 0.00001;
    
    // After many iterations, use a smaller step size
    if(_iterationNumber > 1000) {
        stepSize /= 10;
    }
    
    // Iterate over the parameters
    for(int i=0; i<NUM_PERSONAL_PARAMETERS; i++) {
        // Calculate derivative for the parameter
        double derivative = calculateDerivative(i);
        //std::cout << "Derivative of par. index " << i << " is " << derivative << std::endl; 
        
        // Update the parameter's value
        _parameters[i] += stepSize*derivative;

        // Update the 3D head model with the new parameters
        calculateHeadModel(_parameters, _headModel);
    }
    
    // Clear the _sampleHeadPoseAngles before recalculating them all
    _sampleHeadPoseAngles.clear();
    
    // Update the rvec and tvec vectors for all samples using the updated 3D head model
    for(int i=0; i<_sampleCount; i++) {
        estimateFacePoseFrom2DPoints(_sampleFacePoints[i], _sampleRvecs[i], _sampleTvecs[i], true);
    
        // Calculate the updated Euler angles for face pose (yaw, pitch, roll) and store in the _sampleHeadPoseAngles vector
        _sampleHeadPoseAngles.push_back(getEulerAngles(_sampleRvecs[i]));
    }
    
    //std::cout << "Iteration #"<< _iterationNumber << ", avg. error = " << calculateProjectionErrors(_headModel) << std::endl;
    
    _iterationNumber++;
}

// Calculate the derivative for the parameter at the given index
double FacePoseEstimator::calculateDerivative(int parameterIndex) {
    // Do not let the parameters to deviate more than the allowed amounts (between 10 to 50% depending on parameter)
    //if(fabs(1-_parameters[parameterIndex]) > _parameterAllowedDeviations[parameterIndex])
    //    return 0;
    
    std::vector<double> modifiedParameters = _parameters;
    std::vector<cv::Point3f> modifiedHeadModel;

    // Change value of parameter a little
    modifiedParameters[parameterIndex] += 1.0e-5;

    // Get model corresponding to modifiedParameters
    calculateHeadModel(modifiedParameters, modifiedHeadModel);
    
    // Calculate the change in error metric
    double errorBefore = calculateProjectionErrors(_headModel);
    double errorAfter = calculateProjectionErrors(modifiedHeadModel);
    
    return (errorBefore - errorAfter)*1.0e+5;
}

// Calculate the head model using the given parameters
void FacePoseEstimator::calculateHeadModel(const std::vector<double> parameters, std::vector<cv::Point3f> &headModel) {
    headModel.clear();

    // Sellion
    headModel.push_back(cv::Point3f(0., 0.,0.));

    // Right and left eyes
    headModel.push_back(cv::Point3f(parameters[PAR_EYE_DEPTH]*-20., parameters[PAR_EYE_SEPARATION]*-59.5,-5.));
    headModel.push_back(cv::Point3f(parameters[PAR_EYE_DEPTH]*-20., parameters[PAR_EYE_SEPARATION]*+59.5,-5.));

    // Nose
    headModel.push_back(cv::Point3f(parameters[PAR_NOSE_DEPTH]*22.0, 0., parameters[PAR_NOSE_LENGTH]*-48.0));

    // Menton
    headModel.push_back(cv::Point3f(0., 0.,parameters[PAR_MENTON_LENGTH]*-117.5));

#ifdef EXTENDED_FACE_MODEL
    // Right and left ears
    headModel.push_back(cv::Point3f(parameters[PAR_EAR_DEPTH]*-100., parameters[PAR_EAR_SEPARATION]*-74.5,-6.));
    headModel.push_back(cv::Point3f(parameters[PAR_EAR_DEPTH]*-100., parameters[PAR_EAR_SEPARATION]*+74.5,-6.));

    // Stommion
    headModel.push_back(cv::Point3f(parameters[PAR_STOMMION_DEPTH]*10.0, 0., parameters[PAR_STOMMION_LENGTH]*-73.0));
#endif
}

// Calculate the 3D positions of 4 points defining the eye region for 
// both right and left eyes.
void FacePoseEstimator::calculateEyeRectangleCorners() {
    double eyeSeparation = _parameters[PAR_EYE_SEPARATION]*59.5*2;
    
    // Calculate eye region rectangle height and width
    double eyeHeight = eyeSeparation*0.17;
    double eyeWidth = 2*eyeHeight;
    double eyeHalfHeight = eyeHeight*0.5;
    
    cv::Point3f rightEyeCorner = _headModel[INDEX_RIGHT_EYE];
    cv::Point3f leftEyeCorner = _headModel[INDEX_LEFT_EYE];
    
    // RIGHT EYE
    // Clear previous points
    _eyeRectangleCorners3d.clear();
    
    // Top right corner
    _eyeRectangleCorners3d.push_back(cv::Point3f(rightEyeCorner.x, rightEyeCorner.y,            rightEyeCorner.z + eyeHalfHeight));
    // Bottom right corner
    _eyeRectangleCorners3d.push_back(cv::Point3f(rightEyeCorner.x, rightEyeCorner.y,            rightEyeCorner.z - eyeHalfHeight));
    // Bottom left corner
    _eyeRectangleCorners3d.push_back(cv::Point3f(rightEyeCorner.x, rightEyeCorner.y + eyeWidth, rightEyeCorner.z - eyeHalfHeight));
    // Top left corner
    _eyeRectangleCorners3d.push_back(cv::Point3f(rightEyeCorner.x, rightEyeCorner.y + eyeWidth, rightEyeCorner.z + eyeHalfHeight));
    
    // LEFT EYE
    // Clear previous points
    _eyeRectangleCorners3dLeft.clear();
    
    // Top right corner
    _eyeRectangleCorners3dLeft.push_back(cv::Point3f(leftEyeCorner.x, leftEyeCorner.y - eyeWidth, leftEyeCorner.z + eyeHalfHeight));
    // Bottom right corner
    _eyeRectangleCorners3dLeft.push_back(cv::Point3f(leftEyeCorner.x, leftEyeCorner.y - eyeWidth, leftEyeCorner.z - eyeHalfHeight));
    // Bottom left corner
    _eyeRectangleCorners3dLeft.push_back(cv::Point3f(leftEyeCorner.x, leftEyeCorner.y,            leftEyeCorner.z - eyeHalfHeight));
    // Top left corner
    _eyeRectangleCorners3dLeft.push_back(cv::Point3f(leftEyeCorner.x, leftEyeCorner.y,            leftEyeCorner.z + eyeHalfHeight));
}

// Calculates the average projection error using the given 3D head model
double FacePoseEstimator::calculateProjectionErrors(const std::vector<cv::Point3f> model) {
    std::vector<cv::Point2f> projectedPoints;
    double reprojectionErrors = 0;
    
    // Iterate over face samples
    for(int i=0; i<_sampleCount; i++) {
        // Reproject the head model using this sample's rvec and tvec
        projectPoints(model, _sampleRvecs[i], _sampleTvecs[i], projectedPoints);
        
        
        // Accumulate projection errors
        for(int j=0; j<projectedPoints.size(); j++) {
            reprojectionErrors += cv::norm(projectedPoints[j]-_sampleFacePoints[i][j]);
        }
    }

    // Return average reprojection error
    return reprojectionErrors/_sampleCount;
}

// Projects the given 3D model using the provided rotation and translation vectors
void FacePoseEstimator::projectPoints(const std::vector<cv::Point3f> model, const cv::Mat rvec, const cv::Mat tvec, std::vector<cv::Point2f> &projectedPoints) {
    cv::projectPoints(model, rvec, tvec, _projection, cv::noArray(), projectedPoints);
}

// Estimates the rvec and tvec vectors using the calculated _headModel and given facePoints (2d pixel positions on the image)
void FacePoseEstimator::estimateFacePoseFrom2DPoints(const std::vector<cv::Point2f> facePoints, cv::Mat &rvec, cv::Mat &tvec, bool useExtrinsicGuess) {
    cv::solvePnP(getUsedHeadModel(), facePoints,
            _projection, cv::noArray(),
            rvec, tvec, useExtrinsicGuess,
            1);    // Hardcoded value for cv::SOLVEPNP_EPNP or CV_EPNP (to fix problems with different versions of OpenCV)
//            0);    // Hardcoded value for cv::SOLVEPNP_ITERATIVE or CV_ITERATIVE (to fix problems with different versions of OpenCV)   
}

// Function to get the Euler angles (yaw, pitch, roll) from the OpenCV rotation matrix
cv::Vec3d FacePoseEstimator::getEulerAngles(cv::Mat rotationVector) {
    cv::Vec3d eulerAngles;
    
    // Convert the rotation vector to rotation matrix
    cv::Mat rotationMatrix;
    cv::Rodrigues(rotationVector, rotationMatrix);
    
    //cv::Matx33f rotation_matrix = rotationMatrix;
    cv::Mat faceAxes = cv::Mat::eye(3, 3, CV_64FC1);
    cv::Mat rotatedAxes = rotationMatrix*faceAxes;

    // Face axes (3 unit vectors in each axis)     
    double yaw = atan2(rotatedAxes.at<double>(0, 0), -rotatedAxes.at<double>(2, 0));
    double pitch = atan2(rotatedAxes.at<double>(1, 0), -rotatedAxes.at<double>(2, 0));
    double roll = atan2(rotatedAxes.at<double>(0, 2), -rotatedAxes.at<double>(1, 2));
    
    eulerAngles[0] = yaw;
    eulerAngles[1] = pitch;
    eulerAngles[2] = roll;
    
    return eulerAngles;
}


// Save the personal parameters to a text file
void FacePoseEstimator::saveParameters() {
    // Make sure the folder for saving user parameters exists  
    boost::filesystem::create_directories(USER_PARAMETERS_FOLDER);
    std::string fileName = std::string(USER_PARAMETERS_FOLDER) + "/" + Application::Settings::subject + ".txt";
    
    std::ofstream outputFile(fileName.c_str());
    boost::archive::text_oarchive oArchive(outputFile);
    
    // Serialize the _parameters vector
    oArchive & _parameters;
}

// Load the personal parameters from the text file, if it exists
void FacePoseEstimator::loadParameters() {
    // Check if the parameters file for this user exists
    std::string fileName = std::string(USER_PARAMETERS_FOLDER) + "/" + Application::Settings::subject + ".txt";
    
    if(boost::filesystem::exists(fileName)) {
        std::ifstream inputFile(fileName.c_str());
        {
            boost::archive::text_iarchive iArchive(inputFile);
            
            // Deserialize the _parameters vector
            iArchive & _parameters;
            
            // Do not learn any more when parameters are loaded
            Application::Settings::faceStructureLearning = false;
        }
    }
}