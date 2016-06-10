#pragma once

// Use BLAS and LAPACK libraries for dlib
//#define DLIB_USE_BLAS
//#define DLIB_USE_LAPACK

#include "Component.h"

#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <dlib/image_io.h>
#include <dlib/opencv.h>

// Indexes for 3D points included in the head model
#define INDEX_SELLION 0
#define INDEX_RIGHT_EYE 1
#define INDEX_LEFT_EYE 2
#define INDEX_NOSE 3
#define INDEX_MENTON 4

// 3 additional points for the extended model
#define INDEX_RIGHT_EAR 5
#define INDEX_LEFT_EAR 6
#define INDEX_STOMMION 7

// Indexes for 3D points for the eye rectangle corners
#define INDEX_EYE_TOP_OUTER 0
#define INDEX_EYE_TOP_INNER 1
#define INDEX_EYE_BOTTOM_OUTER 2
#define INDEX_EYE_BOTTOM_INNER 3


// Indexes for the personal parameters for the head model
#define PAR_EYE_DEPTH 0
#define PAR_EYE_SEPARATION 1
#define PAR_NOSE_DEPTH 2
#define PAR_NOSE_LENGTH 3
#define PAR_MENTON_LENGTH 4

#define PAR_EAR_DEPTH 5
#define PAR_EAR_SEPARATION 6
#define PAR_STOMMION_DEPTH 7
#define PAR_STOMMION_LENGTH 8

//#define EXTENDED_FACE_MODEL

// If extended is defined, there will be 4 more parameters (ear and stommion)
#ifdef EXTENDED_FACE_MODEL
#define NUM_PERSONAL_PARAMETERS 9
#else
#define NUM_PERSONAL_PARAMETERS 5
#endif


enum FACIAL_FEATURE {
    NOSE=30,
    RIGHT_EYE=36,
    LEFT_EYE=45,
    RIGHT_SIDE=0,
    LEFT_SIDE=16,
    EYEBROW_RIGHT=21,
    EYEBROW_LEFT=22,
    MOUTH_UP=51,
    MOUTH_DOWN=57,
    MOUTH_RIGHT=48,
    MOUTH_LEFT=54,
    SELLION=27,
    MOUTH_CENTER_TOP=62,
    MOUTH_CENTER_BOTTOM=66,
    MENTON=8
};

/*
 * Face pose estimation algorithm by Kazemi, V. and Sullivan, J.
 *
 *      One Millisecond Face Alignment with an Ensemble of Regression Trees. Vahid Kazemi and Josephine Sullivan, CVPR 2014
 *
 * Implementation adapted from dlib examples
 *
 *      https://github.com/davisking/dlib/blob/master/examples/face_landmark_detection_ex.cpp
 */
class FacePoseEstimator: public Component {
public:
	FacePoseEstimator();
	cv::Point findEyeCenter(cv::Mat eyeImage);
	void process();
	void draw();
    bool isActive();
	
    bool isFaceFound;
	cv::Rect faceRectangle;
	std::vector<cv::Point> facialLandmarks;
    
    cv::Point2f rightEye;
    cv::Point2f leftEye;
    std::vector<cv::Point2f> eyeRectangleCorners;    // 2D points defining the bounds of the eye rectangles
    std::vector<cv::Point2f> eyeRectangleCorners3dLeft;

private:
    bool _isActive;
    
    dlib::array2d<dlib::rgb_pixel> _videoFrame;
    dlib::array2d<dlib::rgb_pixel> _reducedSizeVideoFrame;
    dlib::frontal_face_detector _faceDetector;   // dlib face detector
    dlib::shape_predictor _faceShapePredictor;   // Pre-trained face shape predictor
    
    cv::Mat _rvec, _tvec;
    cv::Matx33f _projection;
    std::vector<cv::Point3f> _genericHeadModel;
    std::vector<cv::Point3f> _headModel;
    
    std::vector<cv::Point3f> _eyeRectangleCorners3d;    // 3D points defining the bounds of the eye rectangles
    std::vector<cv::Point3f> _eyeRectangleCorners3dLeft;

    int _sampleCount;
    int _lastSampleFrame;
    std::vector< std::vector<cv::Point2f> > _sampleFacePoints;
    std::vector<cv::Mat> _sampleRvecs;
    std::vector<cv::Mat> _sampleTvecs;
    std::vector<cv::Vec3d> _sampleHeadPoseAngles;
    std::vector<cv::Mat> _facialLandmarks3d;

    // Variables for coordinate descent
    std::vector<double> _parameters;
    std::vector<double> _parameterAllowedDeviations;
    int _iterationNumber;
    
    bool detectFace();
    void estimateFacePose();
    bool shouldAddFaceSample();
    void addFaceSample();
    
    // Coordinate descent related functions for personalizing the face parameters
    void coordinateDescentIteration();
    double calculateDerivative(int parameterIndex);
    void calculateHeadModel(const std::vector<double> parameters, std::vector<cv::Point3f> &headModel);
    double calculateProjectionErrors(const std::vector<cv::Point3f> model);
    void projectPoints(const std::vector<cv::Point3f> model, const cv::Mat rvec, const cv::Mat tvec, std::vector<cv::Point2f> &projectedPoints);
    void estimateFacePoseFrom2DPoints(const std::vector<cv::Point2f> facePoints, cv::Mat &rvec, cv::Mat &tvec, bool useExtrinsicGuess);
    std::vector<cv::Point3f> getUsedHeadModel();
    cv::Vec3d getEulerAngles(cv::Mat rotationVector);
    
    void calculateEyeRectangleCorners();
    
    // Save & load personal parameters
    void saveParameters();
    void loadParameters();
};
