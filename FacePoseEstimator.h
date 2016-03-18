#pragma once

// Use BLAS and LAPACK libraries for dlib
#define DLIB_USE_BLAS
#define DLIB_USE_LAPACK

#include "Component.h"

#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <dlib/image_io.h>
#include <dlib/opencv.h>


/*
 * OpenCV Camshift implementation adapted from Billy Lamberta's gist:
 *
 *      https://gist.github.com/lamberta/231696
 */
typedef struct {
  cv::Mat hsv;     //input image converted to HSV
  cv::Mat hue;     //hue channel of HSV image
  cv::Mat mask;    //image for masking pixels
  cv::Mat prob;    //face probability estimates for each pixel

  cv::Mat hist;     //histogram of hue in original face image
} TrackedObject;


/*
 * OpenCV head pose angle estimation implementation adapted from CHILI Lab / EPFL:
 *
 *      https://github.com/chili-epfl/attention-tracker/blob/master/src/head_pose_estimation.hpp
 */
// Anthropometric for adults (More or less the average of 95th percentile male & female values)
// Relative position of various facial feature relative to sellion
// Values taken from https://en.wikipedia.org/wiki/Human_head
// X points forward
const static cv::Point3f P3D_SELLION(0., 0.,0.);
const static cv::Point3f P3D_RIGHT_EYE(-20., -65,-5.);
const static cv::Point3f P3D_LEFT_EYE(-20., 65,-5.);
const static cv::Point3f P3D_RIGHT_EAR(-100., -74.5,-6.);
const static cv::Point3f P3D_LEFT_EAR(-100., 74.5,-6.);
const static cv::Point3f P3D_NOSE(21.0, 0., -46.0);
const static cv::Point3f P3D_STOMMION(10.0, 0., -73.0);
const static cv::Point3f P3D_MENTON(0., 0.,-128.5);


const static cv::Point3f P3D_EYE_REGION_TOP_RIGHT(-20., -65, 8.);
const static cv::Point3f P3D_EYE_REGION_TOP_LEFT(-20., 65, 8.);
const static cv::Point3f P3D_EYE_REGION_BOTTOM_RIGHT(-20., -65, -15.);
const static cv::Point3f P3D_EYE_REGION_BOTTOM_LEFT(-20., 65, -15.);

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
	
    bool isFaceInitialized;
	cv::Rect faceRectangle;
	cv::RotatedRect faceRotatedRectangle;
    dlib::rectangle faceRectangleDlib;
	std::vector<cv::Point> facialLandmarks;
    cv::Matx44d headPose;
	
private:
    bool _isActive;
    dlib::array2d<dlib::rgb_pixel> _videoFrame;
    dlib::array2d<dlib::rgb_pixel> _reducedSizeVideoFrame;
    dlib::frontal_face_detector _faceDetector;   // dlib face detector
    dlib::shape_predictor _faceShapePredictor;   // Pre-trained face shape predictor
    
    TrackedObject _faceTracker;                     // Camshift tracker for the face


    cv::Mat _rvec, _tvec;
    cv::Matx33f _projection;
    std::vector<cv::Point3f> _headPoints;
    std::vector<cv::Point3f> _eyeRegionPoints;
    
    bool detectFace();
    void allocateFaceTracker();
    void extractHueAndMask();
    void resetFaceTracking(cv::Rect faceDetection);
    void trackFace();
    void calculatePose();
};
