# To configure the project and create the Makefile, run the following on the terminal:
#			qmake -spec macx-g++ opengazer.pro

CONFIG	+=	qt
QT += gui widgets

HEADERS += 	Calibrator.h HeadTracker.h LeastSquares.h EyeExtractor.h GazeTracker.h MainGazeTracker.h OutputMethods.h PointTracker.h FaceDetector.h Point.h utils.h BlinkDetector.h FeatureDetector.h mir.h DebugWindow.h Application.h Video.h AnchorPointSelector.h Command.h ImageWindow.h ImageWidget.h TestWindow.h Prefix.hpp HiResTimer.h EyeCenterDetector.h GoogleGlassWindow.h EyeExtractorSegmentationGroundTruth.h FrogGame.h HistogramFeatureExtractor.h PointTrackerWithTemplate.h GazeTrackerHistogramFeatures.h Component.h Configuration.h FacePoseEstimator.h

SOURCES += 	opengazer.cpp Calibrator.cpp HeadTracker.cpp LeastSquares.cpp EyeExtractor.cpp GazeTracker.cpp MainGazeTracker.cpp OutputMethods.cpp PointTracker.cpp FaceDetector.cpp Point.cpp utils.cpp BlinkDetector.cpp FeatureDetector.cpp mir.cpp DebugWindow.cpp Application.cpp Video.cpp AnchorPointSelector.cpp Command.cpp ImageWindow.cpp ImageWidget.cpp TestWindow.cpp HiResTimer.cpp EyeCenterDetector.cpp GoogleGlassWindow.cpp EyeExtractorSegmentationGroundTruth.cpp  FrogGame.cpp HistogramFeatureExtractor.cpp PointTrackerWithTemplate.cpp GazeTrackerHistogramFeatures.cpp Configuration.cpp FacePoseEstimator.cpp ./data/dlib/dlib/all/source.cpp

TARGET  = 	opengazer

QMAKE_CFLAGS 	+= `pkg-config opencv --cflags` -I./data/dlib/ -include Prefix.hpp -Wno-sign-compare -Wno-reorder -Wno-unused-variable -Wno-unused-function -Wno-unused-result -Wno-unused-parameter -Wno-unknown-pragmas
QMAKE_CXXFLAGS 	+= `pkg-config opencv --cflags` -I./data/dlib/ -include Prefix.hpp -Wno-sign-compare -Wno-reorder -Wno-unused-variable -Wno-unused-function -Wno-unused-result -Wno-unused-parameter -Wno-unknown-pragmas
QMAKE_LIBS 	+= `pkg-config opencv --libs`

macx {
    QMAKE_MAC_SDK=macosx10.11
    
    # Mac OS X linker parameters and include directories
    QMAKE_LIBS += -L/usr/local/lib -lm -ldl -lfann -lboost_filesystem-mt -lboost_system-mt -lgsl  -lgslcblas -llapack -lblas

    # NOTE: Depending on your Mac OS X / XCode version you might need to use the
    # configuration that is commented out
    QMAKE_CFLAGS 	+= -stdlib=libc++ -DDLIB_NO_GUI_SUPPORT -DUSE_AVX_INSTRUCTIONS=ON -DUSE_SSE4_INSTRUCTIONS=ON
    QMAKE_CXXFLAGS 	+= -stdlib=libc++ -DDLIB_NO_GUI_SUPPORT -DUSE_AVX_INSTRUCTIONS=ON -DUSE_SSE4_INSTRUCTIONS=ON
    QMAKE_LFLAGS    += -stdlib=libc++

    QMAKE_CXXFLAGS += -I`brew --prefix boost`/include
    QMAKE_CXXFLAGS += -I`brew --prefix gsl`/include

    #QMAKE_CFLAGS 	+= -stdlib=libstdc++
    #QMAKE_CXXFLAGS 	+= -stdlib=libstdc++
}

unix:!macx {
    # Linux linker parameters and include directories
    QMAKE_LIBS += -L/usr/local/lib -L/opt/local/lib -lm -ldl -lgthread-2.0 -lfann -lboost_filesystem -lboost_system -lgsl -lgslcblas -llapack -lblas
    #QMAKE_LIBS += -L/usr/local/cuda-6.5/lib64/ -lGLEW

    QMAKE_CFLAGS 	+= -DDLIB_NO_GUI_SUPPORT -DUSE_AVX_INSTRUCTIONS=ON -DUSE_SSE4_INSTRUCTIONS=ON
    QMAKE_CXXFLAGS 	+= -DDLIB_NO_GUI_SUPPORT -DUSE_AVX_INSTRUCTIONS=ON -DUSE_SSE4_INSTRUCTIONS=ON
    INCLUDEPATH += /usr/local/include
    QMAKE_CXXFLAGS += -I/usr/include/eigen3
}
