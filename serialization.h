#pragma once

#include <opencv2/opencv.hpp>
#include <boost/serialization/split_free.hpp>
#include <boost/serialization/vector.hpp>
 
BOOST_SERIALIZATION_SPLIT_FREE(cv::Mat)
namespace boost {
	namespace serialization {
		/** Serialization support for cv::Mat */
		template<class Archive>
		void save(Archive & ar, const cv::Mat& m, const unsigned int version);

		/** Serialization support for cv::Mat */
		template<class Archive>
		void load(Archive & ar, cv::Mat& m, const unsigned int version);
	}
}
 
BOOST_SERIALIZATION_SPLIT_FREE(cv::Point)
namespace boost {
	namespace serialization {
		/** Serialization support for cv::Point */
		template<class Archive>
		void save(Archive & ar, const cv::Point& m, const unsigned int version);

		/** Serialization support for cv::Point */
		template<class Archive>
		void load(Archive & ar, cv::Point& m, const unsigned int version);
	}
}