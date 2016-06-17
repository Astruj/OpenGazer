#pragma once

#include <opencv2/opencv.hpp>
#include <boost/serialization/split_free.hpp>
#include <boost/serialization/vector.hpp>
 

BOOST_SERIALIZATION_SPLIT_FREE(cv::Mat)
BOOST_SERIALIZATION_SPLIT_FREE(cv::Point)

// cv::Mat and cv::Point serialization support in boost
// Code adapted from: https://cheind.wordpress.com/2011/12/06/serialization-of-cvmat-objects-using-boost/
namespace boost {
	namespace serialization {
		// Serialization
		template<class Archive>
		void save(Archive & ar, const cv::Mat& m, const unsigned int version)
		{
			size_t elem_size = m.elemSize();
			size_t elem_type = m.type();

			ar & m.cols;
			ar & m.rows;
			ar & elem_size;
			ar & elem_type;

			const size_t data_size = m.cols * m.rows * elem_size;
			ar & boost::serialization::make_array(m.ptr(), data_size);
		}

		// Deserialization
		template<class Archive>
		void load(Archive & ar, cv::Mat& m, const unsigned int version)
		{
			int cols, rows;
			size_t elem_size, elem_type;

			ar & cols;
			ar & rows;
			ar & elem_size;
			ar & elem_type;

			m.create(rows, cols, elem_type);

			size_t data_size = m.cols * m.rows * elem_size;
			ar & boost::serialization::make_array(m.ptr(), data_size);
		}
		
		// Serialization
		template<class Archive>
		void save(Archive & ar, const cv::Point& p, const unsigned int version)
		{
			ar & p.x;
			ar & p.y;
		}

		// Deserialization
		template<class Archive>
		void load(Archive & ar, cv::Point& p, const unsigned int version)
		{
			ar & p.x;
			ar & p.y;
		}
	}
}