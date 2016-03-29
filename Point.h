#pragma once

struct Point {
	double x;
	double y;

	Point();
	Point(double x, double y);
	Point(cv::Point2f const &point);
	Point(cv::Point const &point);

	double distance(Point other) const;
    double distance2f(cv::Point2f other) const;
	int closestPoint(const std::vector<Point> &points) const;

	void operator=(cv::Point const &point);
    void operator=(cv::Point2f const &point);
	Point operator+(const Point &other) const;
	Point operator-(const Point &other) const;
};

std::ostream &operator<< (std::ostream &out, const Point &p);
std::istream &operator>> (std::istream &in, Point &p);

