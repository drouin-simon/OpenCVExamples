#ifndef __CrossingTracker_h_
#define __CrossingTracker_h_

#include <opencv2/opencv.hpp>
#include <vector>

class CrossingTracker
{

public:

    CrossingTracker();

    bool TrackCrossing( cv::Mat & img, cv::Rect & bounds, cv::Point2f & crossingPos );

protected:

    void FindXing( cv::Mat & img, std::vector<cv::Point2f> & xing );
    bool FindBoundingBoxOfColoredArea( cv::Mat & img, cv::Rect & bounds );

    // params used to find crossing
    int m_nGate;
    int m_blurRadius;
    double m_minXingDist;

    // temp images
    cv::Mat m_imageGrey;
    cv::Mat m_imageBlur;
    cv::Mat m_subImg;

    // For colored region finding
    int m_minimumBoundSize;
    int m_dilateErodeSize;
    cv::Scalar m_hsvMin;
    cv::Scalar m_hsvMax;
    cv::Mat m_imageHsv;
    cv::Mat m_threshImg;
    cv::Mat m_erodeImg;
    cv::Mat m_dilateImg;
    cv::Mat m_temp;
};

#endif
