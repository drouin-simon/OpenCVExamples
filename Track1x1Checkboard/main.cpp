#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int maxCorners = 4;
RNG rng(12345);

void TrackCorners( Mat & src );

int main (int argc, char * const argv[])
{
    VideoCapture capture(0);
    if( !capture.isOpened() )
    {
        cout << "Cannot open capture session" << endl;
        return -1;
    }

    double captureWidth = capture.get( CV_CAP_PROP_FRAME_WIDTH );
    double captureHeight = capture.get( CV_CAP_PROP_FRAME_HEIGHT );
    cout << "Capture size: ( " << captureWidth << ", " << captureHeight << " )" << endl;
    Size capSize( captureWidth, captureHeight );

    namedWindow("Track1x1CheckBoard");
    createTrackbar( "maxCorners", "Track1x1CheckBoard", &maxCorners, 20 );

    Mat img;
    Mat imageGrey;
    /*Mat corners;
    Mat cornersNorm;
    Mat cornersNormScale;
    Mat thresholdedImg;
    Mat andImg;

    int blockSize = 2;
    int apertureSize = 3;
    double k = 0.04;
    int thresh = 150;

    createTrackbar( "Block Size", "Track1x1CheckBoard", &blockSize, 7 );
    createTrackbar( "Aperture size", "Track1x1CheckBoard", &apertureSize, 10 );
    createTrackbar( "corner threshold", "Track1x1CheckBoard", &thresh, 256 );

    const float black_level = 20.f;
    const float white_level = 130.f;
    const float black_white_gap = 70.f;*/

    std::vector<Point2f> imagePoints;

    for(;;)
    {
        // Capture image
        bool res = capture.grab();
        if( !( capture.grab() && capture.retrieve( img ) ) )
        {
            cerr << "timeout" << endl;
            waitKey( 50 );
            continue;
        }

        cv::cvtColor( img, imageGrey, CV_BGR2GRAY );
        cv::Size patternSize( 3, 3 );
        bool found = cv::findChessboardCorners( imageGrey, patternSize, imagePoints, cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE + cv::CALIB_CB_FAST_CHECK );

        if( found )
        {
            cv::cornerSubPix( imageGrey, imagePoints, cv::Size( 11, 11 ), cv::Size( -1, -1 ), cv::TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1 ) );

            if( imagePoints.size() >= 5 )
            {
                Point2f floatPoint = imagePoints[4];
                Point center( floatPoint.x, floatPoint.y );
                circle( img, center, 5, Scalar( 0, 0, 255 ) );
            }
        }
        //drawChessboardCorners( img, patternSize, imagePoints, found );

        /*cv::cvtColor( img, imageGrey, CV_BGR2GRAY );
        Mat white = imageGrey.clone();
        Mat black = imageGrey.clone();

        erode( white, white, Mat() );
        dilate( black, black, Mat() );

        bitwise_xor( white, black, andImg );*/

        /*
        cv::cvtColor( img, imageGrey, CV_BGR2GRAY );
        cornerHarris( imageGrey, corners, blockSize, apertureSize, k, BORDER_DEFAULT );

        normalize( corners, cornersNorm, 0, 255, NORM_MINMAX, CV_32FC1, Mat() );
        convertScaleAbs( cornersNorm, cornersNormScale );

        Scalar threshMin( thresh );
        Scalar threshMax( 256 );
        inRange( cornersNormScale, threshMin, threshMax, thresholdedImg );
        */

        //TrackCorners( img );
        imshow( "Track1x1CheckBoard", img );

        // Capture keyboard
        char code = (char)waitKey( 50 );
        if( code == 27 || code == 'q' || code == 'Q' )
            break;
    }

    return 0;
}

void TrackCorners( Mat & src )
{
    Mat srcGray;
    cv::cvtColor( src, srcGray, CV_BGR2GRAY );

    /// Parameters for Shi-Tomasi algorithm
    vector<Point2f> corners;
    double qualityLevel = 0.01;
    double minDistance = 10;
    int blockSize = 3;
    bool useHarrisDetector = false;
    double k = 0.04;

    // Apply corner detection
    goodFeaturesToTrack( srcGray, corners, maxCorners, qualityLevel, minDistance, Mat(), blockSize, useHarrisDetector, k );

    // Draw corners detected
    for( int i = 0; i < corners.size(); i++ )
    {
        circle( src, corners[i], 4, Scalar(rng.uniform(0,255), rng.uniform(0,255), rng.uniform(0,255)), -1, 8, 0 );
    }

    /// Show what you got
    imshow( "Track1x1CheckBoard", src );

    /// Set the neeed parameters to find the refined corners
    //Size winSize = Size( 5, 5 );
    //Size zeroZone = Size( -1, -1 );
    //TermCriteria criteria = TermCriteria( CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 40, 0.001 );

    /// Calculate the refined corner locations
    //cornerSubPix( src_gray, corners, winSize, zeroZone, criteria );

}
