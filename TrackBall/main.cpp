#include <iostream>
#include <vector>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/tracking.hpp>

using namespace cv;
using namespace std;

int hsvMin[4] = { 150, 84, 130, 0 };
int hsvMax[4] = { 358, 256, 255, 0 };

int dilateErodeSize = 1;

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

    namedWindow("TrackBall-capture");

    namedWindow("TrackBall-processed");
    createTrackbar( "Hue Min", "TrackBall-processed", &hsvMin[0], 360 );
    createTrackbar( "Hue Max", "TrackBall-processed", &hsvMax[0], 360 );
    createTrackbar( "Sat Min", "TrackBall-processed", &hsvMin[1], 255 );
    createTrackbar( "Sat Max", "TrackBall-processed", &hsvMax[1], 255 );
    createTrackbar( "Val Min", "TrackBall-processed", &hsvMin[2], 255 );
    createTrackbar( "Val Max", "TrackBall-processed", &hsvMax[2], 255 );
    createTrackbar( "Dilate/erode size", "TrackBall-processed", &dilateErodeSize, 5 );

    Mat img;
    Mat hsvImg;
    Mat threshImg;
    Mat erodeImg;
    Mat dilateImg;
    Mat blurredImg;
    Mat temp;

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

        // convert to hsv
        cvtColor( img, hsvImg, CV_BGR2HSV );

        // threashold
        Scalar threshMin( hsvMin[0], hsvMin[1], hsvMin[2], hsvMin[3] );
        Scalar threshMax( hsvMax[0], hsvMax[1], hsvMax[2], hsvMax[3] );
        inRange( hsvImg, threshMin, threshMax, threshImg );

        // Dilate/Erode
        Mat kernel = getStructuringElement( MORPH_ELLIPSE, Size( 2 * dilateErodeSize + 1, 2 * dilateErodeSize + 1 ), Point( dilateErodeSize, dilateErodeSize ) );
        erode( threshImg, erodeImg, kernel );
        dilate( erodeImg, dilateImg, kernel );

        // Find contours
        vector< vector<Point> > contours;
        vector<Vec4i> hierarchy;
        dilateImg.copyTo( temp );
        findContours( temp, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE );
        drawContours( img, contours, -1, Scalar( 255, 0, 0, 255 ) );

        // Get the bounding box of the most likely contour:
        //  1 - bounding box more or less square (0.8 < width/heigh ratio < 1.2 )
        //  2 - likely area:
        //
        int largerTarget = 0;
        int largerIndex = -1;
        for( unsigned i = 0; i < contours.size(); ++i )
        {
            Rect rect = boundingRect( contours[i] );
            int maxSide = rect.width > rect.height ? rect.width : rect.height;
            double ratio = (double)rect.width / rect.height;
            if( maxSide > 5 && maxSide < 150 && ratio > 0.8 && ratio < 1.2 )
            {
                if( maxSide > largerTarget )
                {
                    largerTarget = maxSide;
                    largerIndex = i;
                }
            }

            //RotatedRect rect = minAreaRect( contours[i] );
        }

        if( largerIndex != -1 )
            rectangle( img, boundingRect( contours[ largerIndex ] ), Scalar( 255, 255, 0, 0 ) );

        /*// Blur image
        GaussianBlur( dilateImg, blurredImg, Size(9, 9), 2, 2 );

        // Find circles
        vector<Vec3f> circles;
        HoughCircles( blurredImg, circles, CV_HOUGH_GRADIENT, 2, 100 );
        for( size_t i = 0; i < circles.size(); i++ )
        {
            Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
            int radius = cvRound(circles[i][2]);
            // draw the circle center
            circle( img, center, 3, Scalar(0,255,0), -1, 8, 0 );
            // draw the circle outline
            circle( img, center, radius, Scalar(0,0,255), 3, 8, 0 );
        }*/

        // Show images
        imshow( "TrackBall-capture", img );
        imshow( "TrackBall-processed", dilateImg );

        // Capture keyboard
        char code = (char)waitKey( 50 );
        if( code == 27 || code == 'q' || code == 'Q' )
            break;
    }

    return 0;
}
