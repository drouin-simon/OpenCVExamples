#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>
#include "crossingtracker.h"

using namespace cv;
using namespace std;

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

    namedWindow("TraXing");
    CrossingTracker tracker;

    Mat img;

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

        Point2f crossingPos;
        Rect bounds;
        if( tracker.TrackCrossing( img, bounds, crossingPos ) )
        {
            circle( img, crossingPos, 4, Scalar( 0, 0, 255 ), 1 );
            rectangle( img, bounds, Scalar( 255, 255, 0, 0 ) );
        }

        //TrackCorners( img );
        imshow( "TraXing", img );

        // Capture keyboard
        char code = (char)waitKey( 50 );
        if( code == 27 || code == 'q' || code == 'Q' )
            break;
    }

    return 0;
}
