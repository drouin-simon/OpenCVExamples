#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int nGate = 38;
int blurRadius = 2;

void FindXing( Mat & img, vector<Point2f> & xing, int nGate );

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
    createTrackbar( "Gate", "TraXing", &nGate, 255 );
    createTrackbar( "Blur radius", "TraXing", &blurRadius, 10 );

    Mat img;
    Mat imageGrey;
    Mat imageBlur;

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

        // Detect corners
        cv::cvtColor( img, imageGrey, CV_BGR2GRAY );
        int blurSize = 2 * blurRadius + 1;
        cv::GaussianBlur( imageGrey, imageBlur, Size(blurSize,blurSize), 0 );

        vector<Point2f> xing;
        FindXing( imageBlur, xing, nGate );

        // Draw xing in image
        for( unsigned k = 0; k < xing.size(); ++k )
        {
            circle( img, xing[k], 4, Scalar( 0, 0, 255 ), -1 );
        }

        stringstream s;
        s << "Nb Xings: " << xing.size();
        putText( img, s.str(), Point( 10, img.rows - 10 ), FONT_HERSHEY_PLAIN, 1.0, Scalar( 255, 255, 255 ) );

        //TrackCorners( img );
        imshow( "TraXing", img );

        // Capture keyboard
        char code = (char)waitKey( 50 );
        if( code == 27 || code == 'q' || code == 'Q' )
            break;
    }

    return 0;
}

// Code copied from PTAM that was adapted to run on OpenCV images instead of libCVD images

const int pixelRingRadius = 3;
const int pixelRing[16][2] =
{
    {  0,  3 },
    {  1,  3 },
    {  2,  2 },
    {  3,  1 },
    {  3,  0 },
    {  3, -1 },
    {  2, -2 },
    {  1, -3 },
    {  0, -3 },
    { -1, -3 },
    { -2, -2 },
    { -3, -1 },
    { -3,  0 },
    { -3,  1 },
    { -2,  2 },
    { -1,  3 }
};

inline uchar getRingValue( cv::Mat & img, uchar * currentIt, int ringIndex )
{
    uchar * res = currentIt + pixelRing[ringIndex][1] * img.step[0] + pixelRing[ringIndex][0];
    return *res;
}

inline bool IsCorner( cv::Mat & img, uchar * it, int nGate )
{
    assert( img.type() == CV_8UC1 );

    // Does a quick check to see if a point in an image could be a grid corner.
    // Does this by going around a 16-pixel ring, and checking that there's four
    // transitions (black - white- black - white - )
    // Also checks that the central pixel is blurred.

    // Find the mean intensity of the pixel ring...
    int nSum = 0;
    static uchar abPixels[16];
    for( int i = 0; i < 16; i++ )
    {
        abPixels[ i ] = getRingValue( img, it, i );
        nSum += abPixels[i];
    };
    int nMean = nSum / 16;
    int nHiThresh = nMean + nGate;
    int nLoThresh = nMean - nGate;

    // If the center pixel is (not?) roughly the same as the mean, this isn't a corner.
    // we expect the center pixel to be fuzzy
    int nCenter = *it;
    if( nCenter <= nLoThresh || nCenter >= nHiThresh )
        return false;

    // Count transitions around the ring... there should be four!
    bool bState = (abPixels[15] > nMean);  // bState is true for a white pixel
    int nSwaps = 0;
    for( int i = 0; i < 16; i++ )
    {
        uchar bValNow = abPixels[i];
        if( bState )
        {
            if( bValNow < nLoThresh )
            {
                bState = false;
                nSwaps++;
            }
        }
        else
            if( bValNow > nHiThresh )
            {
                bState = true;
                nSwaps++;
            };
    }
    return (nSwaps == 4);
}


// assuming img is a greyscale image
void FindXing( Mat & img, vector<Point2f> & xing, int nGate )
{
    uchar * itRow = img.ptr( pixelRingRadius, pixelRingRadius );
    int maxCol = img.cols - pixelRingRadius - 1;
    int maxRow = img.rows - pixelRingRadius - 1;
    for( int row = pixelRingRadius; row < maxRow; ++row, itRow+=img.step[0] )
    {
        uchar * it = itRow;
        for( int col = pixelRingRadius; col < maxCol; ++col, ++it )
        {
            if( IsCorner( img, it, nGate ) )
                xing.push_back( Point2f( (float)col, (float)row ) );
        }
    }
}
