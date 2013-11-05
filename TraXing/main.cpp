#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int nGate = 38;
int blurRadius = 2;
double minXingDist = 4.0;

Mat img;
Mat imageGrey;
Mat imageBlur;

// For colored region finding
int minimumBoundSize = 15;
int dilateErodeSize = 1;
int hsvMin[4] = { 25, 84, 130, 0 };
int hsvMax[4] = { 82, 256, 256, 0 };
Mat imageHsv;
Mat threshImg;
Mat erodeImg;
Mat dilateImg;
Mat temp;

void FindXing( Mat & img, vector<Point2f> & xing, int nGate );
bool FindBoundingBoxOfColoredArea( Mat img, double minBoundSize, Rect & bounds );

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

        // Find colored blob's bounding box. If not found, give up
        Rect bounds;
        vector<Point2f> xing;
        if( FindBoundingBoxOfColoredArea( img, minimumBoundSize, bounds ) )
        {
            // Get the sub-image that contains the bounding box
            Mat subImg = img( bounds );

            // Detect corners
            cv::cvtColor( subImg, imageGrey, CV_BGR2GRAY );
            int blurSize = 2 * blurRadius + 1;
            cv::GaussianBlur( imageGrey, imageBlur, Size(blurSize,blurSize), 0 );

            FindXing( imageBlur, xing, nGate );

            // Draw first xing in image (assume there is only one)
            if( xing.size() > 0 )
            {
                // Find subpixel coord of corner
                cv::cornerSubPix( imageGrey, xing, cv::Size( 11, 11 ), cv::Size( -1, -1 ), cv::TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1 ) );
                Point2f center( (double)bounds.x + xing[0].x, (double)bounds.y + xing[0].y );
                circle( img, center, 4, Scalar( 0, 0, 255 ), 1 );
            }
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
            {
                // keep point only if different enough from prev.
                bool found = false;
                Point2f newP( (float)col, (float)row );
                for( unsigned p = 0; p < xing.size() && !found; ++p )
                {
                    if( norm( xing[p] - newP ) < minXingDist )
                        found = true;
                }
                if( !found )
                    xing.push_back( newP );
            }
        }
    }
}


bool FindBoundingBoxOfColoredArea( Mat img, double minBoundSize, Rect & bounds )
{
    // convert to hsv
    cvtColor( img, imageHsv, CV_BGR2HSV );

    // threashold
    Scalar threshMin( hsvMin[0], hsvMin[1], hsvMin[2], hsvMin[3] );
    Scalar threshMax( hsvMax[0], hsvMax[1], hsvMax[2], hsvMax[3] );
    inRange( imageHsv, threshMin, threshMax, threshImg );

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
        int minSide = rect.width > rect.height ? rect.height : rect.width;
        double ratio = (double)rect.width / rect.height;
        if( minSide > minBoundSize && maxSide < 150 && ratio > 0.8 && ratio < 1.2 )
        {
            if( maxSide > largerTarget )
            {
                largerTarget = maxSide;
                largerIndex = i;
            }
        }
    }

    if( largerIndex != -1 )
    {
        bounds = boundingRect( contours[ largerIndex ] );
        rectangle( img, bounds, Scalar( 255, 255, 0, 0 ) );
        return true;
    }
    return false;
}
