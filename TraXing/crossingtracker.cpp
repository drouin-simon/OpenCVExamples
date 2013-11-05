#include "crossingtracker.h"

using namespace cv;
using namespace std;

CrossingTracker::CrossingTracker()
    : m_hsvMin( 25, 84, 130 )
    , m_hsvMax( 82, 256, 256 )
{
    m_nGate = 38;
    m_blurRadius = 2;
    m_minXingDist = 4.0;
    m_minimumBoundSize = 15;
    m_dilateErodeSize = 1;
}

bool CrossingTracker::TrackCrossing( Mat & img, Rect & bounds, Point2f & crossingPos )
{
    // Find colored blob's bounding box. If not found, give up
    if( !FindBoundingBoxOfColoredArea( img, bounds ) )
        return false;

    // Get the sub-image that contains the bounding box
    m_subImg = img( bounds );

    // Detect corners
    cv::cvtColor( m_subImg, m_imageGrey, CV_BGR2GRAY );
    int blurSize = 2 * m_blurRadius + 1;
    cv::GaussianBlur( m_imageGrey, m_imageBlur, Size(blurSize,blurSize), 0 );

    vector<Point2f> xing;
    FindXing( m_imageBlur, xing );

    // need at least 1 corner, hope there is only one
    if( xing.size() < 1 )
        return false;

    // Find subpixel coord of corner
    cv::cornerSubPix( m_imageGrey, xing, cv::Size( 11, 11 ), cv::Size( -1, -1 ), cv::TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1 ) );
    crossingPos = Point2f( (double)bounds.x + xing[0].x, (double)bounds.y + xing[0].y );
    return true;
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
void CrossingTracker::FindXing( Mat & img, vector<Point2f> & xing )
{
    uchar * itRow = img.ptr( pixelRingRadius, pixelRingRadius );
    int maxCol = img.cols - pixelRingRadius - 1;
    int maxRow = img.rows - pixelRingRadius - 1;
    for( int row = pixelRingRadius; row < maxRow; ++row, itRow+=img.step[0] )
    {
        uchar * it = itRow;
        for( int col = pixelRingRadius; col < maxCol; ++col, ++it )
        {
            if( IsCorner( img, it, m_nGate ) )
            {
                // keep point only if different enough from prev.
                bool found = false;
                Point2f newP( (float)col, (float)row );
                for( unsigned p = 0; p < xing.size() && !found; ++p )
                {
                    if( norm( xing[p] - newP ) < m_minXingDist )
                        found = true;
                }
                if( !found )
                    xing.push_back( newP );
            }
        }
    }
}


bool CrossingTracker::FindBoundingBoxOfColoredArea( Mat & img, Rect & bounds )
{
    // convert to hsv
    cvtColor( img, m_imageHsv, CV_BGR2HSV );

    // threashold
    inRange( m_imageHsv, m_hsvMin, m_hsvMax, m_threshImg );

    // Dilate/Erode
    Mat kernel = getStructuringElement( MORPH_ELLIPSE, Size( 2 * m_dilateErodeSize + 1, 2 * m_dilateErodeSize + 1 ), Point( m_dilateErodeSize, m_dilateErodeSize ) );
    erode( m_threshImg, m_erodeImg, kernel );
    dilate( m_erodeImg, m_dilateImg, kernel );

    // Find contours
    vector< vector<Point> > contours;
    vector<Vec4i> hierarchy;
    m_dilateImg.copyTo( m_temp );
    findContours( m_temp, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE );
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
        if( minSide > m_minimumBoundSize && maxSide < 150 && ratio > 0.8 && ratio < 1.2 )
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
        return true;
    }
    return false;
}
