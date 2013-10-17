#include <iostream>
#include <vector>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/tracking.hpp>

using namespace cv;
using namespace std;

struct mouse_info_struct { int x,y; };
struct mouse_info_struct mouse_info = {-1,-1}, last_mouse;

vector<Point> mousev,kalmanv;

void on_mouse(int event, int x, int y, int flags, void* param)
{
    last_mouse = mouse_info;
    mouse_info.x = x;
    mouse_info.y = y;
}

void DrawCross( Mat & img, Point & center, const Scalar & color, int d )
{
    line( img, Point( center.x - d, center.y - d ), Point( center.x + d, center.y + d ), color, 2, CV_AA, 0 );
    line( img, Point( center.x + d, center.y - d ), Point( center.x - d, center.y + d ), color, 2, CV_AA, 0 );
}

int main (int argc, char * const argv[])
{
    Mat img(500, 500, CV_8UC3);
    KalmanFilter KF(4, 4, 0);
    Mat_<float> state(4, 1); /* (x, y, Vx, Vy) */
    Mat processNoise(4, 1, CV_32F);
    Mat_<float> measurement(4,1); measurement.setTo(Scalar(0));
    char code = (char)-1;

    namedWindow("mouse kalman");
    setMouseCallback("mouse kalman", on_mouse, 0);

    for(;;)
    {
        if (mouse_info.x < 0 || mouse_info.y < 0)
        {
            imshow("mouse kalman", img);
            waitKey(30);
            continue;
        }
        KF.statePre.at<float>(0) = mouse_info.x;
        KF.statePre.at<float>(1) = mouse_info.y;
        KF.statePre.at<float>(2) = 0;
        KF.statePre.at<float>(3) = 0;

        // A : matrix that transform estimate at time t-1 to estimate at time t
        KF.transitionMatrix = *(Mat_<float>(4, 4) << 1,0,1,0,   0,1,0,1,  0,0,1,0,  0,0,0,1);

        // H : matrix that takes a measurement (x,y) and converts it into a state vector (x, y, Vx, Vy )
        setIdentity(KF.measurementMatrix);

        setIdentity(KF.processNoiseCov, Scalar::all(1e-2));
        setIdentity(KF.measurementNoiseCov, Scalar::all(1e-1));
        setIdentity(KF.errorCovPost, Scalar::all(.1));

        mousev.clear();
        kalmanv.clear();

        for(;;)
        {
            // predict new point with physical model
            Mat prediction = KF.predict();
            Point predictPt(prediction.at<float>(0),prediction.at<float>(1));

            // compute measurement
            measurement(0) = mouse_info.x;
            measurement(1) = mouse_info.y;
            measurement(2) = mouse_info.x - last_mouse.x;
            measurement(3) = mouse_info.y - last_mouse.y;

            Point measPt(measurement(0),measurement(1));
            mousev.push_back(measPt);

            Mat estimated = KF.correct(measurement);
            Point statePt(estimated.at<float>(0),estimated.at<float>(1));
            kalmanv.push_back(statePt);

            cout << KF.transitionMatrix << endl;

            // plot points
            img = Scalar::all(0);
            DrawCross( img, statePt, Scalar(255,255,255), 5 );
            DrawCross( img, measPt, Scalar(0,0,255), 5 );
            for (int i = 0; i < mousev.size()-1; i++)
            {
                line(img, mousev[i], mousev[i+1], Scalar(255,255,0), 1);
            }
            for (int i = 0; i < kalmanv.size()-1; i++)
            {
                line(img, kalmanv[i], kalmanv[i+1], Scalar(0,255,0), 1);
            }

            imshow( "mouse kalman", img );
            code = (char)waitKey(100);

            if( code > 0 )
                break;
        }
        if( code == 27 || code == 'q' || code == 'Q' )
            break;
    }

    return 0;
}
