#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <vector>

using namespace cv;
using namespace std;

void faceDetect(Mat &img)
{
    Mat img_gray;
    cvtColor(img, img_gray, COLOR_BGR2GRAY);

    CascadeClassifier faceCascade {"Resources/haarcascade_frontalface_default.xml"};

    vector<Rect> faces;
    faceCascade.detectMultiScale(img_gray, faces, 1.1, 10);

    for(int i = 0; i < faces.size(); i++)
    {
        rectangle(img, faces[i].tl(), faces[i].br(), Scalar(255, 0, 255), 3);
    }
}

void faceBox(const Mat &img, vector<Rect> &bbox)
{
    CascadeClassifier faceCascade {"Resources/haarcascade_frontalface_default.xml"};

    faceCascade.detectMultiScale(img, bbox, 1.1, 10);
}

int main()
{

    VideoCapture cap(1);
    Mat img;
    vector<Rect> bbox;
    Ptr<Tracker> tracker = TrackerCSRT::create();

    cap.read(img);

    faceBox(img, bbox);

    rectangle(img, bbox[0].tl(), bbox[0].br(), Scalar(255, 0, 255), 3);

    tracker->init(img, bbox[0]);

    while(true)
    {
        cap.read(img);

        tracker->update(img, bbox[0]);
        rectangle(img, bbox[0], Scalar(255, 0, 255), 3);

        faceDetect(img);

        imshow("VIDEO", img);

        if(waitKey(1) == 27)
            return 1;
    }

    cout << CV_VERSION << '\n';
	
    return 0;
}