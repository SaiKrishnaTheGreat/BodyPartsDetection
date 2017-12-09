//DetectBodyParts project main.cpp file

//Created by Sai Krishna Dyavarashetty

#include <opencv2/opencv.hpp>

#include <iostream>
#include <stdio.h>

#include "detectBodyParts.h"

using namespace cv;
using namespace std;

#define ENABLE_FACE_DETECTION 0
#define ENABLE_EYE_DETECTION 0
#define ENABLE_EAR_DETECTION 0
#define ENABLE_MOUTH_DETECTION 0
#define ENABLE_NOSE_DETECTION 0
#define ENABLE_SMILE_DETECTION 1
#define ENABLE_LOWERBODY_DETECTION 0
#define ENABLE_UPPERBODY_DETECTION 0
#define ENABLE_FULLBODY_DETECTION 0

int main(int argc, char* argv[])
{
	Mat frame;
	VideoCapture cap;
	cap.open(0);
	if(!cap.isOpened())
	{
		cout << "Can not Open Video Cam "<< endl;
		return -1;
	}
	while(true)
	{
		cap.read(frame);
		flip(frame,frame,1);
		float resizeFactor=0.5;
		resize(frame, frame, Size(), resizeFactor, resizeFactor, INTER_AREA);
		if( !frame.empty() )
        {
            if(ENABLE_FACE_DETECTION) detectFace(frame);
            if(ENABLE_EYE_DETECTION) detectEyes(frame);
            if(ENABLE_EAR_DETECTION) detectEars(frame);
            if(ENABLE_MOUTH_DETECTION) detectMouth(frame);
            if(ENABLE_NOSE_DETECTION) detectNose(frame);
            if(ENABLE_SMILE_DETECTION) detectSmile(frame);
            if(ENABLE_LOWERBODY_DETECTION) detectLowerBody(frame);
            if(ENABLE_UPPERBODY_DETECTION) detectUpperBody(frame);
            if(ENABLE_FULLBODY_DETECTION) detectFullBody(frame);
		}
		else
		{
			cout << " No frame is available:Exit" << endl;
			break;
		}
		int c = waitKey(5);
		if((char)c==27)break;
	}
	cap.release();
	return 0 ;
}



