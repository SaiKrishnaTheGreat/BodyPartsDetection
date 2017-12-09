//Header file for detectBodyParts.cpp

//Created by Sai Krishna Dyavarashetty

#include <opencv2/opencv.hpp>

#include <iostream>
#include <stdio.h>

using namespace cv;
using namespace std;

// Show Frame
void show(String winname,Mat &frame);

// Face detection
void detectFace( Mat frame );

// Eye detection
void detectEyes( Mat frame );

// Ear detection
void detectEars( Mat frame );

// Mouth detection
void detectMouth( Mat frame );

// Nose detection
void detectNose( Mat frame );

// Smile detection
void detectSmile( Mat frame );

// Upper body detection
void detectUpperBody( Mat frame );

// Lower Body detection
void detectLowerBody( Mat frame );

// Full Body detection
void detectFullBody( Mat frame );