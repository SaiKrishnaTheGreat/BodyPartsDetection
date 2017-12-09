//DetectBodyParts project detectBodyParts.cpp file

//Created by Sai Krishna Dyavarashetty

#include "detectBodyParts.h"

using namespace std;
using namespace cv;

Mat frame_gray;

CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
CascadeClassifier leftear_cascade;
CascadeClassifier rightear_cascade;
CascadeClassifier mouth_cascade;
CascadeClassifier nose_cascade;
CascadeClassifier smile_cascade;
CascadeClassifier lowerbody_cascade;
CascadeClassifier upperbody_cascade;
CascadeClassifier fullbody_cascade;

void show(String winname, Mat &frame)
{
	namedWindow(winname,WINDOW_NORMAL);
	imshow(winname,frame);
}

// Face detection
void detectFace( Mat frame ){

	////////Uncomment One of the below line/////Change the path for lbpCascades///////

	//face_cascade.load("./haarcascades/haarcascade_frontalcatface.xml");
    //face_cascade.load("./haarcascades/haarcascade_frontalcatface_extended.xml");
    //face_cascade.load("./haarcascades/haarcascade_frontalface_alt.xml");
    //face_cascade.load("./haarcascades/haarcascade_frontalface_alt2.xml");
    //face_cascade.load("./haarcascades/haarcascade_frontalface_alt_tree.xml");
    //face_cascade.load("./haarcascades/haarcascade_frontalface_default.xml");
    //face_cascade.load("./haarcascades/sk_haarcascade_frontalcatface.xml");
    //face_cascade.load("./haarcascades/sk_haarcascade_frontalcatface_extended.xml");
    face_cascade.load("./haarcascades/sk_haarcascade_frontalface_alt.xml");
    //face_cascade.load("./haarcascades/sk_haarcascade_frontalface_alt2.xml");
    //face_cascade.load("./haarcascades/sk_haarcascade_frontalface_alt_tree.xml");
    //face_cascade.load("./haarcascades/sk_haarcascade_frontalface_default.xml");

    vector<Rect> faces;
    Mat frame_gray;
    
    cvtColor( frame, frame_gray, CV_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );
    
    //Detect faces
	face_cascade.detectMultiScale( frame_gray, faces, 1.3, 3, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30), Size(200,200));
	for( size_t i = 0; i < faces.size(); i++ )
    {
    	rectangle(frame,faces[i].tl(),faces[i].br(),Scalar(255,255,255),2);
	}
	show("ResultWindow",frame);

}

// Eye detection
void detectEyes( Mat frame ){

	face_cascade.load("./haarcascades/sk_haarcascade_frontalface_alt.xml"); //Change the path for other file
	eyes_cascade.load("./haarcascades/haarcascade_eye_tree_eyeglasses.xml"); //Change the path for other file

	vector<Rect> faces;

    cvtColor( frame, frame_gray, CV_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );
    
    // Detect faces
    face_cascade.detectMultiScale( frame_gray, faces, 1.3, 3, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30), Size(200,200));
    
    for( size_t i = 0; i < faces.size(); i++ )
    {
        Mat faceROI = frame_gray( faces[i] );
        std::vector<Rect> eyes;
        
        // In each face, detect eyes
        eyes_cascade.detectMultiScale( faceROI, eyes, 1.3, 3, 0 |CV_HAAR_SCALE_IMAGE, Size(30, 30),Size(100,100));
        
        for( size_t j = 0; j < eyes.size(); j++ )
        {
            Point center( faces[i].x + eyes[j].x + eyes[j].width*0.5, faces[i].y + eyes[j].y + eyes[j].height*0.5 );
            int radius = cvRound( (eyes[j].width + eyes[j].height)*0.25 );
            circle( frame, center, radius, Scalar( 255, 255, 255 ), 4, 8, 0 );
        }
	}
	show("ResultWindow",frame);
}

// Ear detection
void detectEars( Mat frame ){

	leftear_cascade.load("./haarcascades/haarcascade_mcs_leftear.xml");
	rightear_cascade.load("./haarcascades/haarcascade_mcs_rightear.xml");

	vector<Rect> leftears, rightears;

    cvtColor( frame, frame_gray, CV_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );
    
    // Detect left ears
    leftear_cascade.detectMultiScale( frame_gray, leftears, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );
    
    // Detect right ears
    rightear_cascade.detectMultiScale( frame_gray, rightears, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );
    
    for( size_t i = 0; i < leftears.size(); i++ )
    {
        Point center( leftears[i].x + leftears[i].width*0.5, leftears[i].y + leftears[i].height*0.5 );
        ellipse( frame, center, Size( leftears[i].width*0.5, leftears[i].height*0.5), 0, 0, 360, Scalar( 255, 255, 0 ), 4, 8, 0 );
    }
    
    for( size_t i = 0; i < rightears.size(); i++ )
    {
        Point center( rightears[i].x + rightears[i].width*0.5, rightears[i].y + rightears[i].height*0.5 );
        ellipse( frame, center, Size( rightears[i].width*0.5, rightears[i].height*0.5), 0, 0, 360, Scalar( 0, 255, 0 ), 4, 8, 0 );
	}

	show("ResultWindow",frame);

}

// Mouth detection
void detectMouth( Mat frame ){

	mouth_cascade.load("./haarcascades/haarcascade_mcs_mouth.xml");

	vector<Rect> mouths;
  
    cvtColor( frame, frame_gray, CV_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );
    
    // Detect mouth
    mouth_cascade.detectMultiScale( frame_gray, mouths, 1.3, 10, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );
    
    for( size_t i = 0; i < mouths.size(); i++ )
    {
        Point center( mouths[i].x + mouths[i].width*0.5, mouths[i].y + mouths[i].height*0.5 );
        ellipse( frame, center, Size( mouths[i].width*0.5, mouths[i].height*0.5), 0, 0, 360, Scalar( 255, 255, 255 ), 4, 8, 0 );
    }
    
	show( "ResultWindow", frame );

}

// Nose detection
void detectNose( Mat frame ){

	nose_cascade.load("./haarcascades/haarcascade_mcs_nose.xml");

	vector<Rect> noses;
    
    cvtColor( frame, frame_gray, CV_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );
    
    // Detect noses
    nose_cascade.detectMultiScale( frame_gray, noses, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );
    
    for( size_t i = 0; i < noses.size(); i++ )
    {
        Point center( noses[i].x + noses[i].width*0.5, noses[i].y + noses[i].height*0.5 );
        ellipse( frame, center, Size( noses[i].width*0.5, noses[i].height*0.5), 0, 0, 360, Scalar( 255, 255, 255 ), 4, 8, 0 );
    }
    
    show( "ResultWindow", frame );

}

// Smile detection
void detectSmile( Mat frame ){

	smile_cascade.load("haarcascades/haarcascade_smile.xml");

	vector<Rect> smiles;
    Mat frame_gray;
    
    cvtColor( frame, frame_gray, CV_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );
    
    // Detect faces
    smile_cascade.detectMultiScale( frame_gray, smiles, 1.3, 10, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );
    
    for( size_t i = 0; i < smiles.size(); i++ )
    {
        Point center( smiles[i].x + smiles[i].width*0.5, smiles[i].y + smiles[i].height*0.5 );
        ellipse( frame, center, Size( smiles[i].width*0.5, smiles[i].height*0.5), 0, 0, 360, Scalar( 255, 255, 255 ), 4, 8, 0 );
    }
    
	show( "ResultWindow", frame );
}

// Upper body detection
void detectUpperBody( Mat frame ){

	upperbody_cascade.load("./haarcascades/haarcascade_upperbody.xml");

	vector<Rect> upperbody;
    Mat frame_gray;
    
    cvtColor( frame, frame_gray, CV_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );
    
    // Detect faces
    upperbody_cascade.detectMultiScale( frame_gray, upperbody, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(100, 100) );
    
    for( size_t i = 0; i < upperbody.size(); i++ )
    {
        Rect temp = upperbody[i];
        temp.y += 100;
        rectangle(frame, temp, Scalar(255,255,255), 4, 8);
    }
    
	show("ResultWindow", frame );

}

// Lower Body detection
void detectLowerBody( Mat frame ){
	lowerbody_cascade.load("./haarcascades/haarcascade_lowerbody.xml");

	vector<Rect> lowerbody;
    Mat frame_gray;
    
    cvtColor( frame, frame_gray, CV_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );
    
    // Detect faces
    lowerbody_cascade.detectMultiScale( frame_gray, lowerbody, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(100, 100) );
    
    for( size_t i = 0; i < lowerbody.size(); i++ )
    {
        Point center( lowerbody[i].x + lowerbody[i].width*0.5, lowerbody[i].y + lowerbody[i].height*0.5 );
        ellipse( frame, center, Size( lowerbody[i].width*0.5, lowerbody[i].height*0.5), 0, 0, 360, Scalar( 255, 255, 255 ), 4, 8, 0 );
    }
    
	show( "ResultWindow", frame );

}

// Full Body detection
void detectFullBody( Mat frame ){

	fullbody_cascade.load("./haarcascades/haarcascade_fullbody.xml");
	vector<Rect> fullbody;
    Mat frame_gray;
    
    cvtColor( frame, frame_gray, CV_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );
    
    // Detect faces
    fullbody_cascade.detectMultiScale( frame_gray, fullbody, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(100, 100) );
    
    for( size_t i = 0; i < fullbody.size(); i++ )
    {
        Point center( fullbody[i].x + fullbody[i].width*0.5, fullbody[i].y + fullbody[i].height*0.5 );
        ellipse( frame, center, Size( fullbody[i].width*0.5, fullbody[i].height*0.5), 0, 0, 360, Scalar( 255, 255, 255 ), 4, 8, 0 );
    }
    
    show( "ResultWindow", frame );
}

