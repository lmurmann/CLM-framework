///////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2014, University of Southern California and University of Cambridge,
// all rights reserved.
//
// THIS SOFTWARE IS PROVIDED “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES,
// INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
// AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL
// THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
// INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY. OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
// ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Notwithstanding the license granted herein, Licensee acknowledges that certain components
// of the Software may be covered by so-called “open source” software licenses (“Open Source
// Components”), which means any software licenses approved as open source licenses by the
// Open Source Initiative or any substantially similar licenses, including without limitation any
// license that, as a condition of distribution of the software licensed under such license,
// requires that the distributor make the software available in source code format. Licensor shall
// provide a list of Open Source Components for a particular version of the Software upon
// Licensee’s request. Licensee will comply with the applicable terms of such licenses and to
// the extent required by the licenses covering Open Source Components, the terms of such
// licenses will apply in lieu of the terms of this Agreement. To the extent the terms of the
// licenses applicable to Open Source Components prohibit any of the restrictions in this
// License Agreement with respect to such Open Source Component, such restrictions will not
// apply to such Open Source Component. To the extent the terms of the licenses applicable to
// Open Source Components require Licensor to make an offer to provide source code or
// related information in connection with the Software, such offer is hereby made. Any request
// for source code or related information should be directed to cl-face-tracker-distribution@lists.cam.ac.uk
// Licensee acknowledges receipt of notices for the Open Source Components for the initial
// delivery of the Software.

//     * Any publications arising from the use of this software, including but
//       not limited to academic journal and conference publications, technical
//       reports and manuals, must cite one of the following works:
//
//       Tadas Baltrusaitis, Peter Robinson, and Louis-Philippe Morency. 3D
//       Constrained Local Model for Rigid and Non-Rigid Facial Tracking.
//       IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2012.    
//
//       Tadas Baltrusaitis, Peter Robinson, and Louis-Philippe Morency. 
//       Constrained Local Neural Fields for robust facial landmark detection in the wild.
//       in IEEE Int. Conference on Computer Vision Workshops, 300 Faces in-the-Wild Challenge, 2013.    
//
///////////////////////////////////////////////////////////////////////////////

// SimpleCLM.cpp : Defines the entry point for the console application.
#include "SimpleCLM.h"
#include "CLM_core.h"

#include <fstream>
#include <sstream>

#include <opencv2/videoio/videoio.hpp>  // Video write
#include <opencv2/videoio/videoio_c.h>  // Video write


#define INFO_STREAM( stream ) \
std::cout << stream << std::endl

#define WARN_STREAM( stream ) \
std::cout << "Warning: " << stream << std::endl

#define ERROR_STREAM( stream ) \
std::cout << "Error: " << stream << std::endl

static void printErrorAndAbort( const std::string & error )
{
    std::cout << error << std::endl;
    abort();
}

#define FATAL_STREAM( stream ) \
printErrorAndAbort( std::string( "Fatal error: " ) + stream )

using namespace std;
using namespace cv;

// clients of SimpleCLM only see opaque pointer.
struct SimpleCLM {
	CLMTracker::CLMParameters clm_parameters;
	CLMTracker::CLM clm_model;	
	double detection_certainty = 0.0;
	float fx = 0;
	float fy = 0; 
	float cx = 0; 
	float cy = 0;
	float pos[3];
	float rot[3];
	int frame_count = 0;
};
vector<string> get_arguments(int argc, char **argv)
{

	vector<string> arguments;

	for(int i = 0; i < argc; ++i)
	{
		arguments.push_back(string(argv[i]));
	}
	return arguments;
}

// Some globals for tracking timing information for visualisation
double fps_tracker = -1.0;
int64 t0 = 0;

// Visualising the results
void visualise_tracking(Mat& captured_image, Mat_<float>& depth_image, SimpleCLM* sclm)
{

	// Drawing the facial landmarks on the face and the bounding box around it if tracking is successful and initialised
	double detection_certainty = sclm->clm_model.detection_certainty;
	bool detection_success = sclm->clm_model.detection_success;

	double visualisation_boundary = 0.2;

	// Only draw if the reliability is reasonable, the value is slightly ad-hoc
	if (detection_certainty < visualisation_boundary)
	{
		CLMTracker::Draw(captured_image, sclm->clm_model);

		double vis_certainty = detection_certainty;
		if (vis_certainty > 1)
			vis_certainty = 1;
		if (vis_certainty < -1)
			vis_certainty = -1;

		vis_certainty = (vis_certainty + 1) / (visualisation_boundary + 1);

		// A rough heuristic for box around the face width
		int thickness = (int)std::ceil(2.0* ((double)captured_image.cols) / 640.0);

		Vec6d pose_estimate_to_draw = CLMTracker::GetCorrectedPoseWorld(sclm->clm_model, sclm->fx, sclm->fy, sclm->cx, sclm->cy);

		// Draw it in reddish if uncertain, blueish if certain
		CLMTracker::DrawBox(captured_image, pose_estimate_to_draw, Scalar((1 - vis_certainty)*255.0, 0, vis_certainty * 255), thickness, sclm->fx, sclm->fy, sclm->cx, sclm->cy);

	}

	// Work out the framerate
	if (sclm->frame_count % 10 == 0)
	{
		double t1 = cv::getTickCount();
		fps_tracker = 10.0 / (double(t1 - t0) / cv::getTickFrequency());
		t0 = t1;
	}

	// Write out the framerate on the image before displaying it
	char fpsC[255];
	std::sprintf(fpsC, "%d", (int)fps_tracker);
	string fpsSt("FPS:");
	fpsSt += fpsC;
	int fontface = CV_FONT_HERSHEY_SIMPLEX;
	float fontscale = 1.3f;
	int thickness = 3;
	int baseline;
	cv::Size textsz = cv::getTextSize(fpsSt, fontface, fontscale, thickness, &baseline);
	//textsz.height -= baseline;
	textsz.width = 5;
	textsz.height += 5;
	cv::putText(captured_image, fpsSt, textsz, fontface, fontscale, CV_RGB(255, 0, 0), thickness);

	#if 0
	if (!sclm->clm_parameters.quiet_mode)
	{
		namedWindow("tracking_result", 1);
		imshow("tracking_result", captured_image);

		if (!depth_image.empty())
		{
			// Division needed for visualisation purposes
			imshow("depth", depth_image / 2000.0);
		}

	}
	waitKey(1);
	#endif
}

int SimpleCLM_initStream(VideoCapture* video_capture, int argc, char** argv) {
	vector<string> arguments = get_arguments(argc, argv);
	bool use_world_coordinates;
	vector<string> files, depth_directories, pose_output_files, tracked_videos_output, landmark_output_files, landmark_3D_output_files;
	CLMTracker::get_video_input_output_params(files, depth_directories, pose_output_files, tracked_videos_output, landmark_output_files, landmark_3D_output_files, use_world_coordinates, arguments);
	// By default try webcam 0
	int device = 0;

	
	bool streaming = false;
	if (files.size() == 0) {
		INFO_STREAM("NO VIDEO FILES");
		streaming = true;
	} else if (files.size() > 1) {
		INFO_STREAM("MULTIPLE VIDEO FILES SPECIFIED. IGNORING ALL EXCEPT FIRST.");
	}



	double fps_vid_in = -1.0;

	// Do some grabbing
	if(!streaming)
	{
		string current_file = files[0];
		INFO_STREAM( "Attempting to read from file: " << current_file );
		*video_capture = VideoCapture( current_file );
		fps_vid_in = video_capture->get(CV_CAP_PROP_FPS);
		
		// Check if fps is nan or less than 0
		if (fps_vid_in != fps_vid_in || fps_vid_in <= 0)
		{
			INFO_STREAM("FPS of the video file cannot be determined, assuming 30");
			fps_vid_in = 30;
		}
	}
	else
	{
		INFO_STREAM( "Attempting to capture from device: " << device );
		*video_capture = VideoCapture( device );
	}

	if( !video_capture->isOpened() ) FATAL_STREAM( "Failed to open video source" );
	else INFO_STREAM( "Device or file opened");

	return 0;
}
void SimpleCLM_initIntrinsics(SimpleCLM* sclm, int videow, int videoh) {
	// If optical centers are not defined just use center of image
	sclm->cx = videow / 2.0f;
	sclm->cy = videoh / 2.0f;
	// Use a rough guess-timate of focal length
	sclm->fx = 500 * (videow / 640.0);
	sclm->fy = 500 * (videoh / 480.0);

	sclm->fx = (sclm->fx + sclm->fy) / 2.0;
	sclm->fy = sclm->fx;
}
void SimpleCLM_setFocalLength(SimpleCLM* sclm, float cx, float cy, float fx, float fy) {
	sclm->cx = cx;
	sclm->cy = cy;
	sclm->fx = fx;
	sclm->fy = fy;
}
// FIXME no cv in this signature. Do pointer / width / height
void SimpleCLM_processFrame(SimpleCLM* sclm, uchar* data, int w, int h) {
	Mat_<uchar> grayscale_image(h, w, data);
	Mat_<float> depth_image;

	// The actual facial landmark detection / tracking
	bool detection_success = CLMTracker::DetectLandmarksInVideo(grayscale_image, depth_image, sclm->clm_model, sclm->clm_parameters);

	// Work out the pose of the head from the tracked model
	Vec6d pose_estimate_CLM = CLMTracker::GetCorrectedPoseCamera(sclm->clm_model, sclm->fx, sclm->fy, sclm->cx, sclm->cy);

	// Visualising the results
	// Drawing the facial landmarks on the face and the bounding box around it if tracking is successful and initialised
	sclm->detection_certainty = sclm->clm_model.detection_certainty;

	Vec6d pose_estimate_to_draw = CLMTracker::GetCorrectedPoseWorld(sclm->clm_model, sclm->fx, sclm->fy, sclm->cx, sclm->cy);
	sclm->pos[0] = pose_estimate_to_draw[0];
	sclm->pos[1] = pose_estimate_to_draw[1];
	sclm->pos[2] = pose_estimate_to_draw[2];

	sclm->rot[0] = pose_estimate_to_draw[3];
	sclm->rot[1] = pose_estimate_to_draw[4];
	sclm->rot[2] = pose_estimate_to_draw[5];


	// Update the frame count
	sclm->frame_count++;
}
void SimpleCLM_visualize(SimpleCLM* sclm, uint8_t* data, int w, int h) {
	Mat_<float> depth_image;
	Mat_<Vec3b> captured_image(h, w, (Vec3b*)data);
	visualise_tracking(captured_image, depth_image, sclm);
}
void SimpleCLM_getPose(SimpleCLM* sclm, float* pos, float* rot) {
	for (int i = 0; i < 3; ++i) {
		pos[i] = sclm->pos[i];
		rot[i] = sclm->rot[i];
	}

}
int SimpleCLM_initModel(SimpleCLM* sclm, int argc, char** argv) {
	vector<string> arguments = get_arguments(argc, argv);
	// Some initial parameters that can be overriden from command line	
	sclm->clm_parameters = CLMTracker::CLMParameters(arguments);
	// The modules that are being used for tracking
	sclm->clm_model = CLMTracker::CLM(sclm->clm_parameters.model_location);
	for (int i = 0 ; i < 3 ; ++i) {
		sclm->pos[i] = 0.0f;
		sclm->rot[i] = 0.0f;
	}
	return 0;
}
SimpleCLM* SimpleCLM_create() {
	return new SimpleCLM();
}
CLMAPI void SimpleCLM_free(SimpleCLM* sclm) {
	delete sclm;
}


int SimpleCLM_run (int argc, char **argv)
{
	SimpleCLM* sclm = SimpleCLM_create();
	SimpleCLM_initModel(sclm, argc, argv);

	VideoCapture video_capture;
	SimpleCLM_initStream(&video_capture, argc, argv);

	Mat captured_image;
	video_capture >> captured_image;		
	SimpleCLM_initIntrinsics(sclm, captured_image.cols, captured_image.rows);

	// END INIT
	bool done = false;	
	INFO_STREAM( "Starting tracking");
	while(!done && !captured_image.empty()) // this is not a for loop as we might also be reading from a webcam
	{
		// Use for timestamping if using a webcam
		int64 t_initial = cv::getTickCount();

		// Timestamp in seconds of current processing
		int fps_vid_in = 30;

		// Reading the images
		Mat_<float> depth_image;
		Mat_<uchar> grayscale_image;

		if(captured_image.channels() == 3)
		{
			cvtColor(captured_image, grayscale_image, CV_BGR2GRAY);				
		}
		else
		{
			grayscale_image = captured_image.clone();				
		}
	
		SimpleCLM_processFrame(sclm, grayscale_image.data, grayscale_image.cols, grayscale_image.rows);
		int elemsize = captured_image.elemSize();
		SimpleCLM_visualize(sclm, captured_image.data, captured_image.cols, captured_image.rows);

		// Output the detected facial landmarks
		#if 0
		if(!landmark_output_files.empty())
		{
			double confidence = 0.5 * (1 - clm_model.detection_certainty);
			landmarks_output_file << frame_count + 1 << ", " << time_stamp << ", " << confidence << ", " << detection_success;
			for (int i = 0; i < clm_model.pdm.NumberOfPoints() * 2; ++ i)
			{
				landmarks_output_file << ", " << clm_model.detected_landmarks.at<double>(i);
			}
			landmarks_output_file << endl;
		}

		// Output the detected facial landmarks
		if(!landmark_3D_output_files.empty())
		{
			double confidence = 0.5 * (1 - clm_model.detection_certainty);
			landmarks_3D_output_file << frame_count + 1 << ", " << time_stamp << ", " << confidence << ", " << detection_success;
			Mat_<double> shape_3D = clm_model.GetShape(fx, fy, cx, cy);
			for (int i = 0; i < clm_model.pdm.NumberOfPoints() * 3; ++i)
			{
				landmarks_3D_output_file << ", " << shape_3D.at<double>(i);
			}
			landmarks_3D_output_file << endl;
		}

		// Output the estimated head pose
		if(!pose_output_files.empty())
		{
			double confidence = 0.5 * (1 - clm_model.detection_certainty);
			pose_output_file << frame_count + 1 << ", " << time_stamp << ", " << confidence << ", " << detection_success
				<< ", " << pose_estimate_CLM[0] << ", " << pose_estimate_CLM[1] << ", " << pose_estimate_CLM[2]
				<< ", " << pose_estimate_CLM[3] << ", " << pose_estimate_CLM[4] << ", " << pose_estimate_CLM[5] << endl;
		}

		// output the tracked video
		if(!tracked_videos_output.empty())
		{		
			writerFace << captured_image;
		}
		#endif

		video_capture >> captured_image;
	
		// detect key presses
		char character_press = cv::waitKey(1);
		
		// restart the tracker
		if(character_press == 'r')
		{
			sclm->clm_model.Reset();
		}
		// quit the application
		else if(character_press=='q')
		{
			done = true;
		}

	}

	SimpleCLM_free(sclm);
	return 0;
}

