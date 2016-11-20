#include "detector.hpp"


//Using for box sort
struct Info
{
	float score;
	const float* head;
};

bool compare (const Info& Info1, const Info& Info2)
{
	return Info1.score > Info2.score;
}


void Detector::read_classes(const string& path)
{
	std::ifstream file( path );
	if ( file )
	{
		string line;
		stringstream buffer;
		buffer << file.rdbuf();
		file.close();
		while( std::getline(buffer,line,'\n') )
		{
			m_classes.push_back(line);
		}
	}
}

Detector::Detector()
{
	m_initialized = false;
}

bool Detector::initialized()
{
	return m_initialized;
}

void Detector::initialize(const string& model_file, const string& weights_file, const string& labels_file)
{
	read_classes(labels_file);
	net_ = new Net<float>(model_file, caffe::TEST);
	net_->CopyTrainedLayersFrom(weights_file);
	m_initialized = true;
}

void Detector::Detection(const string& im_name)
{
	cv::Mat cv_img = cv::imread(im_name);
	if(cv_img.empty())
	{
			std::cout<<"Can not get the image file !"<<endl;
			return ;
	}
	Detection(cv_img, im_name);
}


void Detector::Detection(cv::Mat cv_img, const string& im_name)
{
	float CONF_THRESH = 0.8;
	float NMS_THRESH = 0.3;
	const int  max_input_side=1000;
	const int  min_input_side=600;

	cv::Mat cv_new(cv_img.rows, cv_img.cols, CV_32FC3, cv::Scalar(0,0,0));
	int max_side = max(cv_img.rows, cv_img.cols);
	int min_side = min(cv_img.rows, cv_img.cols);

	float max_side_scale = float(max_side) / float(max_input_side);
	float min_side_scale = float(min_side) /float( min_input_side);
	float max_scale=max(max_side_scale, min_side_scale);

	float img_scale = 1;

	if(max_scale > 1)
	{
		img_scale = float(1) / max_scale;
	}

	int height = int(cv_img.rows * img_scale);
	int width = int(cv_img.cols * img_scale);
	int num_out;
	cv::Mat cv_resized;

	std::cout<<"imagename "<<im_name<<endl;
	float im_info[3];
	float data_buf[height*width*3];
	float *boxes = NULL;
	float *pred = NULL;
	float *pred_per_class = NULL;
	float *sorted_pred_cls = NULL;
	int *keep = NULL;
	const float* bbox_delt;
	const float* rois;
	const float* pred_cls;
	int num;

	for (int h = 0; h < cv_img.rows; ++h )
	{
		for (int w = 0; w < cv_img.cols; ++w)
		{
			cv_new.at<cv::Vec3f>(cv::Point(w, h))[0] = float(cv_img.at<cv::Vec3b>(cv::Point(w, h))[0])-float(102.9801);
			cv_new.at<cv::Vec3f>(cv::Point(w, h))[1] = float(cv_img.at<cv::Vec3b>(cv::Point(w, h))[1])-float(115.9465);
			cv_new.at<cv::Vec3f>(cv::Point(w, h))[2] = float(cv_img.at<cv::Vec3b>(cv::Point(w, h))[2])-float(122.7717);
		}
	}

	cv::resize(cv_new, cv_resized, cv::Size(width, height));
	im_info[0] = cv_resized.rows;
	im_info[1] = cv_resized.cols;
	im_info[2] = img_scale;

	for (int h = 0; h < height; ++h )
	{
		for (int w = 0; w < width; ++w)
		{
			data_buf[(0*height+h)*width+w] = float(cv_resized.at<cv::Vec3f>(cv::Point(w, h))[0]);
			data_buf[(1*height+h)*width+w] = float(cv_resized.at<cv::Vec3f>(cv::Point(w, h))[1]);
			data_buf[(2*height+h)*width+w] = float(cv_resized.at<cv::Vec3f>(cv::Point(w, h))[2]);
		}
	}

	net_->blob_by_name("data")->Reshape(1, 3, height, width);

	Blob<float> * input_blobs= net_->input_blobs()[0];

	#ifdef CPU_ONLY
		memcpy(input_blobs->mutable_cpu_data(), data_buf, sizeof(float) * input_blobs->count());
	#else
		caffe_gpu_memcpy(sizeof(float)* input_blobs->count(), data_buf, input_blobs->mutable_gpu_data());
	#endif

	// net_->blob_by_name("data")->set_cpu_data(data_buf);
	net_->blob_by_name("im_info")->set_cpu_data(im_info);
	net_->ForwardFrom(0);
	bbox_delt = net_->blob_by_name("bbox_pred")->cpu_data();
	num = net_->blob_by_name("rois")->num();


	rois = net_->blob_by_name("rois")->cpu_data();
	pred_cls = net_->blob_by_name("cls_prob")->cpu_data();
	boxes = new float[num*4];
	pred = new float[num*5*class_num];
	pred_per_class = new float[num*5];
	sorted_pred_cls = new float[num*5];
	keep = new int[num];

	for (int n = 0; n < num; n++)
	{
		for (int c = 0; c < 4; c++)
		{
			boxes[n*4+c] = rois[n*5+c+1] / img_scale;
		}
	}


	bbox_transform_inv(num, bbox_delt, pred_cls, boxes, pred, cv_img.rows, cv_img.cols);
	for (int i = 1; i < class_num; i ++)
	{
		for (int j = 0; j< num; j++)
		{
			for (int k=0; k<5; k++)
				pred_per_class[j*5+k] = pred[(i*num+j)*5+k];
		}
		boxes_sort(num, pred_per_class, sorted_pred_cls);
		#ifdef CPU_ONLY
			std::cout << "TODO: CPU impl of NMS" << endl;
		#else
			_nms(keep, &num_out, sorted_pred_cls, num, 5, NMS_THRESH, 0);
		#endif
		vis_detections(cv_img, keep, num_out, sorted_pred_cls, CONF_THRESH, i);
	}


	cv::imshow("vis",cv_img);
	cv::waitKey(50);

	//string cmd = "mkdir -p res/"+im_name;
	//system(cmd.c_str());
	//cv::imwrite("res/"+im_name,cv_img);
	delete []boxes;
	delete []pred;
	delete []pred_per_class;
	delete []keep;
	delete []sorted_pred_cls;

}

/*
 * ===  FUNCTION  ======================================================================
 *         Name:  vis_detections
 *  Description:  Visuallize the detection result
 * =====================================================================================
 */
void Detector::vis_detections(cv::Mat image, int* keep, int num_out, float* sorted_pred_cls, float CONF_THRESH, int nclass)
{
	int i=0;
	cv::Scalar color;
	color = cv::Scalar(0,0,0);
	if (nclass == 0) color = cv::Scalar(255,0,0);
	if (nclass == 1) color = cv::Scalar(0,255,0);
	if (nclass == 2) color = cv::Scalar(0,0,255);
	if (nclass == 3) color = cv::Scalar(255,255,0);
	if (nclass == 4) color = cv::Scalar(255,0,255);
	if (nclass == 5) color = cv::Scalar(0,255,255);
	if (nclass == 6) color = cv::Scalar(255,255,255);
	if (nclass == 7) color = cv::Scalar(128,255,0);
	if (nclass == 8) color = cv::Scalar(255,128,0);
	if (nclass == 9) color = cv::Scalar(255,0,128);
	if (nclass == 10) color = cv::Scalar(128,0,255);
	if (nclass == 11) color = cv::Scalar(0,128,255);
	if (nclass == 12) color = cv::Scalar(128,128,128);
	if (nclass == 13) color = cv::Scalar(50,0,0);
	if (nclass == 14) color = cv::Scalar(50,50,0);
	if (nclass == 15) color = cv::Scalar(50,0,50);
	if (nclass == 16) color = cv::Scalar(0,50,50);
	if (nclass == 17) color = cv::Scalar(0,50,0);
	if (nclass == 18) color = cv::Scalar(0,0,50);
	if (nclass == 19) color = cv::Scalar(180,0,0);
	if (nclass == 20) color = cv::Scalar(0,180,0);

	while(sorted_pred_cls[keep[i]*5+4]>CONF_THRESH && i < num_out)
	{
		if(i>=num_out)
			return;
		cv::rectangle(image,cv::Point(sorted_pred_cls[keep[i]*5+0], sorted_pred_cls[keep[i]*5+1]),cv::Point(sorted_pred_cls[keep[i]*5+2], sorted_pred_cls[keep[i]*5+3]),color,2);

		cv::Size ts = cv::getTextSize(m_classes[nclass], 5, 0.8, 1, NULL);
		cv::putText(image, m_classes[nclass], cv::Point(sorted_pred_cls[keep[i]*5+0], sorted_pred_cls[keep[i]*5+1]), 5, 0.8, cv::Scalar(0,0,0), 1, CV_AA);

		i++;
	}
}

/*
 * ===  FUNCTION  ======================================================================
 *         Name:  boxes_sort
 *  Description:  Sort the bounding box according score
 * =====================================================================================
 */
void Detector::boxes_sort(const int num, const float* pred, float* sorted_pred)
{
	vector<Info> my;
	Info tmp;
	for (int i = 0; i< num; i++)
	{
		tmp.score = pred[i*5 + 4];
		tmp.head = pred + i*5;
		my.push_back(tmp);
	}
	std::sort(my.begin(), my.end(), compare);
	for (int i=0; i<num; i++)
	{
		for (int j=0; j<5; j++)
			sorted_pred[i*5+j] = my[i].head[j];
	}
}

/*
 * ===  FUNCTION  ======================================================================
 *         Name:  bbox_transform_inv
 *  Description:  Compute bounding box regression value
 * =====================================================================================
 */
void Detector::bbox_transform_inv(int num, const float* box_deltas, const float* pred_cls, float* boxes, float* pred, int img_height, int img_width)
{
	float width, height, ctr_x, ctr_y, dx, dy, dw, dh, pred_ctr_x, pred_ctr_y, pred_w, pred_h;
	for(int i=0; i< num; i++)
	{
		width = boxes[i*4+2] - boxes[i*4+0] + 1.0;
		height = boxes[i*4+3] - boxes[i*4+1] + 1.0;
		ctr_x = boxes[i*4+0] + 0.5 * width;
		ctr_y = boxes[i*4+1] + 0.5 * height;
		for (int j=0; j< class_num; j++)
		{
			dx = box_deltas[(i*class_num+j)*4+0];
			dy = box_deltas[(i*class_num+j)*4+1];
			dw = box_deltas[(i*class_num+j)*4+2];
			dh = box_deltas[(i*class_num+j)*4+3];
			pred_ctr_x = ctr_x + width*dx;
			pred_ctr_y = ctr_y + height*dy;
			pred_w = width * exp(dw);
			pred_h = height * exp(dh);
			pred[(j*num+i)*5+0] = max(min(pred_ctr_x - 0.5* pred_w, img_width -1), 0);
			pred[(j*num+i)*5+1] = max(min(pred_ctr_y - 0.5* pred_h, img_height -1), 0);
			pred[(j*num+i)*5+2] = max(min(pred_ctr_x + 0.5* pred_w, img_width -1), 0);
			pred[(j*num+i)*5+3] = max(min(pred_ctr_y + 0.5* pred_h, img_height -1), 0);
			pred[(j*num+i)*5+4] = pred_cls[i*class_num+j];
		}
	}
}
