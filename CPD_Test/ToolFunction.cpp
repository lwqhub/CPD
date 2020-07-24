#include "pch.h"
#include "ToolFunction.h"

#include <iostream>   
#include <fstream>   
#include <iomanip>   
#include <set>   
#include <vector>   
#include <string>   

#include <pcl/io/pcd_io.h>

using namespace std;

ToolFunction::ToolFunction()
{
}


ToolFunction::~ToolFunction()
{
}

bool ToolFunction::convertFromSTLToPLY(std::string filePath)
{
	class Vertex
	{
	private:
		float x, y, z;
	public:
		Vertex(float x1, float y1, float z1) :x(x1), y(y1), z(z1) {}
		int Num;
		bool operator<(const Vertex&s)const
		{
			if (x < s.x) return true;
			if ((x == s.x) && (y < s.y)) return true;
			return false;
		}
		float getX() { return x; }
		float getY() { return y; }
		float getZ() { return z; }
	};

	ifstream stl(filePath);
	if (!stl)
	{
		cerr << "Open stl file error!" << endl;
		return false;
	}
	filePath.replace((int)filePath.rfind(".") + 1, 3, "ply");
	ofstream ply(filePath);
	if (!ply)
	{
		cerr << "Open ply file error!" << endl;
		return false;
	}

	vector<Vertex>Vert_temp;
	string s;
	stl >> s;
	while (!stl.eof())
	{
		if (s == "vertex")
		{
			float x_t, y_t, z_t;
			stl >> x_t >> y_t >> z_t;
			Vertex ver(x_t, y_t, z_t);
			Vert_temp.push_back(ver);
		}
		stl >> s;
	}

	int face_num = Vert_temp.size() / 3;

	set<Vertex>Vert_out(Vert_temp.begin(), Vert_temp.end());

	ply << setiosflags(ios::fixed) << setprecision(10);
	ply << "ply" << endl;
	ply << "format ascii 1.0" << endl;
	ply << "comment Write By mightBXG" << endl;
	ply << "element vertex " << Vert_out.size() << endl;
	ply << "property float x" << endl;
	ply << "property float y" << endl;
	ply << "property float z" << endl;
	ply << "element face " << face_num << endl;
	ply << "property list uchar int vertex_indices" << endl;
	ply << "end_header" << endl;
	set<Vertex>::iterator it_set = Vert_out.begin();
	int i = 0;
	while (it_set != Vert_out.end())
	{
		const_cast<Vertex &>(*it_set).Num = i;
		it_set++; i++;
	}

	it_set = Vert_out.begin();
	while (it_set != Vert_out.end())
	{
		ply << const_cast<Vertex &>(*it_set).getX() << " " << const_cast<Vertex &>(*it_set).getY() << " " << const_cast<Vertex &>(*it_set).getZ() << endl;
		it_set++;
	}
	vector<Vertex>::iterator it_vec = Vert_temp.begin();
	while (it_vec != Vert_temp.end())
	{
		ply << "3";
		for (i = 0; i < 3; i++)
		{
			ply << " " << (*(Vert_out.find(*it_vec))).Num;
			it_vec++;
		}
		ply << endl;
	}

	stl.close();
	ply.close();

	return true;
}

bool ToolFunction::convertFromTXTToPCD(std::string filePath)
{
	std::vector<double> txtPoints;
	ifstream fin(filePath.c_str());
	if (fin)
	{
		double temp;
		while (fin >> temp)
		{
			txtPoints.push_back(temp);
		}
	}
	pcl::PointCloud<pcl::PointXYZ> cloud;
	cloud.width = txtPoints.size() / 3;
	cloud.height = 1;
	cloud.points.resize(cloud.width * cloud.height);
	for (int j = 0; j < cloud.width; j++)
	{
		cloud.points[j].x = txtPoints[3 * j];
		cloud.points[j].y = txtPoints[3 * j + 1];
		cloud.points[j].z = txtPoints[3 * j + 2];
	}
	filePath.replace((int)filePath.rfind(".") + 1, 3, "pcd");
	pcl::io::savePCDFileASCII(filePath, cloud);
	return true;
}

void ToolFunction::downSampling(PointCloudT::Ptr cloud_input, PointCloudT::Ptr cloud_output, float sizeX, float sizeY, float sizeZ)
{
	pcl::VoxelGrid<pcl::PointXYZ> sor;
	sor.setInputCloud(cloud_input);
	sor.setLeafSize(sizeX, sizeY, sizeZ);
	sor.filter(*cloud_output);
	return;
}

pcl::PointCloud<pcl::Normal>::Ptr ToolFunction::getNormals(PointCloudT::Ptr cloud, double radius)
{
	pcl::PointCloud<pcl::Normal>::Ptr normalsPtr(new pcl::PointCloud<pcl::Normal>);
	pcl::NormalEstimation<PointT, pcl::Normal> norm_est;
	norm_est.setInputCloud(cloud);
	norm_est.setRadiusSearch(radius);
	norm_est.compute(*normalsPtr);
	return normalsPtr;
}

FPFHCloud::Ptr ToolFunction::getFeatures(PointCloudT::Ptr cloud, pcl::PointCloud<pcl::Normal>::Ptr normals, double radius)
{
	FPFHCloud::Ptr features(new FPFHCloud);
	pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
	pcl::FPFHEstimation<PointT, pcl::Normal, FPFHT> fpfh_est;
	fpfh_est.setInputCloud(cloud);
	fpfh_est.setInputNormals(normals);
	fpfh_est.setSearchMethod(tree);
	fpfh_est.setRadiusSearch(radius);
	fpfh_est.compute(*features);
	return features;
}

Eigen::Matrix4f ToolFunction::sac_ia_align(PointCloudT::Ptr source, PointCloudT::Ptr target,
	FPFHCloud::Ptr source_feature, FPFHCloud::Ptr target_feature,
	int max_sacia_iterations, double min_correspondence_dist, double max_correspondence_dist)
{
	pcl::SampleConsensusInitialAlignment<PointT, PointT, FPFHT> sac_ia;
	Eigen::Matrix4f final_transformation;
	sac_ia.setInputSource(source);
	sac_ia.setSourceFeatures(source_feature);
	sac_ia.setInputTarget(target);
	sac_ia.setTargetFeatures(target_feature);
	sac_ia.setMaximumIterations(max_sacia_iterations);
	sac_ia.setMinSampleDistance(min_correspondence_dist);
	sac_ia.setMaxCorrespondenceDistance(max_correspondence_dist);
	PointCloudT::Ptr finalcloud(new PointCloudT);
	sac_ia.align(*finalcloud);
	final_transformation = sac_ia.getFinalTransformation();
	return final_transformation;
}

Eigen::Matrix4f ToolFunction::roughRegistration(PointCloudT::Ptr source, PointCloudT::Ptr target)
{
	PointCloudT::Ptr source_copy(new PointCloudT);
	PointCloudT::Ptr target_copy(new PointCloudT);

	//复制点云
	pcl::copyPointCloud(*source, *source_copy);
	pcl::copyPointCloud(*target, *target_copy);

	//去除无效点
	vector<int> indices1;
	vector<int> indices2;
	pcl::removeNaNFromPointCloud(*source_copy, *source_copy, indices1);
	pcl::removeNaNFromPointCloud(*target_copy, *target_copy, indices2);

	//降采样
	downSampling(source_copy, source_copy, 5, 5, 5);
	downSampling(target_copy, target_copy, 5, 5, 5);

	//计算法向量
	pcl::PointCloud<pcl::Normal>::Ptr source_normal(new pcl::PointCloud<pcl::Normal>);
	pcl::PointCloud<pcl::Normal>::Ptr target_normal(new pcl::PointCloud<pcl::Normal>);
	source_normal = getNormals(source_copy, 20.0);
	target_normal = getNormals(target_copy, 20.0);

	//计算FPFH特征    
	FPFHCloud::Ptr source_feature(new FPFHCloud);
	FPFHCloud::Ptr target_feature(new FPFHCloud);
	source_feature = getFeatures(source_copy, source_normal, 50.0);
	target_feature = getFeatures(target_copy, target_normal, 50.0);

	//SAC-IA配准
	Eigen::Matrix4f init_transform;
	init_transform = sac_ia_align(source_copy, target_copy, source_feature, target_feature, 1000, 0.01, 1000.0);
	return init_transform;
}

void ToolFunction::viewPair(PointCloudT::Ptr cloud1, PointCloudT::Ptr cloud2,
	PointCloudT::Ptr cloud1al, PointCloudT::Ptr cloud2al) {

	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
	viewer->initCameraParameters();
	int v1(0), v2(0);
	viewer->createViewPort(0.0, 0.0, 0.5, 1.0, v1);
	viewer->setBackgroundColor(0, 0, 0, v1);
	viewer->addText("Before Alignment", 10, 10, "v1 text", v1);
	PointCloudColorHandlerCustom<PointT> green(cloud1, 0, 255, 0);
	PointCloudColorHandlerCustom<PointT> red(cloud2, 255, 0, 0);
	viewer->addPointCloud(cloud1, green, "v1_target", v1);
	viewer->addPointCloud(cloud2, red, "v1_sourse", v1);

	viewer->createViewPort(0.5, 0.0, 1.0, 1.0, v2);
	viewer->setBackgroundColor(0, 0, 0, v2);
	viewer->addText("After Alignment", 10, 10, "v2 text", v2);
	PointCloudColorHandlerCustom<PointT> green2(cloud1al, 0, 255, 0);
	PointCloudColorHandlerCustom<PointT> blue(cloud2al, 0, 0, 255);
	viewer->addPointCloud(cloud1al, green2, "v2_target", v2);
	viewer->addPointCloud(cloud2al, blue, "v2_sourse", v2);
	viewer->spin();
}

void ToolFunction::convertFromTXTToMatrix(std::string file, Eigen::MatrixXd &M, const int Demension)
{
	std::ifstream infile;
	infile.open(file.data());   //将文件流对象与文件连接起来 
	assert(infile.is_open());   //若失败,则输出错误消息,并终止程序运行 

	std::vector<double> vec((std::istream_iterator<double>(infile)), (std::istream_iterator<double>()));

	M = Eigen::MatrixXd::Zero(vec.size() / Demension, Demension);
	for (int i = 0; i < vec.size(); i = i + Demension)
	{
		for (int k = 0; k < Demension; k++)
		{
			M(i / Demension, k) = vec[i + k];
		}
	}

	infile.close();             //关闭文件输入流 
}









