#pragma once
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>

#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/keypoints/iss_3d.h>

#include <pcl/features/normal_3d.h>
#include <pcl/features/fpfh.h>

#include <pcl/registration/ia_ransac.h>
#include <pcl/visualization/pcl_visualizer.h>

using pcl::visualization::PointCloudColorHandlerGenericField;
using pcl::visualization::PointCloudColorHandlerCustom;

typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloudT;
typedef pcl::PointNormal PointNormalT;
typedef pcl::PointCloud<PointNormalT> PointCloudWithNormals;
typedef pcl::FPFHSignature33 FPFHT;
typedef pcl::PointCloud<FPFHT> FPFHCloud;

/** \brief class ToolFunction provides some basic operations used in class Registration
  * class ToolFunction is a function class£¬which means it has no need to instantiate, using a ptr is enough
  */
class ToolFunction
{
public:
	ToolFunction();
	~ToolFunction();

	/** \brief Provide the filepath of the input file
	  * \Type of the input file should be stl
	  */
	bool convertFromSTLToPLY(std::string);

	/** \brief Provide the filepath of the input file
	  * \Type of the input file should be txt
	  */
	bool convertFromTXTToPCD(std::string);

	/** \brief Provide a pointer to the input dataset and a pointer to the output dataset
	  * \Resize the leafSize by providing three float numbers which have the same default value:10.0
	  */
	void downSampling(PointCloudT::Ptr cloud_input, PointCloudT::Ptr cloud_output, float sizeX = 10.0, float sizeY = 10.0, float sizeZ = 10.0);

	/** \brief Provide a pointer to the input dataset
	  * \Resize the RadiusSearch by providing a double number which has a default value:20.0
	  */
	pcl::PointCloud<pcl::Normal>::Ptr getNormals(PointCloudT::Ptr cloud, double radius = 20.0);

	/** \brief Provide a pointer to the input dataset and a pointer to the normals of the dataset
	  * \Resize the RadiusSearch by providing a double number which has a default value:50.0
	  */
	FPFHCloud::Ptr getFeatures(PointCloudT::Ptr cloud, pcl::PointCloud<pcl::Normal>::Ptr normals, double radius = 50.0);

	/** \brief SAC-IA algorithm, which provides a rough estimation of the registration between input point clouds
	  * \Resize the MaximumIterations by providing a int number which has a default value:1000
	  * \Resize the MinSampleDistance by providing a double number which has a default value:0.01
	  * \Resize the MaxCorrespondenceDistance by providing a double number which has a default value:1000
	  */
	Eigen::Matrix4f sac_ia_align(PointCloudT::Ptr source, PointCloudT::Ptr target, FPFHCloud::Ptr source_feature, FPFHCloud::Ptr target_feature,
		int max_sacia_iterations = 1000, 
		double min_correspondence_dist = 0.01, 
		double max_correspondence_dist = 1000);

	/** \brief Provide two pointers that point to two input datasets respectively
	  * \This function provides a rough estimation of the registration between two input point clouds
	  */
	Eigen::Matrix4f roughRegistration(PointCloudT::Ptr source, PointCloudT::Ptr target);

	/** \brief This function displays cloud1&cloud2 and cloud1al&cloud2al in two separate viewers
	  */
	void viewPair(PointCloudT::Ptr cloud1, PointCloudT::Ptr cloud2,
		PointCloudT::Ptr cloud1al, PointCloudT::Ptr cloud2al);

	void convertFromTXTToMatrix(std::string file, Eigen::MatrixXd &M, const int Demension);
};