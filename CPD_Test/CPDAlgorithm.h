#pragma once
#include <Eigen/Dense>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloudT;

using namespace Eigen;

namespace cpd
{
	struct Cloud_property
	{
		pcl::PCLHeader header;
		uint32_t width;
		uint32_t height;
		bool is_dense;
		Eigen::Quaternionf sensor_orientation_;
		Eigen::Vector4f    sensor_origin_;
		long size;
	};

	struct Normal
	{
		double xd[3] = { 0 };
		double yd[3] = { 0 };
		double xscale = 0;
		double yscale = 0;
	};

	class CPDAlgorithm
	{
	public:
		CPDAlgorithm();
		~CPDAlgorithm();

		void setInputSource(const PointCloudT::Ptr source_cloud);
		void setInputTarget(const PointCloudT::Ptr target_cloud);
		void setInputSource(const MatrixXd &source_cloud);
		void setInputTarget(const MatrixXd &target_cloud);
		void setMaximumIterations(int max_it_input);
		void setBeta(double beta_input);
		void setLambda(double lambda_input);
		void setTolerance(double tol_input);
		void setOutlier(double outlier_input);
		void setSigma2(double sigma2_input);
		void align(PointCloudT::Ptr align_cloud);
		void align(MatrixXd &align_cloud);

	private:
		static const int D = 3;//Demension
		MatrixXd target;
		MatrixXd source;
		MatrixXd W;
		MatrixXd T;
		double beta = 2;
		double lambda = 3;
		int max_it = 150;
		double tol = 1e-05;
		double outlier = 0.1;
		double sigma2 = 0;
		int iter = 0;
		Cloud_property cloud_out_property;

		void cpd_GRBF(MatrixXd &x, MatrixXd &y,
			double beta, double lambda, int max_it, double tol, double outliers, double &sigma2, int &iter,
			MatrixXd &W, MatrixXd &T);
		void cpd_G(MatrixXd &x, MatrixXd &y, double beta, MatrixXd &G);
		void cpd_P(MatrixXd &x, MatrixXd &y, double &sigma2, double outliers,
			MatrixXd &P1, MatrixXd &Pt1, MatrixXd &PX, double &L);
		void cpd_Pcorrespondence(MatrixXd &x, MatrixXd &y, double sigma2, double outlier, MatrixXd &Pc);
		void cpd_normalize(MatrixXd &x, MatrixXd &y, Normal& normal);
		void cpd_denormalize(MatrixXd &T, Normal& normal);
		void convertFromPointCloudToMatrixXd(const PointCloudT::Ptr cloud_in, MatrixXd& matrix_out);
		void convertFromMatrixXdToPointCloud(const MatrixXd& matrix_in, PointCloudT::Ptr cloud_out);
		void initializeCloudProperty(const PointCloudT::Ptr cloud_in);
	};
}