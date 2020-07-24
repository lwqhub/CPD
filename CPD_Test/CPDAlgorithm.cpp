#include "pch.h"
#include "CPDAlgorithm.h"
#include <Eigen/Sparse>
#include <pcl/common/copy_point.h>

using namespace cpd;

CPDAlgorithm::CPDAlgorithm()
{
}


CPDAlgorithm::~CPDAlgorithm()
{
}

void CPDAlgorithm::setMaximumIterations(int max_it_input)
{
	max_it = max_it_input;
}

void CPDAlgorithm::setInputSource(const PointCloudT::Ptr source_cloud)
{
	convertFromPointCloudToMatrixXd(source_cloud, source);
	initializeCloudProperty(source_cloud);
}

void CPDAlgorithm::setInputTarget(const PointCloudT::Ptr target_cloud)
{
	convertFromPointCloudToMatrixXd(target_cloud, target);
}

void CPDAlgorithm::setInputSource(const MatrixXd &source_cloud)
{
	source = source_cloud;
}

void CPDAlgorithm::setInputTarget(const MatrixXd &target_cloud)
{
	target = target_cloud;
}

void CPDAlgorithm::setBeta(double beta_input)
{
	beta = beta_input;
}

void CPDAlgorithm::setLambda(double lambda_input)
{
	lambda = lambda_input;
}

void CPDAlgorithm::setTolerance(double tol_input)
{
	tol = tol_input;
}

void CPDAlgorithm::setOutlier(double outlier_input)
{
	outlier = outlier_input;
}

void CPDAlgorithm::setSigma2(double sigma2_input)
{
	sigma2 = sigma2_input;
}

void CPDAlgorithm::align(PointCloudT::Ptr align_cloud)
{
	Normal normal;
	cpd_normalize(target, source, normal);
	cpd_GRBF(target, source, 
		beta, lambda, max_it, tol, outlier, sigma2, iter, 
		W, T);
	for (long i = 0; i < T.rows(); i++)
	{
		for (int k = 0; k < D; k++)
			T(i, k) = T(i, k)*normal.xscale + normal.xd[k];
	}
	convertFromMatrixXdToPointCloud(T, align_cloud);
}

void CPDAlgorithm::align(MatrixXd &align_cloud)
{
	Normal normal;
	cpd_normalize(target, source, normal);
	cpd_GRBF(target, source,
		beta, lambda, max_it, tol, outlier, sigma2, iter,
		W, align_cloud);
	cpd_denormalize(align_cloud, normal);
}

void CPDAlgorithm::convertFromPointCloudToMatrixXd(const PointCloudT::Ptr cloud_in, MatrixXd& matrix_out)
{
	long size = cloud_in->points.size();
	matrix_out = MatrixXd::Zero(size, D);
	for (long i = 0; i < size; ++i)
	{
		matrix_out(i, 0) = cloud_in->points[i].x;
		matrix_out(i, 1) = cloud_in->points[i].y;
		matrix_out(i, 2) = cloud_in->points[i].z;
	}
}

void CPDAlgorithm::convertFromMatrixXdToPointCloud(const MatrixXd& matrix_in, PointCloudT::Ptr cloud_out)
{
	cloud_out->header   = cloud_out_property.header;
	cloud_out->width    = cloud_out_property.width;
	cloud_out->height   = cloud_out_property.height;
	cloud_out->is_dense = cloud_out_property.is_dense;
	cloud_out->sensor_orientation_ = cloud_out_property.sensor_orientation_;
	cloud_out->sensor_origin_ = cloud_out_property.sensor_origin_;
	cloud_out->points.resize(cloud_out_property.size);

	for (long i = 0; i < cloud_out_property.size; ++i)
	{
		cloud_out->points[i].x = matrix_in(i, 0);
		cloud_out->points[i].y = matrix_in(i, 1);
		cloud_out->points[i].z = matrix_in(i, 2);
	}
}

void CPDAlgorithm::cpd_GRBF(MatrixXd &x, MatrixXd &y,
	double beta, double lambda, int max_it, double tol, double outliers, double &sigma2, int &iter,
	MatrixXd &W, MatrixXd &T)
{
	int n = x.rows();
	int m = y.rows();

	sigma2 = 0;
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < m; j++)
		{
			for (int k = 0; k < D; k++)
				sigma2 += pow(x(i, k) - y(j, k), 2);
		}
	}
	sigma2 = sigma2 / (m*n*D);
	double sigma2_init = sigma2;

	//Initialization
	T = y;
	W = MatrixXd::Zero(m, D);
	iter = 0;
	double ntol = tol + 10;
	double L = 1;

	//Construct affinity matrix G
	MatrixXd G;
	cpd_G(y, y, beta, G);

	while ((iter < max_it) && (ntol > tol) && (sigma2 > 1e-8))
	{
		double L_old = L;

		MatrixXd P1, Pt1, PX;
		cpd_P(x, T, sigma2, outliers, P1, Pt1, PX, L);

		L = L + lambda / 2 * (W.transpose()*G*W).trace();
		ntol = abs((L - L_old) / L);

		//M-step. Solve linear system for W.
		SparseMatrix<double> dP(m, m);
		for (int i = 0; i < m; i++)
		{
			dP.coeffRef(i, i) = P1(i, 0);
		}
		W = (dP*G + lambda * sigma2*MatrixXd::Identity(m, m)).inverse()*(PX - dP * y);

		/*
		//another method to caculate W
		MatrixXd dPy = MatrixXd::Zero(PX.rows(), PX.cols());
		MatrixXd dPG = MatrixXd::Zero(G.rows(), G.cols());

		for (int i = 0; i < PX.rows(); i++)
		{
			dPy.row(i) = y.row(i)*P1(i, 0);
			dPG.row(i) = G.row(i)*P1(i, 0);
		}
		W = (dPG + lambda*sigma2*MatrixXd::Identity(m, m)).inverse()*(PX - dPy);
		*/

		//update Y postions
		T = y + G * W;

		double Np = 0;
		for (int i = 0; i < P1.rows(); i++)
		{
			Np += P1(i, 0);
		}

		double sigma2save = sigma2;

		double sum_first = 0, sum_second = 0;

		for (int i = 0; i < n; i++)
			for (int k = 0; k < D; k++)
				sum_first += x(i, k)*x(i, k)*Pt1(i, 0);

		for (int j = 0; j < m; j++)
			for (int k = 0; k < D; k++)
				sum_second += T(j, k)*T(j, k)*P1(j, 0);

		sigma2 = abs((sum_first + sum_second - 2 * (PX.transpose()*T).trace()) / (Np*D));

		iter++;
	}
	return;
}

void CPDAlgorithm::cpd_G(MatrixXd &x, MatrixXd &y, double beta, MatrixXd &G)
{
	double k = -2 * pow(beta, 2);

	int n = x.rows();
	int m = y.rows();

	G = MatrixXd::Zero(n, m);
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < m; j++)
		{
			double vector_norm = 0;
			for (int k = 0; k < D; k++)
				vector_norm += pow(x(i, k) - y(j, k), 2);
			vector_norm = vector_norm / k;
			G(i, j) = exp(vector_norm);
		}
	}
}

void CPDAlgorithm::cpd_P(MatrixXd &x, MatrixXd &y, double &sigma2, double outliers,
	MatrixXd &P1, MatrixXd &Pt1, MatrixXd &PX, double &L)
{
	struct cpd_P_comp
	{
		void operator()(MatrixXd &x, MatrixXd &y, double* sigma2, double* outlier,
			MatrixXd &P1, MatrixXd &Pt1, MatrixXd &Px, double* E, int N, int M, int D)
		{
			int		n, m, d;
			double	ksig, diff, razn, outlier_tmp, sp;
			double	*P, *temp_x;

			P1 = MatrixXd::Zero(M, 1);
			Pt1 = MatrixXd::Zero(N, 1);
			Px = MatrixXd::Zero(M, D);

			P = (double*)calloc(M, sizeof(double));
			temp_x = (double*)calloc(D, sizeof(double));

			*E = 0;
			ksig = -2.0 * *sigma2;
			outlier_tmp = (*outlier*M*pow(-ksig * 3.14159265358979, 0.5*D)) / ((1 - *outlier)*N);

			for (n = 0; n < N; n++) {
				sp = 0;
				for (m = 0; m < M; m++) {
					razn = 0;
					for (d = 0; d < D; d++) {
						diff = x(n, d) - y(m, d);  diff = diff * diff;
						razn += diff;
					}
					*(P + m) = exp(razn / ksig);
					sp += *(P + m);
				}
				sp += outlier_tmp;
				Pt1(n, 0) = 1 - outlier_tmp / sp;
				for (d = 0; d < D; d++) {
					*(temp_x + d) = x(n, d) / sp;
				}
				for (m = 0; m < M; m++) {
					P1(m, 0) += *(P + m) / sp;
					for (d = 0; d < D; d++) {
						Px(m, d) += *(temp_x + d)**(P + m);
					}
				}
				*E += -log(sp);
			}
			*E += D * N*log(*sigma2) / 2;
			free((void*)P);
			free((void*)temp_x);
			return;
		}
	};

	int N = x.rows();
	int M = y.rows();

	cpd_P_comp cpd_comp;
	cpd_comp(x, y, &sigma2, &outliers, P1, Pt1, PX, &L, N, M, D);
}

void CPDAlgorithm::cpd_Pcorrespondence(MatrixXd &x, MatrixXd &y, double sigma2, double outlier, MatrixXd &Pc)
{
	struct cpd_Pcorrespondence_comp
	{
		void operator()(MatrixXd &x, MatrixXd &y, double* sigma2, double* outlier,
			MatrixXd &Pc, int N, int M, int D)
		{
			int		n, m, d;
			double	ksig, diff, razn, outlier_tmp, temp_x, sp;
			double	*P, *P1;

			Pc = MatrixXd::Zero(M, 1);

			P = (double*)calloc(M, sizeof(double));
			P1 = (double*)calloc(M, sizeof(double));

			ksig = -2.0 * (*sigma2 + 1e-3);
			outlier_tmp = (*outlier*M*pow(-ksig * 3.14159265358979, 0.5*D)) / ((1 - *outlier)*N);
			if (outlier_tmp == 0) outlier_tmp = 1e-10;

			for (n = 0; n < N; n++) {
				sp = 0;
				for (m = 0; m < M; m++) {
					razn = 0;
					for (d = 0; d < D; d++) {
						diff = x(n, d) - y(m, d);  diff = diff * diff;
						razn += diff;
					}
					*(P + m) = exp(razn / ksig);
					sp += *(P + m);
				}
				sp += outlier_tmp;
				for (m = 0; m < M; m++) {
					temp_x = *(P + m) / sp;
					if (n == 0)
					{
						*(P1 + m) = *(P + m) / sp;
						Pc(m) = n + 1;
					};
					if (temp_x > *(P1 + m))
					{
						*(P1 + m) = *(P + m) / sp;
						Pc(m) = n + 1;
					}
				}
			}
			free((void*)P);
			free((void*)P1);
			return;
		}
	};

	int N = x.rows();
	int M = y.rows();

	cpd_Pcorrespondence_comp cpd_comp;
	cpd_comp(x, y, &sigma2, &outlier, Pc, N, M, D);
}

void CPDAlgorithm::cpd_normalize(MatrixXd &x, MatrixXd &y, Normal& normal)
{
	int n = x.rows();
	int m = y.rows();

	for (int i = 0; i < D; i++)
	{
		normal.xd[i] = x.col(i).mean();
		normal.yd[i] = y.col(i).mean();
	}

	double x2_sum = 0, y2_sum = 0;
	for (int i = 0; i < D; i++)
	{
		MatrixXd x_means = MatrixXd::Ones(n, 1) * normal.xd[i];
		x.col(i) = x.col(i) - x_means;
		x2_sum += x.col(i).transpose()*x.col(i);

		MatrixXd y_means = MatrixXd::Ones(m, 1) * normal.yd[i];
		y.col(i) = y.col(i) - y_means;
		y2_sum += y.col(i).transpose()*y.col(i);
	}

	normal.xscale = sqrt(x2_sum / n);
	normal.yscale = sqrt(y2_sum / m);

	for (int i = 0; i < D; i++)
	{
		x.col(i) = x.col(i) / normal.xscale;
		y.col(i) = y.col(i) / normal.yscale;
	}
	return;
}

void CPDAlgorithm::cpd_denormalize(MatrixXd &T, Normal &normal)
{
	for (long i = 0; i < T.rows(); i++)
	{
		for (int k = 0; k < D; k++)
			T(i, k) = T(i, k)*normal.xscale + normal.xd[k];
	}
}

void CPDAlgorithm::initializeCloudProperty(const PointCloudT::Ptr cloud_in)
{
	cloud_out_property.header   = cloud_in->header;
	cloud_out_property.width    = cloud_in->width;
	cloud_out_property.height   = cloud_in->height;
	cloud_out_property.is_dense = cloud_in->is_dense;
	cloud_out_property.sensor_orientation_ = cloud_in->sensor_orientation_;
	cloud_out_property.sensor_origin_ = cloud_in->sensor_origin_;
	cloud_out_property.size = cloud_in->points.size();
}
