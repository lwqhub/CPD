// CPD_Test.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include "pch.h"
#include <iostream>
#include "CPDAlgorithm.h"
#include "Toolfunction.h"
#include <pcl/io/pcd_io.h>
#include <pcl/console/time.h>   

using namespace std;
using namespace cpd;

int main()
{
	MatrixXd X, Y, T;
	ToolFunction *toolfunction;
	toolfunction->convertFromTXTToMatrix("X.txt", X, 3);
	toolfunction->convertFromTXTToMatrix("Y.txt", Y, 3);
	
	pcl::console::TicToc time;
	time.tic();
	CPDAlgorithm cpd;
	cpd.setInputSource(Y);
	cpd.setInputTarget(X);
	//cpd.setBeta(2);
	//cpd.setLambda(3);
	//cpd.setMaximumIterations(150);
	//cpd.setOutlier(0.1);
	//cpd.setSigma2(0);
	//cpd.setTolerance(1e-05);
	cpd.align(T);
	cout << "Total Time:" << time.toc() << "ms" << endl;

	ofstream out("Y'.txt");
	for (int i = 0; i < T.rows(); i++)
		out << T(i, 0) << "  " << T(i, 1) << "  " << T(i, 2) << endl;
	out.close();

	toolfunction->convertFromTXTToPCD("X.txt");
	toolfunction->convertFromTXTToPCD("Y.txt");
	toolfunction->convertFromTXTToPCD("Y'.txt");

	PointCloudT::Ptr cloud_in(new PointCloudT);  // Original point cloud
	PointCloudT::Ptr cloud_tr(new PointCloudT);  // Transformed point cloud
	PointCloudT::Ptr cloud_cpd(new PointCloudT);  // CPD output point cloud

	pcl::io::loadPCDFile("X.pcd", *cloud_in);
	pcl::io::loadPCDFile("Y.pcd", *cloud_tr);
	pcl::io::loadPCDFile("Y'.pcd", *cloud_cpd);
	
	toolfunction->viewPair(cloud_in, cloud_tr, cloud_in, cloud_cpd);
}

