#include <iostream>
#include <vector>
#include <ctime>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

//计算聚类中心，cNum为聚类中心的数目，m为控制收敛速度的参数，通常为2
void calcCenters(Mat& image, vector<Mat>& U, int cNum, double m, double* center)
{
	double sSum;
	double sum;
	for (int k = 0; k < cNum; k++)
	{
		sSum = 0;
		sum = 0;
		for (int i = 0; i < image.rows; i++)
			for (int j = 0; j < image.cols; j++)
			{
				sSum += pow(U[k].at<double>(i,j), m);
				sum += pow(U[k].at<double>(i,j), m) * (double)(image.at<uchar>(i,j));
			}
		center[k] = sum / sSum;	//得到聚类中心
	}
}

void FLICM(Mat& image, vector<Mat>& U, double m, int cNum, int winSize, int maxIter, double thrE, int& iter)
{
	int sStep = (winSize - 1) / 2;
	double* center = new double[cNum];
	calcCenters(image, U, cNum, m, center);
	double dMax = 10.0;	// 先初始化一个较大的值，|U_new-U_old|
	// vector默认复制构造函数为浅拷贝，需自己复制
	Mat tmp(image.size(), CV_64FC1);	// 复制时，使用的临时变量
	vector<Mat> Uold;
	for (int k = 0; k < cNum; k++)
	{
		for (int i = 0; i < image.rows; i++)
			for (int j = 0; j < image.cols; j++)
				tmp.at<double>(i,j) = U[k].at<double>(i,j);
		Uold.push_back(tmp);
	}

	vector<double> d1(cNum, 0);	// d1为公式（17）中的G_ki
	vector<double> d2(cNum, 0);	// d2为公式（19）中的分母的前半部分
	double sSum = 0;
	double dd;
	int x, y;	// 坐标点x, y
	double dist = 0.0;	// 距离
	double val = 0;	// 像素的灰度值
	double* cenOld = new double[cNum];
	while (dMax > thrE && iter < maxIter)	// 步骤6
	{
		for (int i = 0; i < image.rows; i++)
		{
			for (int j = 0; j < image.cols; j++)
			{
				for (int k = 0; k < cNum; k++)
				{
					sSum =  4.9407e-324;	//重置sSum
					for (int ii = -sStep; ii <= sStep; ii++)	// sStep是窗口半径，ii和jj是对局部区域计算
						for (int jj = -sStep; jj <= sStep; jj++)
						{
							x = i + ii;		// 有待思考
							y = j + jj;
							dist = sqrt(pow((x - i), 2)+ pow((y - j), 2));	// 局部区域中， 点(x, y) 和 (i, j)的距离
							// （x,y）不能超过边界，也不能与（i,j）重合
							if ( x >= 0 && x < image.rows && y >= 0 && y < image.cols && (ii != 0 || jj != 0))
								{
									val = (double)image.at<uchar>(x, y);	 // i的邻域点j点的灰度值，公式（17）
									sSum = sSum + 1.0 / (1.0 + dist) * (1 - pow(Uold[k].at<double>(i, j), m)) * pow(abs(val - center[k]), 2);
								}	
						}
					d1[k] = sSum;
					d2[k] = pow(abs((double)image.at<uchar>(i,j) - center[k]), 2);
				}
				for (int k = 0; k < cNum; k++)
				{
					if (d1[k] == 0)
						d1[k] =  4.9407e-324;	 // 接近于0的极小值
					if (d2[k] == 0)
						d2[k] =  4.9407e-324;	 // 接近于0的极小值
				}
				for (int k = 0; k < cNum; k++)
				{
					dd = d1[k] + d2[k];	// 每个k都用到啦
					sSum =  4.9407e-324;
					for (int ii = 0; ii < cNum; ii++)
						sSum = sSum + pow((dd / (d1[ii] + d2[ii])), (1.0 / (m - 1.0)));
					U[k].at<double>(i,j) = 1.0 / sSum;
				}
			}	// end for j
		}	// end for i
		for (int k = 0; k < cNum; k++)
			cenOld[k] = center[k];
		calcCenters(image, U, cNum, m, center);	 // 利用旧隶属度矩阵，计算初始聚类中心
		for (int k = 0; k < cNum; k++)
		{
			if (dMax < abs(cenOld[k] - center[k]))
				dMax = abs(cenOld[k] - center[k]);
			Uold[k] = U[k];
		}
		cout << "第" << iter << "次迭代" << endl;
		iter++; // 记录迭代次数
	}	// end for while

	delete [] center;
	delete [] cenOld;
}

void Flicm_Cluster(Mat& img, Mat& out_img, int cNum, double m, int winSize, int maxIter, double thrE, int& iter)
{
	Mat gray;
	img.copyTo(gray);
	Mat image(img.size(), CV_64FC1);
	if (gray.channels() > 1)	// 转化为灰度图
	{
		cvtColor(gray, gray, CV_RGB2GRAY);
		image = gray;	// 转换为0-255的double型
	}	
	else
		image = img;
	// 随机初始化隶属度矩阵U,大小为(H,W,cNum)
	vector<Mat> U;	// 隶属度矩阵U
	// 一定要提前指定U中元素的类型和大小！
	for (int k = 0; k < cNum; k++)
	{
		Mat u(img.size(), CV_64FC1);
		U.push_back(u);
	}

	Mat col_sum(image.rows, image.cols, CV_64FC1);
	// 一定要初始化！
	for (int i = 0; i < image.rows; i++)
		for (int j = 0; j < image.cols; j++)
			col_sum.at<double>(i, j) = 0.0;

	for (int i = 0; i < image.rows; i++)
	{
		for (int j = 0; j < image.cols; j++)
			for (int k = 0; k < cNum; k++)
			{
				U[k].at<double>(i,j) = rand() / (double)(RAND_MAX+1.0);	// 产生一个0~1之间的小数
				col_sum.at<double>(i,j) += U[k].at<double>(i,j);
			}
	}
	for (int k = 0; k < cNum; k++)
		divide(U[k], col_sum, U[k]);

	FLICM(image, U, m, cNum, winSize, maxIter, thrE, iter);
	
	// 根据最大隶属度，判别每个像素点所属类别
	Mat clus(image.size(), CV_8UC1);
	int max_clus;
	for (int i = 0; i < clus.rows; i++)
		for (int j = 0; j < clus.cols; j++)
		{
			max_clus = 0;
			for (int k = 1; k < cNum; k++)
			{
				if (U[k].at<double>(i,j) > U[k-1].at<double>(i,j))
					max_clus = k;
			}
			clus.at<uchar>(i,j) = max_clus;
		}
		// 显示聚类的结果图片
		for (int i = 0; i < out_img.rows; i++)
			for (int j = 0; j < out_img.cols; j++)
			{
				/*for (int k = 0; k < cNum; k++)
					if (clus.at<uchar>(i,j) == k)
					{
						out_img.at<cv::Vec3b>(i,j)[0] = 255*k;
						out_img.at<cv::Vec3b>(i,j)[1] = 255*k;
						out_img.at<cv::Vec3b>(i,j)[2] = 255*k;
					}*/
				if (clus.at<uchar>(i,j) == 0)
				{
					out_img.at<cv::Vec3b>(i,j)[0] = 0;
					out_img.at<cv::Vec3b>(i,j)[1] = 255;
					out_img.at<cv::Vec3b>(i,j)[2] = 0;
				}

				else if (clus.at<uchar>(i,j) == 1)
				{
					out_img.at<cv::Vec3b>(i,j)[0] = 0;
					out_img.at<cv::Vec3b>(i,j)[1] = 0;
					out_img.at<cv::Vec3b>(i,j)[2] = 255;
				}
				else
				{
					out_img.at<cv::Vec3b>(i,j)[0] = 255;
					out_img.at<cv::Vec3b>(i,j)[1] = 255;
					out_img.at<cv::Vec3b>(i,j)[2] = 255;
				}
			}
}

int main()
{
	// 计算运行时间
	clock_t start, end;
	start = clock();

	// 设置聚类算法的一些初始参数
	int cNum = 3;	// 聚类中心数量
	double m = 2;	// 模糊指数m
	int winSize = 3;	// 局部窗口直径
	int maxIter = 100;	// 最大迭代次数
	double thrE =  0.00001;	  // 收敛阈值
	Mat img = imread("brain.tif");
	Mat out_img(img.size(), CV_8UC3);
	int iter = 0;
	Flicm_Cluster(img, out_img, cNum, m, winSize, maxIter, thrE, iter);
	cout << "总共迭代了" << iter << "次" << endl;
	imshow("dst", out_img);
	//imwrite("1_8_1.jpg", out_img);

	// 显示运行多少时间
	end = clock();
	double total_time = 0;
	total_time = (end - start) / CLOCKS_PER_SEC;
	cout << "迭代" << iter << "次需要花费" << total_time << "秒" << endl; 

	waitKey();

	return 0;
}