// Kmeans.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "FeatureScale.h"
#include <stdio.h>
#include <iostream>
#include <ctime>
using namespace std;


void test_feature_scale()
{
	const int LENGTH = 5;
	const int DIMENSIONS = 10;
	double test[LENGTH][DIMENSIONS];
	srand(time(0));
	for (int i = 0;i < LENGTH; i++)
	{
		for (int j = 0;j < DIMENSIONS; j++)
		{
			test[i][j] = rand() % (10 * (j + 1));
			cout << test[i][j] << " ";
		}
		cout << endl;
	}
	feature_scale<double, LENGTH, DIMENSIONS>(test);	
	cout << "Feature scaled ... " << endl;
	for (int i = 0;i < LENGTH; i++)
	{
		for (int j = 0;j < DIMENSIONS; j++)
		{
			cout << test[i][j] << " ";
		}
		cout << endl;
	}
	cin.get();
}

void test_cluster()
{

}
int main()
{
	test_feature_scale();
}