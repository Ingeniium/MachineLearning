#pragma once
#include <cmath>
#include <vector>
#include <algorithm>
using namespace std;
template<typename floating_point_type, int dimensions>
class KCluster
{
	/* NOTE: since the centroid will soon represent a mean of some points 
	   rather than an actual points, we shall keep an actual copy rather
	   than a pointer to a point */
	floating_point_type centroid[dimensions];
	/* Vector use because the number of points a cluster might have
	   is variable. */
	vector<floating_point_type*> points;

	public void add(floating_point_type* point)
	{
		points.push_back(point);
	}

	public void remove(floating_point_type* point)
	{
		points.erase(remove(points.begin(), points.end(), point), points.end());
	}

	/* This prevents any points from lingering in another cluster due to an
	   assignment from a previous iteration */
	public void clear()
	{
		points.clear();
	}

	/* To initialize the cluster, copy input point data into the centroid.*/
	public KCluster(floating_point_type* point)
	{
		memcpy(centroid, point, dimensions * sizeof(floating_point_type));
	}

	/* Using the newly assinged points, compute a new centroid by 
	   making it the means of the cluster's poitns. */
	public void comp_centroid()
	{
		/* Reset centroid data to 0 */
		memset(centroid, 0, dimensions * sizeof(floating_point_type));
		int size = points.size();
		/* Compute the means of each dimension by simply adding the values
		   of each point's dimension to the centoid's. */
		for (int i = 0; i < size; i++)
		{
			/* Cache current data point */
			floating_point_type* point = points[i];
			for (int j = 0; j < dimensions; j++)
				centroid[j] += point[j];
		}
		/* After getting the sums, simply divide by size to get means */
		for (int j = 0; j < dimensions; j++)
			centroid[j] /= size;
	}

	/* Gets the "distance" between the point and this cluster's centroid. */
	floating_point_type compare(floating_point_type* input)
	{
		floating_point_type result = 0;
		for (int i = 0;i < dimensions; i++)
			result += pow(centroid[i] - input[i], 2);
		return result;
	}

	
};