#pragma once
#include "FeatureScale.h"
#include "Cluster.h"
#include <vector>

/* NOTE: This implementation assumes that K > 1 */
template<typename floating_point_type, int K, int length, int dimensions>
KCluster<floating_point_type, dimensions>* Kmeans(floating_point_type points[length][dimensions], int iterations = 1000)
{
	/* Normalize the data such the variable ranges of each dimensions won't matter */
	feature_scale<floating_point_type, length, dimensions>(points);
	KCluster<floating_point_type, dimensions>* clusters;
	/* Initialize K clusters to the first K points. */
	for (int i = 0;i < K; i++)
	{
		clusters = new KCluster<floating_point_type, dimensions>(points[i]);
		clusters++;
	}
	/* Reset cluster counter to the beginning of the array */
	clusters -= K;
	KCluster<floating_point_type, dimensions>* closest = cluster;
	int iter = 0;
	while (iter < iterations)
	{
		/* Clear the clusters at the start of each iteration so that
		   each cluster ins't affected by previous iteration results. */
		clusters->clear();
		for (int i = 0; i < length; i++)
		{
			/* Set the closest to the first cluster at the beginning.
			Cache data point. */
			floating_point_type* point = points[i];
			closest = clusters;
			for (int j = 1; j < K; j++)
			{
				/* If the distance of the closest  cluster is greater than
				   that of the current one, set new closest cluster. */
				if (closest->compare(point) > clusters->compare(point))
				{
					closest = clusters;
				}
				/* Advance current cluster */
				clusters++;
			}
			/* Add the point to the closest cluster */
			closest->add(point);
			/* Reset current cluster position */
			clusters -= K;
		}
		/* Have each cluster compute their new centroids */
		for (int i = 0; i < K; i++)
		{
			clusters->comp_centroid();
			clusters++;
		}
		iter++;
		clusters -= K;
	}
	return clusters;
}
