#pragma once
#include <cstring>
#include <string>
#include <iostream>
template<typename floating_point_type, int length, int dimensions>
/* Scales the data so that each dimension of each data point is between 0 and 1.
  param length is how many data points are in points, while dimesions is how many
  dimesions each point possesses. */
void feature_scale(floating_point_type points[length][dimensions])
{
	/* Make space for the mins and maxes of each dimension. To start off,  
	   initialize them to the first point fo the data set.*/
	floating_point_type min[dimensions];
	floating_point_type max[dimensions];
	memcpy(min, points[0], dimensions * sizeof(floating_point_type));
	memcpy(max, points[0], dimensions * sizeof(floating_point_type));

	/* Compare each point in the data set to the min and the max 
	  data point. Start at 1 as min and max at the start are already
	  the first data point. */
	for (int i = 1;i < length; i++)
	{
		/* Cache data point */
		floating_point_type* point = points[i];
		/* Compare (and possibly set) min and maxes of each
		   dimension */
		for (int j = 0;j < dimensions; j++)
		{
			/* Cache dimension */
			floating_point_type dim = point[j];
			if (dim < min[j])
				min[j] = dim;
		    if (dim > max[j])
				max[j] = dim;
		}
	}
	/* Compute the ranges of each dimension */
	floating_point_type ranges[dimensions];
	for (int i = 0; i < dimensions;i++)
	{
		ranges[i] = max[i] - min[i];
		if (ranges[i] == 0)
			ranges[i] = 1;
		else if (ranges[i] < 0)
			throw "range of a dimension is negative!";
	}

	for (int i = 0; i < length; i++)
	{
		/* Cache data point */
		floating_point_type* point = points[i];
		/* Perform scaling on each dimension of the data
		   point */
		for (int j = 0;j < dimensions; j++)
		{
			point[j] -= min[j];
			point[j] /= ranges[j];
		}
	}
}