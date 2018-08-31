using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

class UnsupervisedLearning
{
    private static double[,] GetClusterAverages(List<List<int>> assignments,double[,] examples,int num_clusters)
    {
        double[,] clusters = new double[num_clusters, examples.GetLength(1)];
        for (int centroid = 0; centroid < assignments.Count; centroid++)
        {
            for (int point = 0; point < assignments[centroid].Count; point++)
            {
                int example_num = assignments[centroid][point];
                for(int variable = 0;variable < examples.GetLength(1);variable++)
                {
                    clusters[centroid, variable] += examples[example_num, variable];
                }
            }
            for (int variable = 0; variable < examples.GetLength(1); variable++)
            {
                clusters[centroid, variable] /= assignments[centroid].Count;
            }
        }
        
        return clusters;
    }

    private static double KClusterCost(double[,] examples,double[,] clusters,int example,int cluster)
    {
        double sum = 0;
        for (int i = 0; i < examples.GetLength(1); i++)
        {
            sum += Math.Pow(clusters[cluster,i] - examples[example,i], 2);
        }
        return sum;
    }

    private static double[,] RandomClusterInitilization(double[,] examples,int num_clusters)
    {
        Random rand = new Random();
        double[,] clusters = new double[num_clusters, examples.GetLength(1)];
        int clust = 0;
        HashSet<double> set = new HashSet<double>();
        while(clust != num_clusters)
        {
            int num = rand.Next(examples.GetLength(0));
            if(set.Add(num))
            {
                for (int i = 0; i < examples.GetLength(1); i++)
                    clusters[clust, i] = examples[num, i];
                clust++;
            }
        }
        return clusters;
    }

    public static int[] KMeans(double[,] examples,int num_clusters,int max_iterations)
    {
        if(num_clusters < 2)
        {
            throw new ArgumentException("There must be at least two clusers/categories!");
        }
        if(examples == null)
        {
            throw new ArgumentNullException("Examples is null!");
        }
        Random rand = new Random();
        double[,] clusters = RandomClusterInitilization(examples, num_clusters);
        int iter = 0;
        while(iter < max_iterations)
        {
            iter++;
            List<List<int>> cluster_assignments = new List<List<int>>();
            double[] variable_totals = new double[examples.GetLength(1)];
            for(int i = 0;i < num_clusters;i++)
            {
                cluster_assignments.Add(new List<int>());
            }
            for (int example = 0; example < examples.GetLength(0); example++)
            {
                double min_cluster_cost = Double.MaxValue;
                int last_cluster = 0;
                for (int centroid = 0; centroid < clusters.GetLength(0); centroid++)
                {
                    double cost = KClusterCost(examples, clusters, example, centroid);

                    if (min_cluster_cost > cost)
                    {
                        min_cluster_cost = cost;
                        cluster_assignments[centroid].Add(example);
                        cluster_assignments[last_cluster].Remove(example);
                        last_cluster = centroid;
                    }
                }
            }
        }
        return null;
    }
}