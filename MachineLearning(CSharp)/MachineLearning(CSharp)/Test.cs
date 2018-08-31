using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

class Test
{
    public static void Main(String[] Args)
    {
        ForwardPropogateTest();
        //double[][] examples = {new double[]{1,4,5,9 },
        //    new double[]{1,2,8,7 },
        //    new double[]{1,3,9,5 }
        //};
        //double[] linear_results = { 36, 70, 52 };
        //double[] logistic_results = { 1, 0, 1 };
        //double[] linear_theta = RegressionImpl.LinearRegression(examples, linear_results);
        //Console.WriteLine("LINEAR REGRESSION");
        //for(int example = 0;example < examples.Length;example++)
        //{
        //    double approx_total = 0;
        //    for(int theta = 0;theta < linear_theta.Length;theta++)
        //    {
        //        double result = linear_theta[theta] * examples[example][theta];
        //        Console.WriteLine(linear_theta[theta] + " * " + examples[example][theta] + " = " + result);
        //        approx_total += result;
        //    }
        //    Console.WriteLine("Example " + (example + 1) + " approximated to be " + approx_total);
        //    Console.WriteLine();
        //}
        //Console.WriteLine();
        //Console.WriteLine("LOGISTIC REGRESSION");
        //double[] logistic_theta = RegressionImpl.LogisticRegression(examples, logistic_results);
        //foreach (double theta in logistic_theta)
        //{
        //    Console.Write(theta + " ");
        //}
        //Console.WriteLine();
        //Console.WriteLine("Total logistic cost to be approximated as " + RegressionImpl.LogisticCost(logistic_theta, examples, logistic_results));
        //Console.ReadKey();
    }
    public static void BackPropogateTest()
    {
        double[][] examples = {new double[]{1,4,5},
            new double[]{1,2,8 },
            new double[]{1,3,9 }
        };
        double[][][] thetas = new double[2][][];
        thetas[0] = new double[3][];
        thetas[0][0] = new double[] { 2, 5, 4 };
        thetas[0][1] = new double[] { 9, 1, 4 };
        thetas[0][2] = new double[] { 3, 2, 1 };
        thetas[1] = new double[2][];
        thetas[1][0] = new double[] { 8, 4, 4, 3 };
        thetas[1][1] = new double[] { 7, 2, 3, 10 };
    }
    public static void ForwardPropogateTest()
    {
        Console.WriteLine("NEURAL NETWORKS - FORWARD PROPOGATION");
        double[][] examples = {new double[]{1,4,5},
            new double[]{1,2,8 },
            new double[]{1,3,9 }
        };
        double[][][] thetas = new double[2][][];
        thetas[0] = new double[3][];
        thetas[0][0]  = new double[]{ 2,5,4};
        thetas[0][1] = new double[] { 9, 1, 4 };
        thetas[0][2] = new double[] {3, 2, 1 };
        thetas[1] = new double[2][];
        thetas[1][0] = new double[] { 8, 4, 4,3 };
        thetas[1][1] = new double[] {7, 2, 3,10 };
        foreach(double[] example in examples)
        {
            double[][] layers = RegressionImpl.ForwardPropogate(thetas, example);
            foreach(double[] layer in layers)
            {
                foreach(double val in layer)
                {
                    Console.Write(val + " ");
                }
                Console.WriteLine();
            }
            Console.WriteLine();
        }
        Console.WriteLine();
        double[][] grad = RegressionImpl.BackPropogation(thetas, examples[1], new double[] { 108, 72 });
        Console.ReadKey();
    }
}
