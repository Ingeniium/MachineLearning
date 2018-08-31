using System;
using System.Collections.Generic;
using System.Linq;

class RegressionImpl
{
    public delegate double Hypothesis(double[] thetas, double[] variables);
    public delegate double Cost(double[] thetas, double[][] examples, double[] results, int test_index);

    /*Returns cost of using a linear combination function with coefficents theta
     * on data examples to approximate their results.*/
    public static double LinearCost(double[] thetas, double[][] examples, double[] results, int test_index = 0)
    {
        if (test_index == examples.Length)
        {
            test_index = 0;
        }
        double total = 0;
        for (int i = test_index; i < examples.Length; i++)
        {
            double example_cost = LinearRegressionHypothesis(thetas, examples[i]);
            total += Math.Pow(example_cost, 2);
        }
        total /= 2 * examples.Length;
        return total;
    }

    /*/*Returns cost of using the sigmoid of a linear combination function 
     * with coefficents theta on data examples to approximate their results.*/
    public static double LogisticCost(double[] thetas, double[][] examples, double[] results, int test_index = 0)
    {
        if (test_index == examples.Length)
        {
            test_index = 0;
        }
        double total = 0;
        for (int i = test_index; i < examples.Length; i++)
        {
            double example_cost = LogisticRegressionHypothesis(thetas, examples[i]);
            example_cost = results[i] * Math.Log(example_cost) + (1 - results[i]) * Math.Log(1 - example_cost);
            total += example_cost;
        }
        total /= -1 * examples.Length;
        return total;
    }


    /*examples expected to be a rectangular array with rows being training examples and columns
     * being individual features.Each feature is expected to be multiplied by their respective
     * theta in thetas.The results are the output of each training example.Hypothesis can be
     * a regression function.The first column of the example array is expected to be filled with ones.*/
    private static double GradientDescent(Hypothesis hypothesis, Cost cost, double[] thetas, double[][] examples, double[] results, double learning_rate, double lambda, int test_index)
    {
        int iter = 0;
        double multiplier = 1.0 / examples.Length * learning_rate;
        double prev_cost = Double.MaxValue;
        double cur_cost = 0;
        int length = test_index;//Length is set to the test index so as to train with all examples up to that index
        while (Math.Abs(cur_cost - prev_cost) > .00000001)
        {
            iter++;
            double[] differences = new double[length];
            for (int i = 0; i < length; i++)
            {
                differences[i] = hypothesis(thetas, examples[i]) - results[i];
            }
            for (int i = 0; i < thetas.Length; i++)
            {
                double derivative = 0;
                for (int j = 0; j < differences.Length; j++)
                {
                    derivative += differences[j] * examples[j][i];
                }
                derivative *= multiplier;
                thetas[i] -= derivative + lambda / examples[0].Length * thetas[i] * learning_rate;
            }
            prev_cost = cur_cost;
            cur_cost = cost(thetas, examples, results, test_index);
        }
        //Console.WriteLine(cur_cost);
        return cur_cost;
    }

    /* Returns a set of 'thetas', or coeffeicients that can be multiplied to a set of
    * variables to minimize the difference between the sum of those products and said
    * set's respective, true result. Each row in examples is expected to be a set of 
    * variables, with the number of rows being equal to the number of results.
    * The param examples is expected to have each row having the same number of columns.
    * The returned value's size will be equivalent to the number of columns are in a row in
    * examples.The training percent is a number between oone and 0 that represents a fraction of the
    example data used to train the algorithm.Any data not used to directly train the algorithm will be instead used
    to test a solution for general accuracy and hence influence which thetas are returned.*/
    public static double[] LinearRegression(double[][] examples, double[] results, double training_percent = 1)
    {
        return Regression(LinearRegressionHypothesis, LinearCost, examples, results, training_percent);
    }

    public static double[] LogisticRegression(double[][] examples, double[] results, double training_percent = 1)
    {
        return Regression(LogisticRegressionHypothesis, LogisticCost, examples, results, training_percent);
    }

    private static double[] Regression(Hypothesis hypothesis, Cost cost, double[][] examples, double[] results, double training_percent)
    {
        if (examples == null || results == null)
        {
            throw new ArgumentNullException(" One or more args are null!");
        }
        if (results.Count() != examples.Count())
        {
            throw new ArgumentException("Number of training examples should be " +
                "equal to the number of results!");
        }
        if (training_percent <= 0 || training_percent > 1)
        {
            throw new ArgumentException("The percentage of the training set to use for training "
                + "must be greater than 0 and less than or equal to 100 " +
                "(@param training_percent no be greater than 1)");
        }
        //Ceiling functions prevent test_index from being zero and from it not being examples.Length in case training percent is 1.
        int test_index = (int)Math.Ceiling(examples.Length * training_percent);
        double[] thetas = null;
        double min_cost = Double.MaxValue;
        /*Regularization using the lambda local variable done in loop below;only relevant when training_percent is less than 1
        and hence testing examples are present.*/
        for (double lambda = 0; lambda <= 1; lambda += .1)
        {
            double[] current_thetas = new double[examples[0].Count()];
            double cur_cost = GradientDescent(hypothesis, cost, current_thetas, examples, results, .001, lambda, test_index);
            if (cur_cost < min_cost)
            {
                thetas = current_thetas;
                min_cost = cur_cost;
            }
        }
        return thetas;
    }

    private static double LinearRegressionHypothesis(double[] thetas, double[] variables)
    {
        double total = 0;
        for (int i = 0; i < variables.Length; i++)
        {
            total += thetas[i] * variables[i];
        }
        return total;
    }

    /*Returns a number bounded between 0 and 1 based on the variables of a training example
     * and the coefficents theta.*/
    private static double LogisticRegressionHypothesis(double[] thetas, double[] variables)
    {
        return 1.0 / (1.0 + Math.Pow(Math.E, -1 * (LinearRegressionHypothesis(thetas, variables))));
    }

    /*Returns the derivative of the sigmoid function*/
    private static double[] SigmoidGradient(double[][] thetas, double[] variables)
    {
        double[] gradients = new double[thetas.Length];
        for (int i = 0; i < gradients.Length; i++)
        {
            double sum = 0;
            for (int j = 0; j < thetas[i].Length; j++)
            {
                sum += thetas[i][j] * variables[j];
            }
            double e_term = Math.Pow(Math.E, -1 * sum);
            gradients[i] = e_term / Math.Pow(1 + e_term, 2);
        }
        return gradients;
    }

    /*Returns a new set of training examples such that the difference in value ranges 
  * for each type of variable are smaller by subtracting the averages and dividing 
  * by their ranges in order to increase performance. If this is used to feed training data to other functions,
    YOU must normalize any data you are using as inputs to the trained functions.*/
    public static double[][] MeanNormalize(double[][] examples)
    {
        double[][] new_examples = new double[examples.Length][];
        double[] maxes = new double[examples[0].Length];
        double[] mins = new double[examples[0].Length];
        double[] totals = new double[examples[0].Length];
        for (int i = 0; i < examples[0].Length; i++)
        {
            maxes[i] = Double.MinValue;
            mins[i] = Double.MaxValue;
        }
        for (int example = 0; example < examples.Length; example++)
        {
            for (int variable = 1; variable < examples[0].Length; variable++)
            {
                totals[variable] += examples[example][variable];
                if (examples[example][variable] > maxes[variable])
                    maxes[variable] = examples[example][variable];
                if (examples[example][variable] < mins[variable])
                    mins[variable] = examples[example][variable];
            }
        }
        for (int example = 0; example < examples.Length; example++)
        {
            new_examples[example] = new double[examples[0].Length];
            new_examples[example][0] = 1;
            for (int variable = 1; variable < examples[0].Length; variable++)
            {
                double average = totals[variable] / examples.Length;
                double range = maxes[variable] - mins[variable];
                new_examples[example][variable] = (examples[example][variable] - average)
                    / range;
            }
        }
        return new_examples;
    }

    /*Returns the the values within each layer forward propagated with the input weights
     * theta. The param thetas doesn't have to be a rectangular array;The length of an
     * array of doubles within thetas is length of weights/units (including the bias
     * unit) within that layer.The last layer is the final prediciton results.*/
    public static double[][] ForwardPropogate(double[][][] thetas, double[] example)
    {
        double[] current_layer = example;
        double[] next_layer;
        double[][] layers = new double[thetas.Length + 1][];//Total number of layers is equal to the number of two dimensional theta arrays + 1 for the end results.
        layers[0] = new double[example.Length];
        Array.Copy(example, layers[0],example.Length);
        /*Each layer after the first is going to be a sigmoid of the sum of the products
        between each of the example's variable and their corresponding theta value*/
       for (int layer = 1; layer < layers.Length; layer++)
        {
            /*layer - 1 as the values for the upcoming layer and the number of the units
             * relies on the current layers theta weights.*/
            int threshold = layer < layers.Length - 1 ? 1 : 0;//threshold values serves to prevent bias value being added to last layer.
            next_layer = new double[thetas[layer - 1].Length + threshold];
            next_layer[0] = 1;//Account for bias unit for next layer.
            for (int variable = threshold; variable < next_layer.Length; variable++)
            {
                next_layer[variable] = LogisticRegressionHypothesis(thetas[layer - 1][variable - threshold], current_layer);
            }
            current_layer = next_layer;
            layers[layer] = current_layer;
        }
        return layers;
    }

    public static double[][] BackPropogation(double[][][] thetas, double[] example, double[] results)
    {
        double[][] layers = ForwardPropogate(thetas, example);
        double[][] layer_errors = new double[layers.Length][];
        /*The error of the last layer is simply the differences between the actual results and
         * the predicted ones.*/
        layer_errors[layer_errors.Length - 1] = Subtract(layers[layers.Length - 1], results);
        for (int layer = layers.Length - 2; layer > 0; layer--)
        {
            //Bias units do not have error, hence it is length - 1.
            layer_errors[layer] = new double[thetas[layer].Length];
            for (int error = 1; error < layer_errors[layer].Length; error++)
            {
                for (int i = 0; i < thetas[layer].Length - 1; i++)
                {
                    layer_errors[layer][error] += layer_errors[layer + 1][error]
                       * thetas[layer][i][error];
                }
            }
        }
        return ComputeNeuralNetworkPartialDerivatves(layers,thetas,layer_errors);
    }

    private static double[][] ComputeNeuralNetworkPartialDerivatves(double[][] layers,double[][][] thetas, double[][] layer_errors)
    {
        double[][] derivatives = new double[layers.Length][];
        for (int i = 0;i < derivatives.Length;i++)
        {
            double[] gradients = SigmoidGradient(thetas[i], layers[i]);
            derivatives[i] = new double[layers[i].Length];
            for (int j = 0;j < layers[i].Length;j++)
            {
                derivatives[i][j] = layers[i][j] * layer_errors[i + 1][j] * gradients[j];
            }
        }
        return derivatives;
    }

    private static double[] Subtract(double[] lhs, double[] rhs)
    {
        if (lhs.Length != rhs.Length)
        {
            throw new ArgumentException("lhs and rhs are supposed to be same size!");
        }
        double[] difference = new double[lhs.Length];
        for (int i = 0; i < lhs.Length; i++)
        {
            difference[i] = lhs[i] - rhs[i];
        }
        return difference;
    }

}
