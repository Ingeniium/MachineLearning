function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % nsumumber of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
	theta_size = length(theta);
% I created new_theta to use in order to do a simultaneous update on
% my theta values; rather than use already updated theta values for partial
% derivatives in the same step, I don't use update thetas them until I have all updated 
% versions available
	new_theta = theta;
% multiplying matrix X by theta gives a row vector that has each row being
% each training example's sum of the products of features and their respective thetas.
    predictions = X * theta;
	differences = predictions - y;
	for min_theta = 1 : theta_size,
% The next line subtracts the partial derivatives of the squared error cost function from
% each feature specific theta value. The transpose of the differences are used in order
% to multiply them with the columnar set of feature training example instances
% the theta is designated to.
			new_theta(min_theta) = theta(min_theta) - 1 / m * alpha * differences' * X(:,min_theta);
			end;
	theta = new_theta;			






    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
