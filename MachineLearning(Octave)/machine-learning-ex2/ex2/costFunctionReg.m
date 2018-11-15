function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

%NOTE: We don't regularize the first theta value as its supposed to be a constant by itself
% rather than multiplied to a feature that actually does something.
predictions = X * theta;
predictions = sigmoid(predictions);
sqr_theta = theta(2 : length(theta)) .^ 2;
error = -y' * log(predictions) - (1 - y)' * log(1 - predictions);
J = 1 / m * sum(error) + lambda / (2 * m) * sum(sqr_theta);
grad = 1 / m * X' * (predictions - y);
for i = 2 : size(theta),
	grad(i) += lambda / m * theta(i);
end;



% =============================================================

end
