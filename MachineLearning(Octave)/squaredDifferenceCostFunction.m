function difference = squaredDifferenceCostFunction(X,y,theta)

% X is a matrix containing the training examples and their features.
% y is the known results of said training examples
% theta is a vector  containing the set of theta's
  %for each feature in x
m = size(X,1);
predictions = X * theta;
squareError = (predictions - y).^2; % taking an element by element subtraction of theta * features
	% and the known y's.It also squares those individual differences.
if(size(squareError,2) != 1),
squareError = squareError(:);
end;
difference = 1 / (2*m) * sum(squareError); % it is customary to divide the average by half.