function [theta1Grad, theta2Grad] = backPropagation(Theta1,Theta2,X,second_layer,third_layer,y,num_labels,lambda)

m = size(X,1);
theta1Grad = Theta1;
theta2Grad = Theta2;
error3 = zeros(m,num_labels);
for i = 1 : m,
    for label = 1 : num_labels,	
		 error3(i,label) += third_layer(i,label) - (label == y(i));
	end;
end;
error2 = (error3 * Theta2(:,2:end)) .* sigmoidGradient([ones(size(X,1),1),X] * Theta1');
theta2Grad = (error3' * [ones(size(X,1),1),second_layer]) / m;
x_1 = [ones(size(X,1),1),X];
first_x = sigmoidGradient(x_1 * Theta1');
theta1Grad = (error2' * [ones(size(X,1),1),X]) / m;
Theta1(:,1) = 0;
Theta2(:,1) = 0;
Theta1 *= lambda / m;
Theta2 *= lambda / m;
theta2Grad += Theta2;
theta1Grad += Theta1;


		

