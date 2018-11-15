function [y second_layer] = forwardPropagation(theta_1,theta_2,X)

%Computes a set of predicted results using a two layer neural network
second_layer = sigmoid([ones(size(X,1),1),X] * theta_1');
y = sigmoid([ones(size(X,1),1), second_layer] * theta_2');