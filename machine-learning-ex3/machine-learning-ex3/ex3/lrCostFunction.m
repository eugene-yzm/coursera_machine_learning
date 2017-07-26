function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

Xth = X*theta;
% k=-y*log(sigmoid(Xth'));
% l=(-y+1)*log(1-sigmoid(Xth'));
% size(k)
% size(l)
% k-l
temp = theta;
temp(1) = 0;
theta2 = (temp')*temp;
J = 1/m * sum (-y'*log(sigmoid(Xth)) - (-y'+1)*log(1-sigmoid(Xth))) + ...
    lambda/(2*m)*theta2;
grad = 1/m * X'*(sigmoid(Xth)-y) + lambda/m * temp;
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%


grad = grad(:);

end
