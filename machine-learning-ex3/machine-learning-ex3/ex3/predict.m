function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
X = [ones(m, 1) X];
% num_labels = size(Theta2, 1);
size(Theta1)
size(Theta2)
o1 = sigmoid(X*Theta1');
o1 = [ones(size(o1, 1), 1) o1];
o2 = sigmoid(o1*Theta2');
[Y, I] = max(o2, [], 2);
p = I;

end
