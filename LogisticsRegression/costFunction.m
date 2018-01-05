function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
%
% Note: grad should have the same dimensions as theta
%

hypo = 1 ./ (1 + exp(-X * theta));
xx = y .* log(hypo) + ( 1 - y) .* log( 1 - hypo);
J = -1/m * sum(xx);

del1 = sum((hypo - y) .* X(:,1));
del2 = sum((hypo - y) .* X(:,2));
del3 = sum((hypo - y) .* X(:,3));

grad(1) = 1/m * del1;
grad(2) = 1/m * del2;
grad(3) = 1/m * del3;


% =============================================================

end
