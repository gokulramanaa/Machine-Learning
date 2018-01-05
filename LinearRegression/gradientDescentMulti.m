function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

		predictions = X * theta;
		for j = 1:size(X,2)
			del(j) = (1/m) * sum((predictions - y) .* X(:,j));
			
			temp(j) = theta(j) - (alpha * del(j));
		endfor
		for k = 1:size(X,2)
			theta(k) = temp(k);
		endfor
			
	%del1 = (predictions - y) .* X(:,1);
	%del2 = (predictions - y) .* X(:,2);
	%del3 = (predictions - y) .* X(:,3);
	%del1_r = (1/m) * sum(del1);
	%del2_r = (1/m) * sum(del2);
	%del3_r = (1/m) * sum(del3);
	
	%temp1_pre = predictions 
	%temp1 = theta(1) - (alpha * del1_r);
	%temp2 = theta(2) - (alpha * del2_r);
	%temp3 = theta(3) - (alpha * del3_r);
	%theta(1) = temp1;
	%theta(2) = temp2;
	%theta(3) = temp3;

	
	
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
