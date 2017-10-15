function [theta, ll] = newton(X,y)
    % newton's method
    % rows of X are training samples
    % rows of Y are corresponding -1/1 values

    % output ll: vector of log-likelihood values at each iteration
    % ouptut theta: parameters

    [m,n] = size(X); %99 2
    max_iters = 100000;

    X = [ones(size(X,1),1), X]; % append col of ones for intercept term
    
    theta = ones(n+1, 1);  % initialize theta
    theta_old = zeros(n+1, 1); % make them very different
    
    threshold = 1e-5;
    while norm(theta - theta_old) > threshold
        disp(norm(theta - theta_old));
        hessian = hessian_of_empirical_loss(theta, X, y);
        gradient = gradient_of_empirical_loss(theta, X, y);
        theta_old = theta;
        theta = theta - inv(hessian) * gradient;
    end
% ans =
% 
%   -25.5466
%     6.4558
%     5.3584
end



function val = J(X, y, theta)
% calculate the empirical loss of X, y given theta
    [m, n] = size(X);
    loss = 0;
    for row = 1:m
        loss = loss + log(1+exp(-z(y(row), X(row,:), theta)));
    end
    val = loss/m;
end

function a=sigmoid(z)
    a = 1.0 ./ (1.0+exp(-z));
end

function H=hessian_of_empirical_loss(theta, X, y)
% build the hessian matrix for theta, x, y
    [m, n] = size(X);
    H = zeros(n, n);
    for hessianX = 1:n
        for hessianY = 1:n
            hessian = 0;
            for row = 1:m
                hessian = hessian + y(row)^2 * X(row, hessianX) * X(row, hessianY) * exp(-z(y(row), X(row,:), theta)); 
            end
             H(hessianX, hessianY) = hessian / m; 
        end
    end
    
end

function gi=gradient_of_empirical_loss(theta_old, X, y)
% build the gradient vector of theta, x, y
    [m, n] = size(X);
    % zeros(n) will create a square
    gi = zeros(n, 1);
    % fill the gradient one by one
    % n thetas
    for k = 1:n
        gradient = 0;
        for row = 1:m
           gradient = gradient - sigmoid(z(y(row), X(row,:), theta_old)) * y(row) * X(row, k);
        end
        gi(k) = gradient / m;
    end
end

function out=z(y, X_vector, theta_vector)
    % y is the result at current row
    % X_vector(n+1, 1) are the parameters of current row
    % theta_vector (n+1, 1) is the old theta
    out = y * X_vector * theta_vector; 
end

% X = dlmread("logistic_x.txt");
% y = dlmread("logistic_y.txt");
% 
% [theta, ll] = newton(X, y);
