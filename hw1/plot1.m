% norm(theta-threa_old)
%   -25.5466
%     6.4558
%     5.3584


% thres*J
% -20.0245
%     5.0980
%     4.3148

X=dlmread('logistic_x.txt');
y=dlmread("logistic_y.txt");

% disp(X);

x1 = X(:,1);
x2 = X(:,2);
% plot(X1, X2);
% scatter(X1, X2);

% norm(theta-threa_old)
% theta = [-25.5466, 6.4558, 5.3584];

% thres*J
theta = [-20.0245, 5.0980, 4.3148];
X = [ones(size(X,1),1), X];

result = X * theta';

% disp(result);

% Plot first class
scatter(x1(result > 0.5), x2(result > 0.5), 150, 'b', '*');
% Plot second class.
hold on;
scatter(x1(result < 0.5), x2(result < 0.5), 150, 'r', '*');

% a + bx1 + cx2 = 0.5
% so x2 = (0.5 - a - bx1) / c
line_x1 = (0:0.1:8);
line_x2 = (0.5 - theta(1) - theta(2) * line_x1) ./ theta(3);

plot(line_x1, line_x2);
hold off;

