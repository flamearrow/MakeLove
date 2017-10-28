% 5.b.i
% do_unweighted_linear_regression();

% 5.b.ii
% tau_vector = [5];
% do_weighted_linear_regression(tau_vector);

% 5.b.iii
%tau_vector = [1, 10, 100, 1000];
%do_weighted_linear_regression(tau_vector);


% 5.c.i
% [lambdas, train_qso, test_qso] = reload_data();
% smoothed_train_qso = smooth_data(train_qso, lambdas);
% smoothed_test_qso = smooth_data(test_qso, lambdas);


% 5.c.ii
%[training_error, estimated_fleft] = estimate_f_left(smoothed_train_qso);
% average training_error: 769.7348

% 5.c.iii
% [training_error_test, estimated_fleft_test] = estimate_f_left_with_test_data(smoothed_train_qso, smoothed_test_qso);
% average trainering_error: 30.3842

% test_output_example_1 = estimated_fleft_test(1,:);
% test_input_example_1 = smoothed_test_qso(1,:);
% 
% hold on;
% title("example 1");
% scatter(lambdas, test_output_example_1, 20, 'r', '*');
% scatter(lambdas, test_input_example_1, 20, 'g', '*');
% hold off;
% 
% test_output_example_6 = estimated_fleft_test(6,:);
% test_input_example_6 = smoothed_test_qso(6,:);
%  
% hold on;
% title("example 6");
% scatter(lambdas, test_output_example_6, 20, 'r', '*');
% scatter(lambdas, test_input_example_6, 20, 'g', '*');
% hold off;


function [training_error, estimated_fleft] = estimate_f_left_with_test_data(smoothed_train_qso, smoothed_test_qso)
   % calculate a matrix of m * 50 for estimated fleft values
   % the width is 50 because left only ranges from 1150 to 1199
   % input_data: m X n matrix, feed in training data and testing data
   
   [m, n] = size(smoothed_test_qso);
   estimated_fleft = zeros(m, n);
   
   for j = 1:m
      current_vector = smoothed_test_qso(j,:);
      % furthest distance from current vector
      h_value = h_test(smoothed_train_qso, current_vector);
      for i = 1:n
          % find 3 closest rows
          indices = neighb_test(smoothed_train_qso, current_vector, 3);
          upper = 0;
          lower = 0;
          for index = indices
              distance = d(smoothed_train_qso(index,:), current_vector, 1300);
              fleft = smoothed_train_qso(j, i);
              upper = upper + ker(distance/h_value) * fleft;
              lower = lower + ker(distance/h_value);
          end
          estimated_fleft(j, i) = upper/lower;
      end
   end
   
   training_error = zeros(m, 1);
   for i = 1:m
       % cost is fleft, calculate the entire row
       training_error(i,1) = d(smoothed_test_qso(i,:), estimated_fleft(i,:), 1150);
   end
end

function result = h_test(X, vector)
% calculate the max distance from vector
% distance is defined in d and fright
% X: input matrix mXn, n=450 here
    result = 0;
    [row_count, ~] = size(X);
    for i=1:row_count
        distance = d(vector, X(i,:), 1300);
        if distance > result
            result = distance;
        end
    end
end

function indices = neighb_test(X, vector, k)
% find k row indices from X that are closest to 
% vector, closest defined in d
% X: m X n matrix
    [m,~] = size(X);
    distance_column = zeros(m,1);
    for i = 1:m
        distance_column(i, 1) = d(vector, X(i,:), 1300);
    end
    [~, original_positions] = sort(distance_column);
    indices = original_positions(1:k, 1);
end



function [training_error, estimated_fleft] = estimate_f_left(input_data)
   % calculate a matrix of m * 50 for estimated fleft values
   % the width is 50 because left only ranges from 1150 to 1199
   % input_data: m X n matrix, feed in training data and testing data
   
   [m, n] = size(input_data);
   estimated_fleft = zeros(m, n);
   
   for j = 1:m
      % furthest distance from current vector
      h_value = h(input_data, j);
      for i = 1:n
          % find 3 closest rows
          indices = neighb(input_data, j, 3);
          upper = 0;
          lower = 0;
          for index = indices
              distance = d(input_data(index,:), input_data(j,:), 1300);
              fleft = input_data(j, i);
              upper = upper + ker(distance/h_value) * fleft;
              lower = lower + ker(distance/h_value);
          end
          estimated_fleft(j, i) = upper/lower;
      end
   end
   
   training_error = zeros(m, 1);
   for i = 1:m
       % cost is fleft, calculate the entire row
       training_error(i,1) = d(input_data(i,:), estimated_fleft(i,:), 1150);
   end
end


function result = d(f1_vector, f2_vector, starting_phi_index)
% calculte the d between two vectors
% calculate from starting_phi_index for f(right) calculations
% both vectors are supposed to be 1*450
    % our index starts from 1150
    offset_index = starting_phi_index - 1150 + 1;
    difference_sqr_vector = (f1_vector - f2_vector) .^2;
    result = sum(difference_sqr_vector(1,offset_index:end));
end

function result = ker(t)
    result = max(1-t, 0);
end

function indices = neighb(X, row_index, k)
% find k row indices from X that are closest to 
% X(row_index,:), closest defined in d
% X: m X n matrix
    target_row = X(row_index,:);
    [m,n] = size(X);
    distance_column = zeros(m,1);
    for i = 1:m
        distance_column(i, 1) = d(target_row, X(i,:), 1300);
    end
    [~, original_positions] = sort(distance_column);
    % staring from 2, the first is row_index itself
    indices = original_positions(2:2+k-1, 1);
end

function result = h(X, row_index)
% calculate the max distance from X(row_index:)
% distance is defined in d and fright
% X: input matrix mXn, n=450 here
    result = 0;
    [row_count, ~] = size(X);
    for i=1:row_count
        distance = d(X(row_index,:), X(i,:), 1300);
        if distance > result
            result = distance;
        end
    end
end

function smoothed_data = smooth_data(unsmoothed_data, lambdas)
% smooth the data using tau=5
% unsmoothed_data: a X 450
% lambdas: 1 X 450
    [a, m] = size(unsmoothed_data);
    smoothed_data = zeros(a, m);
    % apply the smooth to each row from the second row
    for i = 1:a
        y = unsmoothed_data(i, :)'; 
        result = create_weighted_results(lambdas, y, 5, lambdas);
        smoothed_data(i,:)=result';
    end
end

function weighted_results = create_weighted_results(X, y, tau, queries)
% map y to a weighted results, smooth the data
    [m,~] = size(X);
    weighted_results = zeros(m, 1);
    for i = 1:m
        % for wieghted linear regression, each query value(queries(i,1)) needs 
        % to be passed to algorithm
        theta = weighted_linear_regression(X, y, tau, queries(i, 1));
        weighted_results(i, 1) = theta(2, 1) * X(i, 1) + theta(1, 1);
    end
end


% for 5.b.i
function do_unweighted_linear_regression()
    [lambdas, train_qso, test_qso] = reload_data();
    X = lambdas;
    y = train_qso(2, :)';

    theta = unweighted_linear_regression(X, y);
    scatter(X, y, 5, 'b');
    hold on;
    scatter(X, theta(2, 1) * X + theta(1, 1), 5, 'r', '*');
    hold off;
end


% for 5.b.ii
function do_weighted_linear_regression(tau)
    [lambdas, train_qso, test_qso] = reload_data();
    X = lambdas;
    y = train_qso(2, :)';

    scatter(X, y, 10, 'b');
    
    colors = ['r' 'g' 'c' 'y' 'm' 'c'];
    hold on;
    [m,~] = size(X);
    j = 1;
    for taui = tau
        weighted_results = zeros(m, 1);
        for i = 1:m
            % for wieghted linear regression, each query value(lambdas(i,1)) needs 
            % to be passed to algorithm
            theta = weighted_linear_regression(X, y, taui, lambdas(i, 1));
            weighted_results(i, 1) = theta(2, 1) * X(i, 1) + theta(1, 1);
        end
        scatter(X, weighted_results, 10, colors(j));
        j = j+1;
    end
    hold off;
end

function [lambdas, train_qso, test_qso] = reload_data()
    clear;
    load quasar_train.csv  quasar_train;
    lambdas = quasar_train(1, :)';
    train_qso = quasar_train(2:end, :);
    load quasar_test.csv quasar_test;
    test_qso = quasar_test(2:end, :);
end

function theta = unweighted_linear_regression(X,y)
    % X: mXn training examples
    % Y: mX1 results

    % ouptut theta: close form solution of unweight linear regression

    [m,n] = size(X);

    X = [ones(size(X,1),1), X]; % append col of ones for intercept term
    % theta = inv(X' * X) * X' * y;
    theta = (X' * X) \ X' * y;
end

function theta = weighted_linear_regression(X,y,t,queryX)
    % X: mXn training examples
    % Y: mX1 results
    % t weight parameter
    % queryX: 1X1 the X value this algorithm is going to be run on

    % ouptut theta: close form solution of unweight linear regression

    [m, n] = size(X);

    X = [ones(size(X,1),1), X]; % append col of ones for intercept term

    W = weight_matrix(t, X, queryX);

%     theta = inv(X' * W * X) * X' * W * y;
    theta = (X' * W * X) \ X' * W * y;
    
end

function W = weight_matrix(t, X, queryX)
    % build a weight matrix from t and X
    % t: weight parameter
    % X: mXn - here X is 2X1
    % queryX: X value to run on
    [m, ~] = size(X);
    W = zeros(m,m);
    for i = 1:m
        % what if X has more than 2 columns?
        W(i,i) = exp(-(queryX - X(i, 2)) ^ 2 / (2 * t ^ 2));
    end
end



