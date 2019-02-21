function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
c_test= [0.01 0.03 0.1 0.3 1 3 10 30];
sigma_test=[0.01 0.03 0.1 0.3 1 3 10 30];

m=size(c_test,2);
n=size(sigma_test,2);
predictions = zeros(size(yval));
error = mean(double(predictions~=yval));
for i=1:m
  for j=1:n
    model= svmTrain(X, y, c_test(i), @(x1, x2) gaussianKernel(x1, x2, sigma_test(j))); 
    predictions=svmPredict(model,Xval);
    error_itr = mean(double(predictions~=yval));
    if error_itr < error
      error=error_itr;
      C = c_test(i);
      sigma=sigma_test(j);
    end
  end
end

printf("final values of C and sigm returned by dataaa3paramas are %f and %f resp.",C,sigma);




% =========================================================================

end
