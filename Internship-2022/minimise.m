function result = minimise(X, Y, Pval)
% Using the stochastic gradient method to train a neural network
% Inputs: 
%  Pval :  values for the wights and biases on NN
%  X : 2xN matrix, each col is point in R2
%  Y : 2xN matrix that classifies each point in X as [1;0] or [0;1]


W2 = zeros(2,2);
W3 = zeros(3,2);
W4 = zeros(2,3);
W2(:) = Pval(1:4);
W3(:) = Pval(5:10);
W4(:) = Pval(11:16);
b2 = Pval(17:18);
b3 = Pval(19:21);
b4 = Pval(22:23);

eta = 0.1;                % learning rate
Niter = 1e5;               % number of SG iterations 
for counter = 1:Niter
    k = randi(length(X));         % choose a training point at random
    x = X(:, k);
    % Forward pass
    a2 = activate(x,W2,b2);
    a3 = activate(a2,W3,b3);
    a4 = activate(a3,W4,b4);
    % Backward pass
    delta4 = a4.*(1-a4).*(a4-Y(:,k));
    delta3 = a3.*(1-a3).*(W4'*delta4);
    delta2 = a2.*(1-a2).*(W3'*delta3);
    % Gradient step
    W2 = W2 - eta*delta2*x';
    W3 = W3 - eta*delta3*a2';
    W4 = W4 - eta*delta4*a3';
    b2 = b2 - eta*delta2;
    b3 = b3 - eta*delta3;
    b4 = b4 - eta*delta4;
end

result = zeros(23,1);
result(1:4) = W2;
result(5:10) = W3;
result(11:16) = W4;
result(17:18) = b2;
result(19:21) = b3;
result(22:23) = b4;