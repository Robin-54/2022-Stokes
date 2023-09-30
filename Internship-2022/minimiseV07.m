function result = minimiseV07_slow(X, Y, Weight, Bias)
% Using the stochastic gradient method to train a neural network
% Inputs: 
%  Pval :  values for the wights and biases on NN
%  X : 2xN matrix, each col is point in R2
%  Y : 2xN matrix that classifies each point in X as [1;0] or [0;1]


layers = length(Bias)+1;

eta = 0.05;                % learning rate
Niter = 4e6;               % number of SG iterations 
for counter = 1:Niter
    k = randi(length(X));         % choose a training point at random
    x = X(:, k);
    % Forward pass
    a = {x};
    for i = 2:layers
        a{i} = activate(a{i-1}, Weight{i-1}, Bias{i-1});
    end
    % Backward pass
    delta{1} = a{length(a)}.*(1-a{length(a)}).*(a{length(a)}-Y(:,k));
    for i = 2:layers-1        
        delta{i} = a{layers-i+1}.*(1-a{layers-i+1}).*(Weight{layers-i+1}'*delta{i-1});
    end
    % Gradient step
    for i = 1:layers-1
        Weight{i} = Weight{i} - eta*delta{layers-i}*a{i}';
    end
    for i = 1:layers - 1
        Bias{i} = Bias{i} - eta*delta{layers-i};
    end
end

result = {Weight, Bias};