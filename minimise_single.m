function result = minimise_single(X, Y, Weight, Bias)
% Using the stochastic gradient method to train a neural network
% Inputs: 
%  Pval :  values for the wights and biases on NN
%  X : 2xN matrix, each col is point in R2
%  Y : 2xN matrix that classifies each point in X as [1;0] or [0;1]


% W2 = zeros(nodeList(1),2);
% W3 = zeros(nodeList(2),nodeList(1));
% W4 = zeros(nodeList(3),nodeList(2));
% W5 = zeros(2,nodeList(3));
% W2(:) = Pval(1:2*nodeList(1)); p = 2*nodeList(1);
% W3(:) = Pval(p+1:p+nodeList(1)*nodeList(2)); p = p+nodeList(1)*nodeList(2);
% W4(:) = Pval(p+1:p+nodeList(2)*nodeList(3)); p = p+nodeList(2)*nodeList(3);
% W5(:) = Pval(p+1:p+nodeList(3)*2); p = p+nodeList(3)*2;
% b2 = Pval(p+1:p+nodeList(1)); p = p+nodeList(1);
% b3 = Pval(p+1:p+nodeList(2)); p = p+nodeList(2);
% b4 = Pval(p+1:p+nodeList(3)); p = p+nodeList(3);
% b5 = Pval(p+1:p+2);

layers = length(Bias)+1;

Diff = Y;
eta = 1;                % learning rate
Niter = 1e6;               % number of SG iterations 
counter = 1;
cost = 1;
while (counter<Niter) && (cost>1e-2)
    k = randi(length(X));         % choose a training point at random
    x = X(k);
    % Forward pass
    a = {x};
    for i = 2:layers
        a{i} = activate(a{i-1}, Weight{i-1}, Bias{i-1});
    end
    Diff(k) = abs(Y(k)-a{end});
    % Backward pass
    delta{1} = a{length(a)}.*(1-a{length(a)}).*(a{length(a)}-Y(k));
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
    cost = norm(Diff);
    counter = counter +1;
end
fprintf("%7d: \t %10.6e\n", counter, cost )
result = {Weight, Bias};
end