function [result, i, cost] = sgd_singleV02(Pval, X, Y, MaxIts)

% Like the first version but optimised

W2 = zeros(2,1);
W3 = zeros(3,2);
W4 = zeros(1,3);
W2(:) = Pval(1:2);
W3(:) = Pval(3:8);
W4(:) = Pval(9:11);
b2 = Pval(12:13);
b3 = Pval(14:16);
b4 = Pval(17);

scale = 2;

Diff = Y;
cost = 1;
PrevCost = norm(Diff);
i = 1;
while (i<= MaxIts)
%     if i==MaxIts
%         fprintf("%10.6e\n",cost)
%     end
    i = i+1;
    k = randi(length(X));         % choose a training point at random
    x = X(k);
    eta = scale;
    % Forward pass
    a2 = activate(x,W2,b2);
    a3 = activate(a2,W3,b3);
    a4 = activate(a3,W4,b4);
    Diff(k) = abs(Y(k)-a4);
    % Backward pass
    delta4 = a4.*(1-a4).*(a4-Y(k));
    delta3 = a3.*(1-a3).*(W4'*delta4);
    delta2 = a2.*(1-a2).*(W3'*delta3);
    % Gradient step
    W2 = W2 - eta*delta2*x';
    W3 = W3 - eta*delta3*a2';
    W4 = W4 - eta*delta4*a3';
    b2 = b2 - eta*delta2;
    b3 = b3 - eta*delta3;
    b4 = b4 - eta*delta4;

     cost = norm(Diff);
%     if(mod(i,100000)==0)
%         cost = norm(Diff);
% %         fprintf("%7d: \t %10.6e\n", i, cost )
%         if cost/PrevCost > 0.994
%             SlackCounter = SlackCounter + 1;
%             if SlackCounter == 3
%                 scale = scale*0.95;
%             end
%         else
%             SlackCounter = 0;
%         end
%         PrevCost = cost;
%     end

end

result = zeros(17,1);
result(1:2) = W2;
result(3:8) = W3;
result(9:11) = W4;
result(12:13) = b2;
result(14:16) = b3;
result(17) = b4;
end