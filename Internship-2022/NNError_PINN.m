function [finalcost, xvals] = NNError_PINN(Pval)

% Cost function for a differential equation
% To be used with lsqnonlin

W2 = zeros(2,1);
W3 = zeros(3,2);
W4 = zeros(1,3);
W2(:) = Pval(1:2);
W3(:) = Pval(3:8);
W4(:) = Pval(9:11);
b2 = Pval(12:13);
b3 = Pval(14:16);
b4 = Pval(17);

N = 75;
xvals = linspace(0, 1,N+1);

xk = xvals;
a2 = activate(xk,W2,b2);
a3 = activate(a2,W3,b3);
a4 = activate(a3,W4,b4);

% Need the deltas for the derivative
delta4 = a4.*(1-a4);%.*(a4-Y(k));
delta3 = a3.*(1-a3).*(W4'*delta4);
delta2 = a2.*(1-a2).*(W3'*delta3);

A2 = activate(0,W2,b2);
A3 = activate(A2,W3,b3);
A4 = activate(A3,W4,b4);
cost(1) = norm(0.3*W2'*delta2 + a4);
cost(2) = 40*norm(A4-1);
finalcost = abs([10*N*abs(A4-1), 0.3*W2'*delta2 + a4]);
%finalcost = abs(cost);
% fprintf("Cost = %7d", finalcost);

end