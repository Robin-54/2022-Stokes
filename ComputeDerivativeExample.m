% An example of how to compute the derivative of a NN
% Uses sgd_singleV02 function (stochastic gradient descent)
% Fixed amount of layers and neurons [1,2,3,1]

clear; 
%close all;

MaxIts = 1e5;
Tol = 1.0e-6;

x = [0  .1 .2 .4 .6 .8 1]; % Note: x=0 must always be in the training set

%% Initial Value problem:
% u(0)=1  and p*u'(x)+u(x)=0  
% so the solution is u(x)=exp(-x/p)
p=.3;
domain = [0,  1];
N = chebop(@(x,u) p*diff(u)+u, domain);
rhs = 0;
N.bc = @(x,u)u(0)-1; % u(0)-1=0, i.e, u(0)=1
u = solvebvp(N, rhs);
y = u(x);

%% Options for solver
Opts = optimoptions('lsqnonlin');
Opts.MaxIterations = 1000;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialize weights/biases and call optimizer
rng(5000);
Pzero =  5*randn(17,1);

ObjFn = @(a)NNError_PINN(a);
[finalP,finalerr,res, exitflag, outp ] = ...
   lsqnonlin(ObjFn,Pzero, [], [], Opts);
fprintf('Nonlin solver did %d iterations\n', ...
   outp.iterations)

%for i=1:10
%    [finalP,finalerr] = lsqnonlin(ObjFn,finalP, [], [], Opts);
%    fprintf('Nonlin solver did %d iterations\n', ...
%   outp.iterations)
%end

[finalcost, test_points] = NNError_PINN(finalP);

% [finalP, IterationCount, FinalCost] = ...
%    sgd_singleV03(Pzero, x, y, MaxIts, Tol);
% fprintf("FinalCost: %8.3e\n", finalerr);

W2 = zeros(2,1);
W3 = zeros(3,2);
W4 = zeros(1,3);
W2(:) = finalP(1:2);
W3(:) = finalP(3:8);
W4(:) = finalP(9:11);
finalb2 = finalP(12:13);
finalb3 = finalP(14:16);
finalb4 = finalP(17);

% grid plot
N = 100;
xvals = linspace(0, 1,N+1);

xk = xvals;
a2 = activate(xk,W2,finalb2);
a3 = activate(a2,W3,finalb3);
a4 = activate(a3,W4,finalb4);
Aval = a4;

figure
plot(xvals,Aval,xvals,u(xvals),'--');
legend('NNIP', 'true soln')

% figure(2)
% plot(xvals,Aval,xvals,u(xvals),'--', test_points, u(test_points), 'o')
% legend('NNIP', 'true soln')

%% Now evaluate the derivative.
% Need the deltas for the derivative
delta4 = a4.*(1-a4);%.*(a4-Y(k));
delta3 = a3.*(1-a3).*(W4'*delta4);
delta2 = a2.*(1-a2).*(W3'*delta3);

% We'll check the derivative in 2 wasy
% 1. Interpolate (xvals, Aval) with a chebfun and compare.
% 2. The residial: p*u'(x)+u(x)    
ChebI = chebfun.interp1(xvals, Aval, 'spline');
DChebI = diff(ChebI);
% figure(2); 
% plot(xvals,W2'*delta2, xvals, DChebI(xvals),'--'); 
% title('Derivative Plot');
% 
% figure(3); 
% plot(xvals,p*W2'*delta2 + Aval); 
% title('Residual');

