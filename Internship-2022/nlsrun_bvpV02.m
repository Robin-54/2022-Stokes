function nlsrun_bvpV02
%% Creates a neural network with a single input and output
% Modelling a boundary value problem
% Uses sgd_singleV02 function (stochastic gradient descent)
% Fixed amount of layers and neurons [1,2,3,1]

% clear; close all;

MaxIts = 1e5;

x = [0 .1 .2 .5 .8 .9 1];

% Chebfun
domain = [0,  1];
N = chebop(@(x,u) -0.01.*diff(u,2)+u-(x>0.4).*(x<0.6), domain);
rhs = 0;
N.bc = @(x,u) [u(0); u(1)];
u = solvebvp(N, rhs);
y = u(x);
% figure(1)
% plot(x,y,'r:o'); hold on; plot(u,'b'); hold off;
% legend('Training Data', 'chebfun');
% title('Input Data');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialize weights/biases and call optimizer
rng(5000);
Pzero = 5*randn(17,1);

finalP = sgd_singleV02(Pzero, x, y, MaxIts);

finalW2 = zeros(2,1);
finalW3 = zeros(3,2);
finalW4 = zeros(1,3);
finalW2(:) = finalP(1:2);
finalW3(:) = finalP(3:8);
finalW4(:) = finalP(9:11);
finalb2 = finalP(12:13);
finalb3 = finalP(14:16);
finalb4 = finalP(17);

% grid plot
N = 100;
xvals = linspace(0, 1,N+1);

xk = xvals;
a2 = activate(xk,finalW2,finalb2);
a3 = activate(a2,finalW3,finalb3);
a4 = activate(a3,finalW4,finalb4);
Aval = a4;


figure(1)
plot(xvals,Aval,x,y,'o',xvals,u(xvals),'--');
legend('NNIP', 'Training Data', 'true soln')

% error = norm(Aval-u(xvals));
% fprintf("Error estimate = %10.5e\n", norm( (Aval- u(xvals))/N))

end