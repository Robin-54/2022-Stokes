%% Test 

% Uses minimise function (stocastic gradient descent)


clear; close all;

x1 = [0.1,0.3,0.1,0.6,0.4,0.6,0.5,0.9,0.4,0.7];
x2 = [0.1,0.4,0.5,0.9,0.2,0.3,0.6,0.2,0.4,0.6];
X = [x1;x2];
red_i = false(1,length(X));
red_i(1:5)=true;
blue_i = ~red_i;
Y = zeros(size(X));
Y(1,red_i)=1;
Y(2,blue_i)=1;

% figure(1);
% plot(Red(1,:),Red(2,:),'ro', Blue(1,:),Blue(2,:),...
%    'bx','MarkerSize',12,'LineWidth',4);
% Axis1 = gca;
% Axis1.XTick = [0 1]; Axis1.YTick = [0 1];
% Axis1.FontWeight = 'Bold'; Axis1.FontSize = 16;
% xlim([0,1])
% ylim([0,1])

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialize weights/biases and call optimizer
rng(5000);
Pzero = 2*randn(23,1);
finalP = minimise(X, Y, Pzero);

% Check this function 
finalW2 = zeros(2,2);
finalW3 = zeros(3,2);
finalW4 = zeros(2,3);
finalW2(:) = finalP(1:4);
finalW3(:) = finalP(5:10);
finalW4(:) = finalP(11:16);
finalb2 = finalP(17:18);
finalb3 = finalP(19:21);
finalb4 = finalP(22:23);

% grid plot
N = 200;
Dx = 1/N;
Dy = 1/N;
xvals = linspace(0,1,N+1);
yvals = linspace(0,1,N+1);
for k1 = 1:N+1
    xk = xvals(k1);
    for k2 = 1:N+1
        yk = yvals(k2);
        xy = [xk;yk];
        a2 = activate(xy,finalW2,finalb2);
        a3 = activate(a2,finalW3,finalb3);
        a4 = activate(a3,finalW4,finalb4);
        Aval(k2,k1) = a4(1);
        Bval(k2,k1) = a4(2);
    end
end
[X1,Y1] = meshgrid(xvals,yvals);


figure(2)
a2 = subplot(1,1,1);
Mval = Aval>Bval;
contourf(X1,Y1,Mval,[0.5 0.5])
hold on;
colormap([1 1 1; 0.8 0.8 0.8])
plot(X(1,red_i), X(2,red_i),'ro', ...
   X(1,blue_i), X(2,blue_i),...
   'bx','MarkerSize',12,'LineWidth',4);
a2.XTick = [0 1];
a2.YTick = [0 1];
a2.FontWeight = 'Bold';
a2.FontSize = 16;
xlim([0,1])
ylim([0,1])
hold off;
% figure(3);
% mesh(X,Y,Mval)

