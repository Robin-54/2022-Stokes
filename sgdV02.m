function sgdV02(nodeList)

% Variable amount of nodes and layers, Uses minimise function

rng(5000);
% Select some random points
N = 3;
X = 0.5*[rand(1,N), 1+rand(1,N), 1+rand(1,N), rand(1,N);
   rand(1,N), rand(1,N), 1+rand(1,N), 1+rand(1,N)];   % y-coords
%red_i = sqrt(X(1,:).^2+X(2,:).^2)<0.8;
red_i = false(1,length(X));
red_i([1:N, (2*N +1):(3*N)])=true;
blue_i = ~red_i;
Y = zeros(size(X));
Y(1,red_i)=1;
Y(2,blue_i)=1;

Blue = [X(:, [(N+1):(2*N) (3*N+1):(4*N)])];
Red = [X(:,[(1:N) (2*N+1:3*N)])];


% nodeList = [2,2,3,2];  % Amount of nodes in each layer
layers = length(nodeList);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialize weights/biases and call optimizer

weight = {};
bias = {};

for i = 1:layers - 1
    weight{i} = 2*randn(nodeList(i+1), nodeList(i));
    bias{i} = 2*randn(nodeList(i+1),1);
end

finalP = minimiseV07(X, Y, weight, bias);

% Check this function 
finalWeight = {};
finalBias = {};
for i = 1:length(finalP{1})
    finalWeight{i} = finalP{1}{i};
    finalBias{i} = finalP{2}{i};
end

% grid plot
N = 25;
xvals = linspace(0,1,N+1);
yvals = linspace(0,1,N+1);
for k1 = 1:N+1
    xk = xvals(k1);
    for k2 = 1:N+1
        yk = yvals(k2);
        a{1} = [xk;yk];
        for i = 2:layers-1
            a{i} = activate(a{i-1}, finalWeight{i-1},finalBias{i-1});
        end
        Aval(k2,k1) = a{length(a)}(1);
        Bval(k2,k1) = a{length(a)}(2);
    end
end

[Xgrid,Ygrid] = meshgrid(xvals,yvals);


figure(2)
%clf
a2 = subplot(1,1,1);
Mval = Aval>Bval;
contourf(Xgrid,Ygrid,Mval,[0.5 0.5])
hold on
colormap([1 1 1; 0.8 0.8 0.8])
plot(Red(1,:),Red(2,:),'ro', Blue(1,:),Blue(2,:),...
   'bx','MarkerSize',12,'LineWidth',4);
a2.XTick = [0 1];
a2.YTick = [0 1];
a2.FontWeight = 'Bold';
a2.FontSize = 16;
xlim([0,1])
ylim([0,1])

%save
% figure(3);
% mesh(X,Y,Mval)
% print -dpng pic_bdy.png
end

