% This script assumes these variables are defined:
%
%   features - input data (row-based).
%   targert - target data (column-based).

x = features';
t = target;

trainFcn = 'trainscg';  % Levenberg-Marquardt

% Fitting Network
hiddenLayerSize = 350;
net = fitnet(hiddenLayerSize,trainFcn);

net.performParam.regularization = 0.1;

% Input and Output Pre/Post-Processing Functions
net.input.processFcns = {'removeconstantrows','mapminmax'};
net.output.processFcns = {'removeconstantrows','mapminmax'};

% Setup Division of Data for Training, Validation, Testing
% For a list of all data division functions type: help nndivide
net.divideFcn = 'dividerand';  % Divide data randomly
net.divideMode = 'sample';  % Divide up every sample
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

% Performance Function
net.performFcn = 'mse';  % Mean squared error

% Train the Network
[net,tr] = train(net,x,t,'useGPU', 'yes');

y = net(x);
e = gsubtract(t,y);
performance = perform(net,t,y);

trainTargets = t .* tr.trainMask{1};
valTargets = t  .* tr.valMask{1};
testTargets = t  .* tr.testMask{1};


trainPerformance = perform(net,trainTargets,y);
valPerformance = perform(net,valTargets,y);
testPerformance = perform(net,testTargets,y);

[~ , trainIndices] = find(isfinite(trainTargets));
[~ , testIndices] = find(isfinite(testTargets));
[~ , valIndices] = find(isfinite(valTargets));

maskSizes = [size(trainIndices) size(valIndices) size(testIndices)];

genFunction(net,'neuralNetworkFunction','MatrixOnly','yes');
