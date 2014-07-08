clear all;
clc;

%%======================================================================
%% Load MNIST database files

X               = loadMNISTImages('mnist/train-images-idx3-ubyte')';
Y               = loadMNISTLabels('mnist/train-labels-idx1-ubyte');

idxLab0to4      = find(Y >= 0 & Y <= 4);
idxLab5to9      = find(Y >= 5);

data = struct;
labelSize       = 800;
unlabelSize     = 5000;
trainSize       = labelSize + unlabelSize;
valSize         = 200;
data.train_x	= [X(idxLab0to4(1:labelSize), :); X(idxLab5to9(1:unlabelSize), :)];
data.train_y	= [double(repmat(Y(idxLab0to4(1:labelSize)), 1, 5) == repmat(0:4, labelSize, 1)); zeros(unlabelSize, 5)];
data.val_x      = X(idxLab0to4(size(idxLab0to4, 1) - valSize + 1:end), :);
data.val_y      = Y(idxLab0to4(size(idxLab0to4, 1) - valSize + 1:end));
            
test_x          = loadMNISTImages('mnist/t10k-images-idx3-ubyte')';
test_y          = loadMNISTLabels('mnist/t10k-labels-idx1-ubyte');
idxTest0to4     = find(test_y >= 0 & test_y <= 4);
test_x          = test_x(idxTest0to4(1:5000), :);
test_y          = test_y(idxTest0to4(1:5000));

%%======================================================================
%% Initializing Parameters

opts = struct;
opts.hiddenSize	= 1500;
opts.batchSize	= 1;
opts.lambda     = 0.05;      % learning rate
opts.alpha      = 0.01;      % trade-off Discriminative RBM vs Generative RBM
opts.beta       = 0.1;       % trade-off D_label vs D_unlabel
opts.patience	= 15;
opts.turnOnGPU  = 1;
params = rbmSetup(data, opts);

%%======================================================================
%% Training RBM

opts.train      = @hdrbmTrain;
% train           = @hdrbmTrain;
% train           = @semiSupervisedTrain;
train           = @selfTaughtTrain;
predict         = @rbmPredict;
model           = earlyStopping(train, predict, params, data, opts);

%%======================================================================
%% Results

p               = rbmPredict(model.params, data.train_x(1:labelSize, :));
model.trainErr  = 100 * mean(p ~= Y(idxLab0to4(1:labelSize)));
disp(['Train error is ' num2str(model.trainErr) '.']);

p               = rbmPredict(model.params, data.val_x);
model.valErr    = 100 * mean(p ~= data.val_y);
disp(['Validation error is ' num2str(model.valErr) '.']);

p               = rbmPredict(model.params, test_x);
model.testErr   = 100 * mean(p ~= test_y);
disp(['Test error is ' num2str(model.testErr) '.']);

model.params.W = gather(model.params.W);
model.params.U = gather(model.params.U);
model.params.b = gather(model.params.b);
model.params.c = gather(model.params.c);
model.params.d = gather(model.params.d);
save 'BestModel' model;
