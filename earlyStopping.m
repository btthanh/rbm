function model = earlyStopping(train, predict, params, data, opts)
    
if opts.turnOnGPU == 1
    data.train_x    = gpuArray(data.train_x);
    data.val_x      = gpuArray(data.val_x);
end
    
save 'bestParams' params;

model       = struct;
patience    = 1;
totalEpochs = 1;
minErr      = 100;

epochTimes  = 0;
allTimes    = tic;
while patience <= opts.patience
    eTemp      = tic;
    params     = train(params, opts, data);
    eTemp      = toc(eTemp);
    epochTimes = epochTimes + eTemp;

    p          = predict(params, data.val_x);
    err        = 100 * mean(p ~= data.val_y);
    disp(['Patience/Epoch: ' num2str(patience) '/' num2str(totalEpochs) '. Validation error is ' num2str(err) ' (in ' num2str(eTemp) ' seconds).']);

    if minErr > err
        minErr = err;
        save 'bestParams' params;
        patience = 1;
    else
        patience = patience + 1;
    end

    totalEpochs = totalEpochs + 1;
end

load bestParams;
model.params      = params;
model.totalEpochs = totalEpochs - 1;
model.epochTimes  = epochTimes / (totalEpochs - 1);
model.allTimes    = toc(allTimes);
model.opts        = opts;
end
