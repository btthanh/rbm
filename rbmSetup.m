function params = rbmSetup(data, opts)
    
	visibleSize = size(data.train_x, 2);
	K = size(data.train_y, 2);
	
	params   = struct;
	params.W = initializeParameters(opts.hiddenSize, visibleSize);
	params.U = initializeParameters(opts.hiddenSize, K);
	params.b = zeros(visibleSize, 1);
	params.c = zeros(opts.hiddenSize, 1);
	params.d = zeros(K, 1);
    
end
