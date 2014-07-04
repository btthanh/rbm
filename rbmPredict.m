function p = rbmPredict(params, X)

m = size(X, 1);
K = size(params.d, 1);
hiddenSize = size(params.c, 1);

g = gpuDevice;
if g.FreeMemory >= 1 % GPU has high memory
    batchSize  = 100;
    numbatches = m / batchSize;
        
    p = zeros(m, 1);
    for l = 1 : numbatches
        idx  = (l - 1) * batchSize + 1 : l * batchSize;
        
        temp = bsxfun(@plus, X(idx, :) * params.W', + params.c');
        temp = exp(bsxfun(@plus, repmat(temp, 1, K), reshape(params.U, 1, numel(params.U)))) + 1;
        temp = reshape(temp', hiddenSize, batchSize * K);
        temp = reshape(temp', K, batchSize * hiddenSize);
        temp = bsxfun(@rdivide, temp, max(temp));
        temp = reshape(temp', batchSize, K * hiddenSize);
        temp = reshape(temp', hiddenSize, batchSize * K);
        temp = prod(temp) .* repmat(exp(params.d'), 1, batchSize);
        temp = reshape(temp, K, batchSize);
        temp = bsxfun(@rdivide, temp, sum(temp));
        
        [~, I] = max(temp);
        p(idx) = gather(I') - 1;
    end
else % GPU has low memory
    p = zeros(m, 1);
    for i = 1:m
        temp = X(i, :) * params.W' + params.c';
        temp = exp(bsxfun(@plus, params.U, temp')) + 1;
        temp = bsxfun(@rdivide, temp, max(temp, [], 2));
        temp = prod(temp) .* exp(params.d');
        temp = temp / sum(temp);
        [~, I] = max(temp);
        p(i) = gather(I) - 1;
    end
end

end
