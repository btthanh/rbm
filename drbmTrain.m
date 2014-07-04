function params = drbmTrain(params, opts, data)

m          = size(data.train_x, 1);
lambda     = opts.lambda;

kk = randperm(m);
for l = 1 : m
    x1          = data.train_x(kk(l), :);
    y1          = data.train_y(kk(l), :);
    
    temp        = params.c + params.W * x1';
    pos         = sigm(temp + params.U * y1');
    neg         = bsxfun(@plus, params.U, temp);
    prob        = exp(neg) + 1;
    prob        = bsxfun(@rdivide, prob, max(prob, [], 2));
    prob        = gather(prod(prob)) .* exp(params.d');
    prob        = prob / sum(prob);
    neg         = bsxfun(@times, sigm(neg), prob);

    temp        = pos - sum(neg, 2);
    params.W	= params.W + lambda * temp * x1;
    params.U	= params.U + lambda * (pos * y1 - neg);
    params.c	= params.c + lambda * temp;
    params.d	= params.d + lambda * (y1 - prob)';
end

end
