function params = hdrbm2Train(params, opts, data)

m          = size(data.train_x, 1);
lambda     = opts.lambda;
alpha      = opts.alpha * lambda;

kk = randperm(m);
for l = 1 : m
    x1          = data.train_x(kk(l), :);
    y1          = data.train_y(kk(l), :);
    
    % RBM Discriminative
    temp        = params.c + params.W * x1';
    pos         = sigm(temp + params.U * y1');
    neg         = bsxfun(@plus, params.U, temp);
    prob        = exp(neg) + 1;
    prob        = bsxfun(@rdivide, prob, max(prob, [], 2));
    prob        = gather(prod(prob)) .* exp(params.d');
    prob        = prob / sum(prob);
    neg         = bsxfun(@times, sigm(neg), prob);
    
    % RBM Generative
    y1Sample    = mnrnd(1, prob);
    [h1, ph1]	= sigmrnd(temp + params.U * y1Sample');
    x2          = sigmrnd(params.b' + h1' * params.W);
    py2         = exp(gather(params.d' + h1' * params.U));
    y2          = mnrnd(1, py2 / sum(py2));
    ph2         = sigm(params.c + params.W * x2' + params.U * y2');

    temp        = pos - sum(neg, 2);
    params.W	= params.W + alpha * (ph1 * x1 - ph2 * x2) + lambda * temp * x1;
    params.U	= params.U + alpha * (ph1 * y1Sample - ph2 * y2) + lambda * (pos * y1 - neg);
    params.b	= params.b + alpha * (x1 - x2)';
    params.c	= params.c + alpha * (ph1 - ph2) + lambda * temp;
    params.d	= params.d + alpha * (y1Sample - y2)' + lambda * (y1 - prob)';
end

end
