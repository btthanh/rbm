function params = semiSupervisedTrain(params, opts, data)

m          = size(data.train_x, 1);
lambda     = opts.lambda;
beta       = opts.beta * lambda;

kk = randperm(m);
for l = 1 : m
    x1          = data.train_x(kk(l), :);
    y1          = data.train_y(kk(l), :);

    if sum(y1) == 0
        temp        = params.c + params.W * x1';
        prob        = bsxfun(@plus, params.U, temp);
        prob        = exp(prob) + 1;
        prob        = bsxfun(@rdivide, prob, max(prob, [], 2));
        prob        = gather(prod(prob)) .* exp(params.d');
        prob        = prob / sum(prob);
        
        y1          = mnrnd(1, prob);

        [h1, ph1]	= sigmrnd(temp + params.U * y1');
        x2          = sigmrnd(params.b' + h1' * params.W);
        py2         = exp(gather(params.d' + h1' * params.U));
        y2          = mnrnd(1, py2 / sum(py2));
        ph2         = sigm(params.c + params.W * x2' + params.U * y2');

        params.W	= params.W + beta * (ph1 * x1 - ph2 * x2);
        params.U	= params.U + beta * (ph1 * y1 - ph2 * y2);
        params.b	= params.b + beta * (x1 - x2)';
        params.c	= params.c + beta * (ph1 - ph2);
        params.d	= params.d + beta * (y1 - y2)';
    else
        d           = struct;
        d.train_x   = x1;
        d.train_y   = y1;
        
        params      = opts.train(params, opts, d);
    end
end

end
