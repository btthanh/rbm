function params = rbmTrain(params, opts, data)

m          = size(data.train_x, 1);
lambda     = opts.lambda;

kk = randperm(m);
for l = 1 : m
    x1          = data.train_x(kk(l), :);
    y1          = data.train_y(kk(l), :);

    [h1, ph1]	= sigmrnd(params.c + params.W * x1' + params.U * y1');
    x2			= sigmrnd(params.b' + h1' * params.W);
    py2			= exp(gather(params.d' + h1' * params.U));
    y2			= mnrnd(1, py2 / sum(py2));
    ph2			= sigm(params.c + params.W * x2' + params.U * y2');

    params.W	= params.W + lambda * (ph1 * x1 - ph2 * x2);
    params.U	= params.U + lambda * (ph1 * y1 - ph2 * y2);
    params.b	= params.b + lambda * (x1 - x2)';
    params.c	= params.c + lambda * (ph1 - ph2);
    params.d	= params.d + lambda * (y1 - y2)';
end

end
