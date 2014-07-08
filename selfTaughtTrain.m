function params = selfTaughtTrain(params, opts, data)

m          = size(data.train_x, 1);
lambda     = opts.lambda;
beta       = opts.beta * lambda;

kk = randperm(m);
for l = 1 : m
    x1          = data.train_x(kk(l), :);
    y1          = data.train_y(kk(l), :);

    if sum(y1) == 0
        [h1, ph1]	= sigmrnd(params.c + params.W * x1');
        x2          = sigmrnd(params.b' + h1' * params.W);
        ph2         = sigm(params.c + params.W * x2');

        params.W	= params.W + beta * (ph1 * x1 - ph2 * x2);
        params.b	= params.b + beta * (x1 - x2)';
        params.c	= params.c + beta * (ph1 - ph2);
    else
        d           = struct;
        d.train_x   = x1;
        d.train_y   = y1;
        
        params      = opts.train(params, opts, d);
    end
end

end
