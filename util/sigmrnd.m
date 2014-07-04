function [sample, prob] = sigmrnd(z)
	prob = sigm(z);
    sample = double(prob > rand(size(z)));
end