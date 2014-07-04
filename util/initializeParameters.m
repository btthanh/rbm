function theta = initializeParameters(hiddenSize, visibleSize)

%% Initialize parameters randomly based on layer sizes.
r  = max(hiddenSize, visibleSize)^(-0.5);  % we'll choose weights uniformly from the interval [-r, r]
theta = rand(hiddenSize, visibleSize) * 2 * r - r;

end
