function [C, mu, iter] = Tyler_est_joint(y, MAX_ITER)

[N K] = size(y);

EPS = 1.0e-4;   % Iteration accuracy

invC0 = eye(N); % inverse of the initial estimate
mu0 = zeros(1,N); 
iter = 1;

z0=y.';
while (iter<MAX_ITER)
    z = z0 - repmat(mu0,K,1);
    s = real(sum(conj(z)*invC0.*z,2));
    % Mean vector
    r_inv = s.^(-1/2);
    mu = sum(z0.*repmat(r_inv,1,N),1)/sum(r_inv);
    % Shape matrix
    w = N./s;
    C = z.'*(conj(z).*repmat(w,1,N))/K;
    C=C/C(1,1);
    d = norm(eye(N)-invC0*C,1);
    if (d<=EPS) break; end
    invC0 = inv(C);
    mu0 = mu;
    iter = iter+1;
end
mu = mu.';

