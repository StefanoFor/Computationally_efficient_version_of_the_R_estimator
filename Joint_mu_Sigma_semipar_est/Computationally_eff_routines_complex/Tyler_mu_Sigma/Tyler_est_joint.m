function [C, mu, iter] = Tyler_est_joint(y, MAX_ITER)
% [C, iter] = chubM(z, q);
% Computes the complex Huber's M-estimator of scatter
%         
% INPUT:
% y    : data matrix with N rows (length of the vector) and K columns (number of observations).
% q    : user defined constant. A default choice is q=0.9.
% OUTPUT:
% C    : M(q)-estimator of scatter 
%         
% Esa Ollila, esollila@wooster.hut.fi 

[N K] = size(y);

EPS = 1.0e-4;   % Iteration accuracy

% SCM = y*y'/K;
% Scatter_SCM = N*SCM/trace(SCM);

invC0 = eye(N); %inv(Scatter_SCM); % inverse of the initial estimate
mu0 = zeros(1,N); 
iter = 1;

z0=y.';
while (iter<MAX_ITER)
    z = z0 - repmat(mu0,K,1);
    s = real(sum(conj(z)*invC0.*z,2));
    % Mean vector
    r_inv = s.^(-1/2);
    mu = sum(z0.*repmat(r_inv,1,N),1)/sum(r_inv);
    % Scape matrix
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

