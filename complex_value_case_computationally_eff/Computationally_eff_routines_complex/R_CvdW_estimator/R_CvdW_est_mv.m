function [ N_VDW_mv,  beta_est] = R_CvdW_est_mv( y, T, pert)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input: 
% Complex-valued data matrix: y (The data is assumed to be zero-mean observations. 
                  % If it doesn't, y has to be centered using 
                  % a preliminary estimator of the location parameter)
% Preliminary consistent estimator: T 
% Perturabtion parameter: pert

% Output:
% Semiparametric efficient R-estimator of the shape: N_VCW
% Convergence parameter: beta_est

% The score function used here is the van der Waerden score.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% The preliminary estimator has to satisfy the constraint [T]_{1,1}=1
T = T/T(1,1);

[N, ~] = size(y);

% Definition of the "small perturbation" matrix 
V = pert*(randn(N,N) +1i*randn(N,N));
V = (V+V')/2;
V(1,1)=0;

% Estimation of alpha
[alpha_est, ~, W] = alpha_estimator_sub_CvdW_mv( y, T, V);

beta_est = 1/alpha_est;

% One-step estimatimator of the shape matrix
N_VDW_mv = T + beta_est*( W - W(1,1)*T);


end
