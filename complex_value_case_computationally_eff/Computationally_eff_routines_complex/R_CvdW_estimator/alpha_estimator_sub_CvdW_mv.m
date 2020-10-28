function [alpha_est, Delta_T, W] = alpha_estimator_sub_CvdW_mv(y, T, V)

[N, K] = size(y);

% Evaluation of the approximation of the efficient central sequence Delta_T in Eq. (47)
% and of the matrix Psi_T, where T is the preliminary estimator
[Delta_T, W, inv_T] = Delta_eval_CvdW_mv(y, T);

% Evaluation of the perturbed approximation of the efficient central sequence Delta_T
T_pert = T + V/sqrt(K);
Delta_T_pert = Delta_only_eval_CvdW_mv(y, T_pert);

% Estimation of alpha (see Eq. (53))
Zc = inv_T*V*inv_T - (trace(inv_T*V)/N)*inv_T;
Zc = Zc(2:end);

% Estimation of alpha 
alpha_est = norm(Delta_T_pert-Delta_T)/norm(Zc);

end

