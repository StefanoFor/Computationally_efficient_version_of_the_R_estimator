function [alpha_est, Delta_T, W] = alpha_estimator_sub_vdW_mv(y, T, V, m_index)

[N, K] = size(y);

% Evaluation of the approximation of the efficient central sequence Delta_T in Eq. (33)
% and of the matrix Psi_T, where T is the preliminary estimator
[Delta_T, W, inv_T] = Delta_eval_vdW_mv(y, T, m_index);

% Evaluation of the perturbed approximation of the efficient central sequence Delta_T
T_pert = T + V/sqrt(K);
Delta_T_pert = Delta_only_eval_vdW_mv(y, T_pert, m_index);

Z = inv_T*V*inv_T - (trace(inv_T*V)/N)*inv_T;

Z_hf = tril(Z) + tril(Z,-1);
Z_vecs  = Z_hf(m_index);
Z_vecs = Z_vecs(2:end);

% Estimation of alpha 
alpha_est = norm(Delta_T_pert-Delta_T)/norm(Z_vecs);
end

