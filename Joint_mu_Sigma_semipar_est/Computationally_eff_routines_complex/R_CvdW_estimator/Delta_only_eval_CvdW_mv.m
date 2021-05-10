function Delta_T = Delta_only_eval_CvdW_mv(y, T)

[N, K] = size(y);

% Evaluation of the score function and of the vector u
[score_vect,u,inv_sr_T,inv_T] = score_rank_sign_CvdW_mv(y,T);

%%%% Evaluation of the approximation of the efficient central sequence in Eq. (33)
score_mat = repmat(sqrt(score_vect),[N,1]);
U_appo = score_mat.*u;
Score_appo_m = U_appo*U_appo';

Delta_T = inv_sr_T * Score_appo_m * inv_sr_T - sum(score_vect)*inv_T/N;
Delta_T = Delta_T(2:end).'/sqrt(K);

end
