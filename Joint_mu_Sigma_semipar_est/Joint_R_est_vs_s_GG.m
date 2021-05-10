clear all
close all
clc


Ns=10^6; % monte carlo trials
Max_it = 250;
N = 8;
perturbation_par = 0.01;
nu_par = 5;

ro=0.8*exp(1j*2*pi/5);
sigma2 = 4;

svect=0.1:0.1:2;

Nl=length(svect);

K = 5*N;
n=[0:N-1];

rx=ro.^n; % Autocorrelation function
Sigma = toeplitz(rx);
mu_t = 0.5*exp(1j*2*pi/7*[0:N-1]');

L=chol(Sigma);
L=L';

Shape_S = Sigma/Sigma(1,1);
%Shape_S = N*Sigma/trace(Sigma);
Inv_Shape_S = inv(Shape_S);
theta_true = Shape_S(:);

T2 = kron(Inv_Shape_S.',Inv_Shape_S);
sr_T2 = sqrtm(T2);
I_N = eye(N);
J_n_per = eye(N^2) - I_N(:)*I_N(:).'/N;
I_N2 = eye(N^2);
P = I_N2(:,2:end).';
K_V = P*(sr_T2*J_n_per);

% J_phi = [1; zeros(N^2-1,1)];
% I_N = eye(N);
% %J_phi = I_N(:);
% U = null(J_phi');

DIM = N^2;

tic
for il=1:Nl
    
    s = svect(il)
    b = (sigma2 * N * gamma( N/s )/gamma( (N+1)/s ) )^s;
    
    MSE_SM = zeros(2*N,2*N);
    MSE_mu_Ty = zeros(2*N,2*N);
    
    bias_SM = zeros(N,1);
    bias_mu_Ty = zeros(N,1);
    
    MSE_SCM = zeros(DIM,DIM);
    MSE_NR_Ty = zeros(DIM,DIM);
    MSE_NR_vdW = zeros(DIM,DIM);
    MSE_NR_t = zeros(DIM,DIM);
    
    bias_SCM = zeros(DIM,1);
    bias_NR_Ty = zeros(DIM,1);
    bias_NR_vdW = zeros(DIM,1);
    bias_NR_t = zeros(DIM,1);
    
    parfor ins=1:Ns
        
        w = (randn(N,K)+1j.*randn(N,K))/sqrt(2);
        w_norm = sqrt(dot(w,w));
        w_n = w./repmat(w_norm,N,1);
        R = gamrnd(N/s,b,1,K);
        x = L*w_n;
        y = mu_t + sqrt(repmat(R,N,1).^(1/s)).*x;
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % SCM
        SM = mean(y,2);
        SCM = (y-SM)*(y-SM)'/K;
        %Scatter_SCM = N*SCM/trace(SCM);
        Scatter_SCM = SCM/SCM(1,1);
        
        err_vect = [SM - mu_t;conj(SM - mu_t)];
        bias_SM = bias_SM + (SM - mu_t)/Ns;
        err_SM = err_vect*err_vect';
        MSE_SM = MSE_SM + err_SM/Ns;
        
        err_v = Scatter_SCM(:)-theta_true;
        bias_SCM = bias_SCM + err_v/Ns;
        err_MAT = err_v*err_v';
        MSE_SCM = MSE_SCM + err_MAT/Ns;
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % FP matrix estimator
        [R_Ty, mu_Ty] = Tyler_est_joint( y, Max_it);
        
        err_vect = [mu_Ty - mu_t;conj(mu_Ty - mu_t)];
        bias_mu_Ty = bias_mu_Ty + (mu_Ty - mu_t)/Ns;
        err_mu_Ty = err_vect*err_vect';
        MSE_mu_Ty = MSE_mu_Ty + err_mu_Ty/Ns;
        
        NR_Ty = R_Ty;
        %NR_Ty = N*R_Ty/trace(R_Ty);
        err_v = NR_Ty(:)-theta_true;
        bias_NR_Ty = bias_NR_Ty + err_v/Ns;
        err_NR_Ty = err_v*err_v';
        MSE_NR_Ty= MSE_NR_Ty + err_NR_Ty/Ns;

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Rank-based estimators
        
        % van der Waeredn score
        [R_vdW, a_vdW] = R_CvdW_est_mv( (y-mu_Ty), R_Ty, perturbation_par);
        NR_vdW = R_vdW;
        %NR_vdW = N*R_vdW/trace(R_vdW);
        err_v = NR_vdW(:)-theta_true;
        bias_NR_vdW = bias_NR_vdW + err_v/Ns;
        err_NR_vdW = err_v*err_v';
        MSE_NR_vdW = MSE_NR_vdW + err_NR_vdW/Ns;

         % t-dist score
        [R_t, a_t] = R_CF_est_mv( (y-mu_Ty), R_Ty, nu_par ,perturbation_par);
        NR_t = R_t;
        %NR_t = N*R_t/trace(R_t);
        err_v = NR_t(:)-theta_true;
        bias_NR_t = bias_NR_t + err_v/Ns;
        err_NR_t = err_v*err_v';
        MSE_NR_t = MSE_NR_t + err_NR_t/Ns;

       

    end
    Fro_MSE_SM(il) = norm(MSE_SM,'fro');
    Fro_MSE_mu_Ty(il) = norm(MSE_mu_Ty,'fro');
    
    L2_bias_SM(il) = norm(bias_SM,'fro');
    L2_bias_mu_Ty(il) = norm(bias_mu_Ty,'fro');
    
    Fro_MSE_SCM(il) = norm(MSE_SCM,'fro');
    Fro_MSE_NR_Ty(il) = norm(MSE_NR_Ty,'fro');
    Fro_MSE_NR_vdW(il) = norm(MSE_NR_vdW,'fro');
    Fro_MSE_NR_t(il) = norm(MSE_NR_t,'fro');
    
    L2_bias_SCM(il) = norm(bias_SCM);
    L2_bias_NR_Ty(il) = norm(bias_NR_Ty);
    L2_bias_NR_vdW(il) = norm(bias_NR_vdW);
    L2_bias_NR_t(il) = norm(bias_NR_t);
    
    a_inv_mean = s^2*gamma((N+2*s-1)/s)/(N*b^(1/s)*gamma(N/s));
    a1 = (s-1)/(2*(N+1));
    a2 = (N + s)/(N + 1);
    
%     % FIM
%     FIM_Sigma = K * (a1*Inv_Shape_S(:)*Inv_Shape_S(:)' + a2*kron(Inv_Shape_S.',Inv_Shape_S));
%     % Constrained CRB
%     CRB = U*inv(U'*FIM_Sigma*U)*U';
    
%     % Semiparametric Efficient FIM
%     SFIM_Sigma = K * a2*(kron(Inv_Shape_S.',Inv_Shape_S) - (1/N)*Inv_Shape_S(:)*Inv_Shape_S(:)');
%     % Constrained SCRB
%     SCRB = U*inv(U'*SFIM_Sigma*U)*U';
    
%     % Two alternative ways to evaluate the SCRB
      % First option
%     SFIM_Sigma = K * a2 * (K_V*K_V');
%     SCRB_app = inv(SFIM_Sigma);
%     SCRB = zeros(N^2,N^2);
%     SCRB(2:end,2:end) = SCRB_app;
    
    e1 = [1; zeros(N^2-1,1)];
    D = eye(N^2)-Shape_S(:)*e1.';
    SCRB = 1/(K*a2)*( D *kron(Shape_S.',Shape_S) *D' );
    
    SCRB_mean = (1/(a_inv_mean*K)) * [Sigma; zeros(N,N);
                                       zeros(N,N);    conj(Sigma)];
    
    %CR_Bound(il) = norm(CRB,'fro');
    SCR_Bound(il) = norm(SCRB,'fro');
    SCR_Bound_mean(il) = norm(SCRB_mean,'fro');
end

color_matrix(1,:)=[0 0 1]; % Blue
color_matrix(2,:)=[1 0 0]; % Red
color_matrix(3,:)=[0 0.5 0]; % Dark Green
color_matrix(4,:)=[0 0 0]; % Black
color_matrix(5,:)=[0 0.5 1]; % Light Blue
color_matrix(6,:)=[1 0.3 0.6]; % Pink
color_matrix(7,:)=[0 0.9 0]; % Light Green

line_marker{1}='-s';
line_marker{2}='--d';
line_marker{3}=':^';
line_marker{4}='-.p';
line_marker{5}='-o';
line_marker{6}='--h';
line_marker{7}='-.*';

figure(1)
semilogy(svect,L2_bias_SM,line_marker{2},'LineWidth',1,'Color',color_matrix(2,:),'MarkerEdgeColor',color_matrix(2,:),'MarkerFaceColor',color_matrix(2,:),'MarkerSize',8);
hold on
semilogy(svect,L2_bias_mu_Ty,line_marker{3},'LineWidth',1,'Color',color_matrix(3,:),'MarkerEdgeColor',color_matrix(3,:),'MarkerFaceColor',color_matrix(3,:),'MarkerSize',8);
grid on;
xlabel('Shape parameter: s');ylabel('L2 norm');
legend('SM','mu Ty')
title('Bias in L2 norm')

figure(2)
semilogy(svect,SCR_Bound_mean,line_marker{1},'LineWidth',1,'Color',color_matrix(1,:),'MarkerEdgeColor',color_matrix(1,:),'MarkerFaceColor',color_matrix(1,:),'MarkerSize',8);
hold on
semilogy(svect,Fro_MSE_SM,line_marker{2},'LineWidth',1,'Color',color_matrix(2,:),'MarkerEdgeColor',color_matrix(2,:),'MarkerFaceColor',color_matrix(2,:),'MarkerSize',8);
hold on
semilogy(svect,Fro_MSE_mu_Ty,line_marker{3},'LineWidth',1,'Color',color_matrix(3,:),'MarkerEdgeColor',color_matrix(3,:),'MarkerFaceColor',color_matrix(3,:),'MarkerSize',8);
grid on;
xlabel('Shape parameter: s');ylabel('Frobenius norm');
legend('SCRB','SM','mu Ty')
title('MSE in Frobenus norm')

figure(3)
semilogy(svect,L2_bias_SCM,line_marker{2},'LineWidth',1,'Color',color_matrix(2,:),'MarkerEdgeColor',color_matrix(2,:),'MarkerFaceColor',color_matrix(2,:),'MarkerSize',8);
hold on
semilogy(svect,L2_bias_NR_Ty,line_marker{3},'LineWidth',1,'Color',color_matrix(3,:),'MarkerEdgeColor',color_matrix(3,:),'MarkerFaceColor',color_matrix(3,:),'MarkerSize',8);
grid on;
semilogy(svect,L2_bias_NR_vdW,line_marker{4},'LineWidth',1,'Color',color_matrix(4,:),'MarkerEdgeColor',color_matrix(4,:),'MarkerFaceColor',color_matrix(4,:),'MarkerSize',8);
grid on;
semilogy(svect,L2_bias_NR_t,line_marker{5},'LineWidth',1,'Color',color_matrix(5,:),'MarkerEdgeColor',color_matrix(5,:),'MarkerFaceColor',color_matrix(5,:),'MarkerSize',8);
%grid on;
%semilogy(lambdavect,Fro_MSE_NR_Wi,line_marker{6},'LineWidth',1,'Color',color_matrix(6,:),'MarkerEdgeColor',color_matrix(6,:),'MarkerFaceColor',color_matrix(6,:),'MarkerSize',8);
%grid on;
%semilogy(lambdavect,Fro_MSE_NR_S,line_marker{7},'LineWidth',1,'Color',color_matrix(7,:),'MarkerEdgeColor',color_matrix(7,:),'MarkerFaceColor',color_matrix(7,:),'MarkerSize',8);
xlabel('Shape parameter: s');ylabel('L2 norm');
legend('CSCM','C-Tyler','vdW R-est','t_\nu R-est')
title('Bias in L2 norm')

figure(4)
semilogy(svect,SCR_Bound,line_marker{1},'LineWidth',1,'Color',color_matrix(1,:),'MarkerEdgeColor',color_matrix(1,:),'MarkerFaceColor',color_matrix(1,:),'MarkerSize',8);
hold on
semilogy(svect,Fro_MSE_SCM,line_marker{2},'LineWidth',1,'Color',color_matrix(2,:),'MarkerEdgeColor',color_matrix(2,:),'MarkerFaceColor',color_matrix(2,:),'MarkerSize',8);
hold on
semilogy(svect,Fro_MSE_NR_Ty,line_marker{3},'LineWidth',1,'Color',color_matrix(3,:),'MarkerEdgeColor',color_matrix(3,:),'MarkerFaceColor',color_matrix(3,:),'MarkerSize',8);
grid on;
semilogy(svect,Fro_MSE_NR_vdW,line_marker{4},'LineWidth',1,'Color',color_matrix(4,:),'MarkerEdgeColor',color_matrix(4,:),'MarkerFaceColor',color_matrix(4,:),'MarkerSize',8);
grid on;
semilogy(svect,Fro_MSE_NR_t,line_marker{5},'LineWidth',1,'Color',color_matrix(5,:),'MarkerEdgeColor',color_matrix(5,:),'MarkerFaceColor',color_matrix(5,:),'MarkerSize',8);
xlabel('Shape parameter: s');ylabel('Frobenius norm');
legend('CSCRB','CSCM','C-Tyler','vdW R-est','t_\nu R-est')
title('MSE in Frobenus norm')



