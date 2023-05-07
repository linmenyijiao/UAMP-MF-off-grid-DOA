 % Function of Estimating the sparse channel by SBL-DP algorithm
function [res] = UAMP_MF_DP_offgrid(Y,A ,B,config)
%% function of SBL_DP2 algorithm
%原GAMP的更新顺序，离网模型  
%% input: Y recieved signals
%          A orthorgnal measurement matrix
%          B the derived matrix of A  
%% get the dimension of the system
aziGrid=config.aziGrid;
DOA_K=config.u_num;
DOA_true=config.Eld;
[N,F] = size(Y);
K=F-1;
[~,L,~ ] = size(A );

%% set the iteration condition
Thr=1.0e-6; %Threshold for Difference by continuous twice iteration
Nmax =6*config.items; %max iteration times
count = 1;
FlagIter = 1;
%% off-gird para
A_hat = A;
A_hat1 = zeros(N,L,F);
beta_k = zeros(L,K);
%grid_r = (aziGrid(end)-aziGrid(1))/(length(aziGrid)-1);%网格分辨率
grid_r =config.grid;
grid_num = (aziGrid(end)-aziGrid(1))/grid_r +1;
iter_beta = 10;
%% Initialization the hyper-parameters
a =0;     % update alpha
b = 0;     % update alpha
% c = 3e-3;     % update alpha_k
% d = 1e-4;     % update alpha_k
c = 1;     % update alpha_k
d = 1e-10;     % update alpha_k
e = 1;     % update gamma
h = 0;     % update gamma

c_update = zeros(L,K);
d_update = zeros(L,K);
gama =50;
rng(0)
%% fixed parameter value
 a_update = a  + N*F;  % first parameter of gamma ditribution    followed
 e_update = e +K-1;  % first parameter of gamma ditribution gama followed

%% Initialization the precision vector of Y
alpha0 = 1;  % precision of noise

%% Initialization the precision vector of K clusters
arpha_K_star =rand(L,K); % precision vector of K different clusters
log_arpha_K_star = log(arpha_K_star);
%% Initialization the assignment vector of K clusters
%phi_fk = rand(K,F)./sum(rand(K,F),1);
phi_fk = ones(K,F)./K;
zeta_fk = zeros(K,F);
%% initrial the percentage of each piece of z_f by uniform distribution
pai = rand(K,2) ;
log_pai = log10(pai);    % log probability of pai
%% Initial value on gamma
% gama=0.5;
%% Initialization the mean and covariance 
x_l = zeros(L,F); v_x_l= ones(L, F);     % Define variable x (belief) E and Var
s_n =  zeros(N,F)  ;                        % Initial s_n N*F  
p_n = zeros(N,F);  v_p_n= ones(N, F);  % Define  l2r message 3  phi E and Var  
q_l = zeros(L,F);  v_q_l =  ones(L,F);% Define  r2l message 7  x  E and Var   L*F
h_n = Y;  v_h_n = ones(N,F)./alpha0; % Define message 5 of h  E and Var N*N*F
% h_n   = zeros(N,1); v_h_n = ones(N,1);    
%% UAMP
U = zeros(N,N,F);
V = zeros(L,L,F);
S = zeros(N,L,F);
%%%%%%%%%%Finish the initialization%%%%%%%%%%%%%%%%
%% main loop
tic;
while FlagIter
   x_l0 = x_l;
   %%  update GAMP 3 l2r message p3_hat E and p3_var
   for f = 1:F
       v_p_n(:, f) = (abs(A_hat(:,:,f)).^2)*v_x_l(:, f)  ;
       p_n(:,f) = A_hat(:,:,f)*x_l(:,f) - v_p_n(:,f).*s_n(:,f);  
   end
   %% update intermediate variables s_n
   v_s_n = 1./(v_h_n+v_p_n );
   s_n = v_s_n .* (h_n-p_n);
   %% update GAMP right out message q_l and   v_q_l
   for f =1:F
       v_q_l(:,f) =  1./( (abs(A_hat(:,:,f)).^2 )'*  v_s_n(:,f) );
       q_l(:,f) = v_q_l(:,f) .*  (A_hat(:,:,f)'*s_n(:,f))  + x_l(:,f);
   end
    %% update GAMP x belief: x_l and v__x_l
     for f =1:F  
         
        v_x_l(:,f) = 1./(arpha_K_star*phi_fk(:,f)  + 1./v_q_l(:,f))  ; 
          
     end
     x_l =  v_x_l .* q_l ./ v_q_l ;
   %% update z belief: Zhat and Zhatvar
   Zhatvar = 1./(1./v_p_n + 1./v_h_n);
   Zhat  = (p_n./v_p_n + Y./v_h_n).*Zhatvar ;
   
%      %% update xf
%     for f = 1:F
%         gam_inter = 0;
%         for k = 1:K
%             gam_inter = gam_inter + phi_fk(k,f) * diag(arpha_K_star(:,k));
%         end
%         gam_f(:,:,f) = gam_inter + alpha0 * A(:,:,f)'*A(:,:,f);
%         x_l(:,f) = alpha0 * inv(gam_f(:,:,f)) * A(:,:,f)'* Y(:,f);
%     end
   %% update arpha_K_star
    sum1 = sum(phi_fk,2);
    for k = 1:K
        for i = 1:L
            sum2 = 0;
            for f = 1:F
                sum2 = sum2 + phi_fk(k,f) * (abs(x_l(i,f)^2) + abs(v_x_l(i,f)));
            end
            c_update(i,k)  = c + sum1(k,1);
            d_update(i,k)  = d + sum2;
            arpha_K_star(i,k) = c_update(i,k)/d_update(i,k);
            log_arpha_K_star(i,k) = psi(c_update(i,k))-log(d_update(i,k));
        end
    end
    %% update phi
    %     %% update phi off-grid 2
    %[q_r_k,n_bp_zr] = update_Z_BP_MF2(c_p,d_p,pai,mean_H,precision_H);
    phi_fk=zeros(K,F);
    theta=zeros(K,F); 
    for f = 1:F
        for k = 1:K
            theta(k,f) =  log_pai(k,1)+sum(log_pai(1:k-1,2))  +  ... 
           sum(log_arpha_K_star(:,k) -abs( trace(diag(arpha_K_star(:,k))* diag(v_x_l(:,f)) ))  -abs( x_l(:,f)'*diag(arpha_K_star(:,k))*x_l(:,f)) ) ;
        end
        theta(:,f) = theta(:,f)-max(theta(:,f));
        phi_fk(:,f) = exp(theta(:,f)) ./ sum( exp(theta(:,f)),1);
    end

%     for f = 1:F
%         for k = 1:K
%             zeta_fk(k,f) = log_pai(k,1)+sum(log_pai(1:k-1,2)) + sum(log_arpha_K_star(:,k))...
%                 - sum( arpha_K_star(:,k).* (x_l(:,f).*conj(x_l(:,f)) + abs(v_x_l(:,f)))  )  ;
%         end
%         zeta_fk(:,f) = zeta_fk(:,f)-max(zeta_fk(:,f));
%         phi_fk(:,f) = exp(zeta_fk(:,f)) ./ sum( exp(zeta_fk(:,f)),1);
%     end
   %% update pai
   tao(:,1) = sum(phi_fk,2)+1;
    for k=1:K
        tao(k,2)=sum(sum(phi_fk(k+1:K,:),1),2)+ gama;
    end 
    log_pai(:,1) = psi(tao(:,1) ) -psi( tao(:,1) + tao(:,2) );
    log_pai(:,2) = psi(tao(:,2)) -psi(tao(:,1) + tao(:,2));
    
    %% update gama
    h_update = h-sum(log_pai(1:K-1,2));
    gama = e_update/h_update;
     %% %% update Noise Accuracy  alpha0 (belief)  
    b_inter = 0;
%     for af = 1:F
%     %此处的方差项使用x的表达式可以得到结果
% %         b_inter = b_inter + abs((Y(:,af)-Zhat(:,af))'* (Y(:,af)-Zhat(:,af)))...
% %             + trace(abs(A_hat(:,:,af)'*A_hat(:,:,af))*abs(diag(v_x_l(:,af)))) ;
%         b_inter = b_inter + abs((Y(:,af)-Zhat(:,af))'* (Y(:,af)-Zhat(:,af)))   + sum(abs(Zhatvar(:,f)));
%           
%     end
 xpower=zeros(grid_num,F) ;
     for f = 1:F
        for k = 1:K
            b_inter = b_inter + phi_fk(k,f)*(abs(norm(Y(:,f)-(A(:,:,f)+B(:,:,f)*diag(beta_k(:,k)))*x_l(:,f))^2) ...
                + trace(abs((A(:,:,f)+B(:,:,f)*diag(beta_k(:,k)))'*(A(:,:,f)+B(:,:,f)*diag(beta_k(:,k))))*abs(diag(v_x_l(:,f)))));
        end
     end
    b_update = b + b_inter;
    
    alpha0 = a_update/b_update;
     v_h_n = ones(N,F)./alpha0 ;
   
     %% update beta_k
     % update beta_k
 
%     [~,idnx] = sort(sum(alpha_k,2),"ascend");
%     [~,idnx_x]= findpeaks(sum(abs(x_l),2),'SortStr',"descend");
%     idx_x=idnx_x(1:DOA_K);

    [~,idnx] = sort(sum(arpha_K_star,2),"ascend");
    idx=idnx(1:DOA_K);


    for k=1:K
        BB=0;
        ww=0;
        for f = 1:F
            BB = BB +phi_fk(k,f)*real( ( conj(B(:,idx,f)'*B(:,idx,f)) .* (x_l(idx,f)*x_l(idx,f)'+ v_x_l(idx,f))  )  );  %多乘了F
            ww = ww+ phi_fk(k,f)*(real(conj(x_l(idx,f)).*(B(:,idx,f)'*(Y(:,f)-A(:,:,f)*x_l(:,f))) )- real(diag(B(:,idx,f)'*A(:,:,f)*diag(v_x_l(:,f)))) )  ;
        end
        if ww==0
            temp1 =0;
        else
            temp1 = ww./diag( BB);
        end
        
        % temp1 =BB\ww;
        if any(abs(temp1) > grid_r/2) || any(diag(BB) == 0)
            for i = 1:iter_beta
                for n = 1:DOA_K
                    temp_beta = beta_k(idx,k);
                    temp_beta(n) = 0;
                    % temp_BB =  BB(n,:);
                    beta_k(idx(n),k) = (ww(n) - BB(n,:)* temp_beta) / BB(n,n);
                    if beta_k(idx(n),k) > grid_r/2
                        beta_k(idx(n),k) = grid_r/2;
                    end
                    if beta_k(idx(n),k) < -grid_r/2
                        beta_k(idx(n),k) = -grid_r/2;
                    end
                    if BB(n,n) == 0
                        beta_k(idx(n),k) = 0;
                    end
                end
            end
        else
            beta_k = zeros(L,K);
            beta_k(idx,k) = temp1;
        end
        %根据beta 的类别得到离网角度
        doa_off =zeros(L,F);
        for f=1:F
            for k=1:K
              doa_off(:,f) = doa_off(:,f) +phi_fk(k,f).*beta_k(:,k);
            end
        end
    end
    if numel(find(isnan(beta_k)))>1
        count
    end
    %beta_k(isnan(beta_k)) = 0;
    %% update A_hat
    for f =1:F
        A_inter  = 0;
        for k =1:K
%            A_inter  = A_inter + phi_fk(k,f)*(A(:,idx,f)+ B( :,idx,f)* diag(beta_k(idx,k)));
            A_inter  = A_inter + phi_fk(k,f)*(A(:,:,f)+ B( :,:,f)* diag(beta_k(:,k)));
        end
%         A_hat1(:,:,f)  =A(:,:,f) ;
%         A_hat1(:,idx,f) = A_inter;
         A_hat1(:,:,f) = A_inter;
    end
 
%    A_hat1(isnan(A_hat1)) = 0;
%    A_hat1(isinf(A_hat1)) = 0;
    %UAMP
%     if  isnan(A_hat1) ==1 || isinf(A_hat1)==1
%         count
%     end
    for f=1:F
      [U(:,:,f),S(:,:,f),V(:,:,f)] = svd(A_hat1(:,:,f));
      A_hat (:,:,f) = U(:,:,f)'*A_hat1(:,:,f);
      h_n(:,f) = U(:,:,f)'*Y(:,f);
    end
    
    
    %% loop end
    %% 判断是否结束循环
    xerr = mean(mean(abs(x_l-x_l0)./abs(x_l0)));
    if (xerr < Thr)||(count >= Nmax)
        FlagIter = 0;
        iter_beta = 10;
    end
    %% Off-grid 计算
   % xpower_rec = mean(abs(x_l).^2,2) + real(diag(Sigma));
    doa_hat = sort(aziGrid(idx)' + mean(doa_off(idx,:),2),"descend");
   % est_doa = aziGrid'+sum(beta_k,2);
    rmse = sqrt(mean((doa_hat- sort(DOA_true, "descend")).^2));
    fprintf("UAMP: xerr %d  DOArmse %d  循环次数 %d  \n",[xerr,rmse,count]);
    % nmse(count)= mean(diag((xdoa-mean_H)'*(xdoa-mean_H))./diag(xdoa'*xdoa));
    
    URMSE(count )  = rmse;
    TMRMSE(count ) = toc ;
    count = count+1;  % 迭代次数叠加
    res.doa_hat=doa_hat;
    res.idx = idx;
    res.arpha_K_star=arpha_K_star;
    res.u=x_l;
    res.URMSE=URMSE;
    res.rmse_end=URMSE(end);
    res.TMRMSE=TMRMSE;
%     res.xpower = mean( abs(x_l).^2  + real(v_x_l ) ,2) ;
    res.grid = aziGrid';
    res.grid(idx) =aziGrid(idx)' + mean(doa_off(idx,:),2);
    for f=1:F
         xpower(:,f) =  abs( x_l(:,f)).^2 + real( v_x_l(:,f));
    end 
   res.xpower=mean(xpower,2);
end
 
end


