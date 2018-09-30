function [theta,beta,tau,gamma,prior,cuv] = FGPP(events,eventsMatrix,T,t0,...
                                    inedges,outedges,prior,params,kernel,...
                                    likelihoodIsRequired,startIteration)
U = params.U;
P = params.P;
K = params.K;
datasetName = params.datasetName;
methodName = params.methodName;
maxNumberOfIterations = params.maxNumberOfIterations;
saveInterval = params.saveInterval;
plottingInIteration = params.plottingInIteration;
isInDebugMode = params.isInDebugMode;
if nargin<11
    startIteration = 0;
end
if nargin<10
    likelihoodIsRequired = 0;
end
if nargin<9
    w=1;
    g = @(x,w) exp(-w*x);
    g_factorized = @(x,w,p) p*exp(-w*x);
    g_log = @(x,w) -w*x;
    G = @(x,w) 1/w*(1-exp(-w*x));
else
    w = kernel.w;
    g = kernel.g;
    g_factorized = kernel.g_factorized;
    G = kernel.G;
    g_log = kernel.g_log;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%       compute B Values          %%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%    B(u)= \sum_{e\in H_u} G(T-t_e) %%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
SUMG_User = zeros(U,U);
for u=1:U
    for p=1:P
        for i=1:length(eventsMatrix{u,p})
            te = eventsMatrix{u,p}(i);
            for v=outedges{u}
                SUMG_User(u,v) = SUMG_User(u,v)+ G(T-te,w);
                if (t0(v) > te)
                    SUMG_User(u,v) = SUMG_User(u,v) - G(t0(v)-te,w);
                end
            end            
        end
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Initialization %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if startIteration ==0
    rng(0);
    gamma = initializeGamma(inedges, outedges, prior,params);
    iteration = 0;
else
    iteration = startIteration;
    load(sprintf('LearnedModels/LearnedModel_%s_%s_K_%d_w_%.1f_iter_%d.mat',methodName,datasetName,params.K,w,startIteration));
end

x = 1:maxNumberOfIterations;
elboList = zeros(maxNumberOfIterations,1);
if startIteration>0
    elboList(startIteration) = -inf;
end
logLikelihoodList = zeros(maxNumberOfIterations,1);
if likelihoodIsRequired
    [expected_theta,  expected_beta, expected_tau] = ...
    estimateParams(gamma, prior, inedges);

    likelihood = logLikelihoodEstimator(events, outedges,...
                        expected_tau,expected_theta,expected_beta, ...
                    SUMG_User, params, ...
                        kernel, T,t0);
    fprintf('loglikelihood is %.2f\n',likelihood);
end

diffElboM = zeros(maxNumberOfIterations,1);
diffElboV = zeros(maxNumberOfIterations,1);
diffElboBeta = zeros(maxNumberOfIterations,1);
diffElboTau = zeros(maxNumberOfIterations,1);
diffElboKsi = zeros(maxNumberOfIterations,1);
diffElboMu = zeros(maxNumberOfIterations,1);
diffElboEta = zeros(maxNumberOfIterations,1);
diffElboS= zeros(maxNumberOfIterations,1);

elbo = -inf;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Main Loop %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
while iteration<maxNumberOfIterations
    iteration = iteration+1;
    fprintf('Iteration %d\n',iteration)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%   S   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    tic
    [cuv, cuk, cpk,sum_entropy_multinomial,sum_sij_gw] = ...
    quadraticTimeComputePhi(eventsMatrix, events, inedges, gamma, prior, w, g_log, params);
    fprintf('1-S Parameters updated.Elapsed Time is %f\n',toc);
    
    % Elbo
    if isInDebugMode
        lastElbo = elbo;
        elbo = ELBO(U,P,K,T,inedges,SUMG_User,...
                sum_sij_gw,sum_entropy_multinomial,...
                prior,gamma,...
                cuk,cpk,cuv,t0);
        diffElboS(iteration) = elbo-lastElbo;
        if diffElboS(iteration)<-1e-3
            fprintf('BUG: S-DiffElbo is negative:%.2f\n',diffElboS(iteration));
        end
        fprintf('%d-S-DiffElbo is:%.2f\n',iteration,diffElboS(iteration));
    end
    % End of Elbo
    %%%%%%%%%%%%%%%%%%%%%%%%%%%   M   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    tic;
    [gamma] = updateM(P,K,cuk,prior,gamma, T,t0);
    fprintf('2-M Parameters Updated.Elapsed Time is %f\n',toc);
    % Elbo
    if isInDebugMode
        lastElbo = elbo;
        elbo = ELBO(U,P,K,T,inedges,SUMG_User,...
                sum_sij_gw,sum_entropy_multinomial,...
                prior,gamma,...
                cuk,cpk,cuv,t0);
        diffElboM(iteration) = elbo-lastElbo;
        if diffElboM(iteration)<-1e-5
            fprintf('BUG: M-DiffElbo is negative:%.2f\n',diffElboM(iteration));
        end
        fprintf('%d-M-DiffElbo is:%.2f\n',iteration,diffElboM(iteration));
    end
    % End of Elbo
    %%%%%%%%%%%%%%%%%%%%%%%%%%%   V   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    tic;
    [gamma] = updateV(P,K,cuk,prior,gamma, T,t0);
    fprintf('3-V Parameters Updated.Elapsed Time is %f\n',toc);
    % Elbo
    if isInDebugMode
        lastElbo = elbo;
        elbo = ELBO(U,P,K,T,inedges,SUMG_User,...
                sum_sij_gw,sum_entropy_multinomial,...
                prior,gamma,...
                cuk,cpk,cuv,t0);
        diffElboV(iteration) = elbo-lastElbo;
        if diffElboV(iteration)<-1e-3
            fprintf('BUG: V-DiffElbo is negative:%.2f\n',diffElboV(iteration));
        end
        fprintf('%d-V-DiffElbo is:%.2f\n',iteration,diffElboV(iteration));
    end
    % End of Elbo
    %%%%%%%%%%%%%%%%%%%%%%%%%%%   Beta   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    tic;
    [gamma] = updateBeta(P,K,cpk,prior,gamma, T,t0,inedges);
    fprintf('4-Beta Parameters Updated.Elapsed Time is %f\n',toc);
    
    % Elbo
    if isInDebugMode
        lastElbo = elbo;
        elbo = ELBO(U,P,K,T,inedges,SUMG_User,...
                sum_sij_gw,sum_entropy_multinomial,...
                prior,gamma,...
                cuk,cpk,cuv,t0);
        diffElboBeta(iteration) = elbo-lastElbo;
        if diffElboBeta(iteration)<-1e-5
            fprintf('BUG: Beta-DiffElbo is negative:%.2f\n',diffElboBeta(iteration));
        end
        fprintf('%d-Beta-DiffElbo is:%.2f\n',iteration,diffElboBeta(iteration));
    end
    % End of Elbo
    %%%%%%%%%%%%%%%%%%%%%%%%%%%   Tau   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    tic
    [gamma] = updateTau(U,inedges,prior,cuv,SUMG_User,gamma);   
    fprintf('5-Tau Parameters Updated.Elapsed Time is %f\n',toc);
    
    % Elbo
    if isInDebugMode
        lastElbo = elbo;
        elbo = ELBO(U,P,K,T,inedges,SUMG_User,...
                sum_sij_gw,sum_entropy_multinomial,...
                prior,gamma,...
                cuk,cpk,cuv,t0);
        diffElboTau(iteration) = elbo-lastElbo;
        if diffElboTau(iteration)<-1e-5
            fprintf('BUG: Tau-DiffElbo is negative:%.2f\n',diffElboTau(iteration));
        end
        fprintf('%d-Tau-DiffElbo is:%.2f\n',iteration,diffElboTau(iteration));
    end
    % End of Elbo
    %%%%%%%%%%%%%%%%%%%%%%%%%%%   Eta   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    tic;
    [gamma] = updateEta(prior,gamma);
    fprintf('6-Eta Parameters Updated.Elapsed Time is %f\n',toc);
    % Elbo
    if isInDebugMode
        lastElbo = elbo;
        elbo = ELBO(U,P,K,T,inedges,SUMG_User,...
                sum_sij_gw,sum_entropy_multinomial,...
                prior,gamma,...
                cuk,cpk,cuv,t0);
        diffElboEta(iteration) = elbo-lastElbo;
        if diffElboEta(iteration)<-1e-5
            fprintf('BUG: Eta-DiffElbo is negative:%.2f\n',diffElboEta(iteration));
        end
        fprintf('%d-Eta-DiffElbo is:%.2f\n',iteration,diffElboEta(iteration));
    end
    % End of Elbo
    %%%%%%%%%%%%%%%%%%%%%%%%%%%   Mu   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    tic;
    [gamma] = updateMu(U,outedges,prior,gamma);
    fprintf('7-Mu Parameters Updated.Elapsed Time is %f\n',toc);
    % Elbo
    if isInDebugMode
        lastElbo = elbo;
        elbo = ELBO(U,P,K,T,inedges,SUMG_User,...
                sum_sij_gw,sum_entropy_multinomial,...
                prior,gamma,...
                cuk,cpk,cuv,t0);
        diffElboMu(iteration) = elbo-lastElbo;
        if diffElboMu(iteration)<-1e-5
            fprintf('BUG: Mu-DiffElbo is negative:%.2f\n',diffElboMu(iteration));
        end
        fprintf('%d-Mu-DiffElbo is:%.2f\n',iteration,diffElboMu(iteration));
    end
    % End of Elbo
    %%%%%%%%%%%%%%%%%%%%%%%%%%%   Ksi   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    tic;
    [gamma] = updateKsi(P,prior,gamma);
    fprintf('8-Ksi Parameters Updated.Elapsed Time is %f\n',toc);
    % Elbo
    if isInDebugMode
        lastElbo = elbo;
        elbo = ELBO(U,P,K,T,inedges,SUMG_User,...
                sum_sij_gw,sum_entropy_multinomial,...
                prior,gamma,...
                cuk,cpk,cuv,t0);
        diffElboKsi(iteration) = elbo-lastElbo;
        if diffElboKsi(iteration)<-1e-5
            fprintf('BUG: Ksi-DiffElbo is negative:%.2f\n',diffElboKsi(iteration));
        end
        fprintf('%d-Ksi-DiffElbo is:%.2f\n',iteration,diffElboKsi(iteration));
    end
    % End of Elbo
%%%%%%%%%%%% Saving Elbo and LogLikelihood %%%%%%%%%%%%%%%%%
    if isInDebugMode==0
        elboList(iteration) = ELBO(U,P,K,T,inedges,SUMG_User,...
                sum_sij_gw,sum_entropy_multinomial,...
                prior,gamma,...
                cuk,cpk,cuv,t0);
        fprintf('Elbo Computed. Elapsed Time is %f\n',toc);
        tic;
        % LogLikelihood
        if likelihoodIsRequired
            [theta, beta, tau] = estimateParams(gamma, prior, inedges);
            newLogLikelihood = logLikelihoodEstimatorTemporalFeatures(inedges,...
                                eventsMatrix,tau,theta,beta, SUMG_User, params, ...
                                w, g, g_factorized, T,t0);
            fprintf('LogLikelihood Computed.Elapsed Time is %f\n',toc);
            logLikelihoodList(iteration) = newLogLikelihood;
        end

        if iteration>1 && elboList(iteration)-elboList(iteration-1)<-1e-5
            fprintf('Learning finished. DeltaElbo=%f. \n',elboList(iteration)-elboList(iteration-1));
            break;
        end
        if mod(iteration,saveInterval) ==0
           [theta, beta, tau] = estimateParams(gamma, prior, inedges);
           save(sprintf('LearnedModels/LearnedModel_%s_%s_K_%d_w_%.1f_iter_%d.mat',methodName,datasetName,params.K,w,iteration),...
            'theta','beta','tau','kernel','params','prior','gamma','cuv');
        end
    end
    
%%  Plotting Results
    if (plottingInIteration == 1)
        subplot(2,1,1);
        plot(x(1:plotDataCount),elboList(1:plotDataCount));
        subplot(2,1,2);
        plot(x(1:plotDataCount),logLikelihoodList(1:plotDataCount));
        drawnow
    end
end
[theta, beta, tau] = estimateParams(gamma, prior, inedges);
save(sprintf('LearnedModels/LearnedModel_%s_%s_K_%d_w_%.1f_iter_%d.mat',methodName,datasetName,params.K,w,maxNumberOfIterations),...
    'theta','beta','tau','kernel','params','prior','gamma','cuv');
end


%% 2.2 m
function [gamma] = updateM(P,K,cuk,prior,gamma, T,t0)
    gamma.m_shp = prior.shape.m+sum(cuk(:,1:K),2)+cuk(:,end);
    sum_expected_beta_p = (T-t0) * squeeze(sum(gamma.beta_shp./gamma.beta_rte)); % T-t0(u) * sum_p E[beta_p]
    v_beta_one_to_K = gamma.v.*sum_expected_beta_p;
    for k = 1:K-1
        v_beta_one_to_K (:,k+1:K) = v_beta_one_to_K (:,k+1:K).* repmat(1-gamma.v(:,k),1,K-k);
    end
    v_beta_after_K =  P * prior.shape.beta/prior.rate.beta* (T-t0).* prod(1-gamma.v,2);
    gamma.m_rte = sum(v_beta_one_to_K,2) + sum(v_beta_after_K,2) + gamma.eta_shp./gamma.eta_rte;
end
%% 2.2 v
function [gamma] = updateV(P,K,cuk,prior,gamma, T,t0)
    U = size(cuk,1);
%     A = cuk(:,1:K);
%     B = zeros(U,K);
%     C = zeros(U,K);
    expected_m = gamma.m_shp./gamma.m_rte;
    sum_expected_beta = sum(gamma.beta_shp./gamma.beta_rte);
    expected_beta_prior = prior.shape.beta/prior.rate.beta;
%     B = 1-prior.alpha-repmat(cuk(:,end),1,K);
%     for k = 1:K-1
%         B(:,k) = B(:,k)-sum(cuk(:,k+1:K),2);
%     end
    for u = 1:U
        for k= 1:K
            a = cuk(u,k);
            b = 1-prior.alpha-cuk(u,end);
            if k<K
                b = b  - sum(cuk(u,k+1:K));
            end
            c = - (T-t0(u))*expected_m(u);
            if k>1
                c = c*prod(1-gamma.v(u,1:k-1));
            end
            sum_to_K = 0;
            if k < K
                temp = gamma.v(u,k+1:K).*sum_expected_beta(k+1:K);
                coeff = 1-gamma.v(u,k+1);
                for l = k+2:K
                    temp(l-k) = temp(l-k)*coeff;
                    coeff = coeff * (1-gamma.v(u,l));
                end
                sum_to_K = sum(temp);
            end
            sum_after_K = P * expected_beta_prior;
            if k<K
               sum_after_K = sum_after_K * prod(1-gamma.v(u,k+1:K));
            end
            c = c * (sum_expected_beta(k) - sum_to_K - sum_after_K);
            
            
            if abs(a) <1e-6 && abs(b)<1e-6
                fprintf('Error: Number of choices is infinite. a and b are both zero.\n');
            elseif abs(a) > 1e-6 && abs(b)<1e-6
                choice1 = -a/c;
                if choice1>0 && choice1<=1
                    gamma.v(u,k) = choice1;
                elseif a<0
                        gamma.v(u,k) = 0.001;
                    elseif c>0
                        gamma.v(u,k) = 0.999;
                else
                   fprintf('Error: b is zero and a =%.3f the only choice is %.3f.\n',choice1); 
                end
            elseif abs(a) < 1e-6 && abs(b)>1e-6
                choice1 = 1+b/c;
                if choice1>=0 && choice1<=1
                    gamma.v(u,k) = choice1;
                elseif b>0
                        gamma.v(u,k) = 0.999;
                    else
                        gamma.v(u,k) = 0.001;
                end
            elseif abs(a) > 1e-6 && abs(b)>1e-6
                delta = (c+b-a)^2+4*a*c;
                if delta<0
                    if b>0
                        gamma.v(u,k) = 0.9999;
                    else
                        fprintf('Error: Delta<0 and b>0. u=%d, k=%d, a=%f, b=%f, c=%f\n',u,k,a,b,c);
                    end
                else
                    choice1 = ((a-b-c)+sqrt(delta))/(-2*c);
                    choice2 = ((a-b-c)-sqrt(delta))/(-2*c);
                    cnt = 0;
                    if choice1 >0 && choice1<1
                        gamma.v(u,k) = choice1;
                        cnt = cnt+1;
                    end
                    if choice2 >0 && choice2<1
                        gamma.v(u,k) = choice2;
                        cnt = cnt+1;
                    end
                    if cnt ~=1
                        if b>0
                            gamma.v(u,k) = 0.9999;
                        else
                            fprintf('Error: Delta>0 and b<0. u=%d, k=%d, a=%f, b=%f, c=%f\n',u,k,a,b,c);
%                         fprintf('Error: Number of choices is %d. u=%d, k=%d, a=%f, b=%f, c=%f, v1=%f and v2=%f\n',cnt,u,k,a,b,c,choice1,choice2);
                        end
                    end
                end
            end
        end
    end
%     C = -1*repmat((T-t0).*(expected_m),1,K);
%     coeff = 1-gamma.v(:,1);
%     for k = 2:K
%         C(:,k)= C(:,k).*coeff;
%         coeff = coeff.* (1-gamma.v(:,k));
%     end
%     temp = repmat(sum_expected_beta,U,1);
%     C = C*(sum_expected_beta(k)-);
    
    
end
%% 2.3 beta
function [gamma] = updateBeta(P,K,cpk,prior,gamma, T,t0,inedges)
    [expected_theta,~,~,~,~,~,~,~] = estimateParams(gamma, prior, inedges);
    expected_theta = expected_theta(:,1:end-1);
    sum_expected_theta_u = (T-t0)'*expected_theta; %(1*U) * (U*K) = 1*K
    gamma.beta_shp = prior.shape.beta+cpk(:,1:K);
    gamma.beta_rte = repmat(sum_expected_theta_u,[P,1])+repmat(gamma.ksi_shp./gamma.ksi_rte,[1,K]);
end
%% 2.1 tau
function [gamma] = updateTau(U,inedges,prior,cuv,SUMG_User,gamma)
    for v=1:U        
        for u=inedges{v} %u influences on v
            gamma.tau_shp(u,v) = prior.shape.tau+cuv(u,v); %c(u,v)
            gamma.tau_rte(u,v) = SUMG_User(u,v) + gamma.mu_shp(u)/gamma.mu_rte(u);
        end
    end
end
%% 2.5 eta
function [gamma] = updateEta(prior,gamma)
    gamma.eta_rte = prior.rate.eta + gamma.m_shp ./ gamma.m_rte;
end
%% 2.6 mu
function [gamma] = updateMu(U,outedges,prior,gamma)
    for u=1:U
        user_expected_tau = sum(gamma.tau_shp(u,outedges{u})./gamma.tau_rte(u,outedges{u}));
        gamma.mu_rte(u) = prior.rate.mu + user_expected_tau;
    end
end
%% 2.7 ksi
function [gamma] = updateKsi(P,prior,gamma)
    for p=1:P
        product_expected_beta = sum( gamma.beta_shp(p,:) ./ gamma.beta_rte(p,:));

        gamma.ksi_rte(p) = prior.rate.ksi + product_expected_beta;
    end
end
