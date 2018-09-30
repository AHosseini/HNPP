function res = ELBO(U,P,K,T,inedges,SUMG_User,...
            sum_sij_gw,sum_entropy_multinomial,...
            prior,gamma,...
            cuk,cpk,cuv,t0)
    %% initialization
    alpha = prior.alpha;
   
   [expected_theta,expected_beta,expected_tau,expected_m,...
    expected_log_theta,expected_log_beta,expected_log_tau,expected_log_m] = estimateParams(gamma, prior, inedges);
    % Computing E[ksi(p)] and E[log(ksi(p))]
    expected_ksi = gamma.ksi_shp ./ gamma.ksi_rte;
    expected_log_ksi = (psi(gamma.ksi_shp) - log(gamma.ksi_rte));
    % Computing E[eta(u)] and E[log(eta(u))]
    expected_eta  = gamma.eta_shp  ./ gamma.eta_rte;
    expected_log_eta  = (psi(gamma.eta_shp)  - log(gamma.eta_rte));
    % Computing E[mu(u)] and E[log(mu(u))]
    expected_mu  = gamma.mu_shp  ./ gamma.mu_rte;
    expected_log_mu  = (psi(gamma.mu_shp)  - log(gamma.mu_rte));
    
    res = 0.0;
    %% E[log p(m | eta)]
    res = res + prior.shape.m*sum(expected_log_eta)+...
        (prior.shape.m-1)*sum(expected_log_m)-...
        sum(expected_eta.*expected_m);
    %% E[log p(v)]
    res = res + K*U*log(alpha)+(alpha-1)*sum(sum(log(1-gamma.v)));
    %% E[log p(beta | ksi)]
    res = res + K*prior.shape.beta*sum(expected_log_ksi)+...
        (prior.shape.beta-1)*sum(sum(expected_log_beta))-...
        sum(sum(repmat(expected_ksi,1,K).*expected_beta(:,1:K)));
    %% E[log p(tau | mu)]
    for v=1:U
        for u=inedges{v}
            res = res+...
                 prior.shape.tau*expected_log_mu(u)...
                +(prior.shape.tau-1)*expected_log_tau(u,v)...
                -expected_mu(u)*expected_tau(u,v); 
        end
    end
    %% E[log (ksi)]
    for p=1:P
        res = res+...
            (prior.shape.ksi-1)*expected_log_ksi(p)...
            -prior.rate.ksi*expected_ksi(p);
    end
    %% E[log (eta)]
    for u=1:U
        res = res+...
            (prior.shape.eta-1)*expected_log_eta(u)...
            -prior.rate.eta*expected_eta(u);
    end
    %% E[log (mu)]
    for u=1:U
        res = res+...
            (prior.shape.mu-1)*expected_log_mu(u)...
            -prior.rate.mu*expected_mu(u);
    end
    %% E[log p(E,S|theta,beta,tau)]
    %cuk * log theta(u,k)
    res = res + sum(sum(cuk(:,1:K).*expected_log_theta(:,1:K)));
    res = res + sum(sum(cuk(:,K+2).*expected_log_theta(:,K+1)));
    temp = (psi(alpha)-psi(1+alpha));
    coeff = temp * exp(temp)/(1-exp(temp))^2;
    res = res + coeff * sum(cuk(:,K+1));
    %cpk * log beta(p,k)
    res = res + sum(sum(cpk(:,1:K).*expected_log_beta(:,1:K)));
    res = res + sum(sum(cpk(:,K+2).*expected_log_beta(:,K+1)));
    %cuv* log(tau(u,v)
    res = res+sum(sum(cuv.*expected_log_tau));    
    res = res+sum_sij_gw;
    
   %% sigma(theta(u,i)*beta(p,i)*T)
    sumTheta = (T-t0)'*expected_theta; %(1*U) * (U*K+1) = 1*K+1
    sumBeta = sum(expected_beta);
    res = res-sumTheta(1:K)*sumBeta(1:K)'; % sum(Theta(u,i)*Beta(p,j)*T
    expected_v_u_K_plus_one = 1/(1+alpha);
    res = res - sumBeta(K+1)*sumTheta(K+1)/expected_v_u_K_plus_one;
    
    %% sigma (tau(v,u) * G(T-ti))
    for v=1:U
        for u=inedges{v}
            res = res - expected_tau(u,v)*SUMG_User(u,v);
        end
    end
    %% ENTROPY
    res = res+sum(sum(gamma_entropy(gamma.beta_shp ,gamma.beta_rte)));
    res = res+sum(sum(gamma_entropy(gamma.m_shp,gamma.m_rte)));
    for v=1:U
        for u=inedges{v}
            res = res+gamma_entropy(gamma.tau_shp(u,v),gamma.tau_rte(u,v));
        end
    end
    res = res+sum(gamma_entropy(gamma.ksi_shp , gamma.ksi_rte));
    res = res+sum(gamma_entropy(gamma.eta_shp , gamma.eta_rte));
    res = res+sum(gamma_entropy(gamma.mu_shp  , gamma.mu_rte));
    res = res+sum_entropy_multinomial;
end

function entropy = gamma_entropy(shp , rte)
    entropy = shp - log(rte) + gammaln(shp) + (1-shp).*psi(shp);
end