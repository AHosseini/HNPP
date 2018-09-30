function  [expected_theta,expected_beta,expected_tau,expected_m,...
    expected_log_theta,expected_log_beta,expected_log_tau,expected_log_m] = estimateParams(gamma, prior, inedges)
    alpha = prior.alpha;
    %% Theta
    [U,K] = size(gamma.v);
    expected_m = gamma.m_shp ./ gamma.m_rte;
    expected_log_m =(psi(gamma.m_shp)- log(gamma.m_rte));
    expected_theta = repmat(expected_m,1,K+1);
    expected_theta(:,1:K)= expected_theta(:,1:K).*(gamma.v);
    expected_theta(:,K+1)= expected_theta(:,K+1)*(1/(1+alpha));
    for k = 1:K
        expected_theta (:,k+1:end) = expected_theta (:,k+1:end) .* repmat(1-gamma.v(:,k),1,K+1-k);
    end
    expected_log_theta = repmat(expected_log_m,1,K+1);
    expected_log_theta(:,1:K)= expected_log_theta(:,1:K)+log(gamma.v);
    expected_log_theta(:,K+1)= expected_log_theta(:,K+1)+psi(1)-psi(1+alpha);
    for k = 1:K
        expected_log_theta (:,k+1:end) = expected_log_theta (:,k+1:end) + repmat(log(1-gamma.v(:,k)),1,K+1-k);
    end
    %% Beta
    [P,K] = size(gamma.beta_shp);
    expected_beta = zeros(P,K+1);
    expected_beta(:,1:K) = gamma.beta_shp ./ gamma.beta_rte;
    expected_beta(:,K+1) = prior.shape.beta/prior.rate.beta;
    expected_log_beta = zeros(P,K+1);
    expected_log_beta(:,1:K) = (psi(gamma.beta_shp) - log(gamma.beta_rte));
    expected_log_beta(:,K+1)= (psi(prior.shape.beta) - log(prior.rate.beta));
    
    %% Tau
    expected_tau = zeros(U,U);
    expected_log_tau = zeros(U,U);
    for v=1:U
        expected_tau(inedges{v},v) = gamma.tau_shp(inedges{v},v)./gamma.tau_rte(inedges{v},v);
        expected_log_tau(inedges{v},v) = psi(gamma.tau_shp(inedges{v},v))-log(gamma.tau_rte(inedges{v},v));
    end
end