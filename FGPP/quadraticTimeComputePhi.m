%%% quadraticTimeComputePhi
function [cuv, cuk, cpk, sum_entropy_multinomial,sum_sij_gw] = ...
    quadraticTimeComputePhi(eventsMatrix, events, inedges, gamma, prior, w, g_log, params)
    U = params.U;
    P = params.P;
    K = params.K;
    alpha = prior.alpha;
    
    cuv = zeros(U,U);
    %cuk(K+2) = sum_{e\in H_u}sum_{k=K+1}^{\infty}E[s_e(k)]
    cuk = zeros(U,K+2);
    %cpk(K+2) = sum_{e\in H_p}sum_{k=K+1}^{\infty}E[s_e(k)]
    cpk = zeros(P,K+2);
    
    sum_entropy_multinomial = 0.0;
    sum_sij_gw = 0.0;
    [~,~,~,~,expected_log_theta,expected_log_beta,expected_log_tau,~] = ...
        estimateParams(gamma, prior, inedges);
%     psi_gamma.theta_shp = psi(gamma.theta_shp);
%     psi_gamma.beta_shp = psi(gamma.beta_shp);
%     psi_gamma.tau_shp = psi(gamma.tau_shp);
%     log_gamma.theta_rte=log(gamma.theta_rte);
%     log_gamma.beta_rte = log(gamma.beta_rte);
%     log_gamma.tau_rte = log(gamma.tau_rte);
%     expected_log_theta = computeExpectedlogTheta(gamma,alpha);
%     expected_log_beta = computeExpectedlogBeta(gamma,prior);
%     expected_log_tau = psi(gamma.tau_shp) - log(gamma.tau_rte);
    
    for n=1:length(events)
        tn = events{n}.time;
        un = events{n}.user;
        pn = events{n}.product;
        
        [valid_events_users,valid_events_times] = ...
            computeValidEvents(tn,un,pn,inedges,eventsMatrix);
        
        
        [log_phi_negative , log_phi_positive] = computeLogPhi(un,tn,pn,...
            expected_log_theta,expected_log_beta,expected_log_tau,...
            valid_events_users,valid_events_times,g_log,w);
        
        [phi_negative , phi_positive, sum_phi_after_K] = computePhi(log_phi_negative,log_phi_positive,alpha);
        [sum_entropy_multinomial,sum_sij_gw] = updateEntropyAndSG(...
            sum_entropy_multinomial,sum_sij_gw,...
            phi_negative,phi_positive,...
            alpha, w, g_log, tn, valid_events_times);
        [cuv, cuk, cpk] = updateCounts(un,pn,cuv,cuk,cpk,...
            valid_events_users,...
            phi_negative,phi_positive,sum_phi_after_K);
    end
end

%%% compute Phi
function [phi_negative , phi_positive, sum_phi_after_K] = computePhi(log_phi_negative,log_phi_positive,alpha)
    max_log_phi = max(log_phi_negative);
    if (~isempty(log_phi_positive))
        max_log_phi = max(max_log_phi,max(log_phi_positive));
    end
    
    phi_negative = exp(log_phi_negative-max_log_phi);
    phi_positive = exp(log_phi_positive-max_log_phi);
    
    
    sum_phi_K_plus_one_to_inf = exp(log_phi_negative(end)-max_log_phi)/(1-exp(psi(alpha)-psi(1+alpha)));
    
    sum_phi = sum(phi_negative(1:end-1))+sum(phi_positive)+sum_phi_K_plus_one_to_inf;
    
    phi_negative = phi_negative/sum_phi;
    phi_positive = phi_positive/sum_phi;
    sum_phi_after_K = sum_phi_K_plus_one_to_inf/sum_phi;
end

%%% compute Log Phi
function [log_phi_negative , log_phi_positive] = computeLogPhi(un,tn,pn,...
        expected_log_theta,expected_log_beta, expected_log_tau,...
        valid_events_users,valid_events_times,g_log,w)
    log_phi_negative = expected_log_theta(un,:) + expected_log_beta(pn,:);
    log_phi_positive = (expected_log_tau(valid_events_users) + g_log(tn-valid_events_times,w));
end


%Valid events are events that can influence on event(tn,un,pn)
%So each event (tm,um,pm) such that um can influence on un,
% & tm < tn & pm = pn is a valid event.
function [valid_events_users,valid_events_times] = ...
    computeValidEvents(tn,un,pn,inedges,eventsMatrix)
    
num_events = 0.0;
for um=inedges{un}
    num_events = num_events+length(eventsMatrix{um,pn});
end
valid_events_length = 0;
valid_events_users = zeros(1,num_events);
valid_events_times = zeros(1,num_events);
for um=inedges{un}
    for tm=eventsMatrix{um,pn}
        if (tm >= tn)
            break;
        end
        valid_events_length = valid_events_length+1;
        valid_events_users(valid_events_length) = um;
        valid_events_times(valid_events_length) = tm;
    end
end

valid_events_users = valid_events_users(1:valid_events_length);
valid_events_times = valid_events_times(1:valid_events_length);

end

%%% updateCounts
function [cuv,cuk,cpk] = ...
    updateCounts(un,pn,cuv,cuk,cpk,...
            valid_events_users,...
            phi_negative,phi_positive,sum_phi_after_K)
    cuk(un,1:end-1) = cuk(un,1:end-1) + phi_negative;
    cuk(un,end) = cuk(un,end) + sum_phi_after_K;
    cpk(pn,1:end-1) = cpk(pn,1:end-1) + phi_negative;
    cpk(pn,end) = cpk(pn,end) + sum_phi_after_K;
    cuv(valid_events_users,un) = cuv(valid_events_users,un)+phi_positive';
end

function [sum_entropy_multinomial,sum_sij_gw] = updateEntropyAndSG(...
            sum_entropy_multinomial,sum_sij_gw,...
            phi_negative,phi_positive,...
            alpha, w, g_log,tn,valid_events_times)
        neg_indices = phi_negative >1e-40;
        entropy_phi_neg = -1*sum(phi_negative(neg_indices(1:end-1)).*log(phi_negative(neg_indices(1:end-1))));
        if phi_negative(end)>1e-40
            x = phi_negative(end);
            coeff = exp(psi(alpha)-psi(1+alpha));
            entropy_phi_neg = entropy_phi_neg - (x*log(x)/(1-coeff)+...
                x * coeff * log(coeff)/(1-coeff)^2);
        end
        sum_entropy_multinomial = sum_entropy_multinomial + entropy_phi_neg;
        pos_indices = phi_positive >1e-40;
        if ~isempty(phi_positive)
            sum_entropy_multinomial = sum_entropy_multinomial -  sum(phi_positive(pos_indices).*log(phi_positive(pos_indices)));
            sum_sij_gw = sum_sij_gw+g_log(tn-valid_events_times,w)*phi_positive';
        end
end
