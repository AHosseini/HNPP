function [ranks, ndcgOverTime, ndcgAtK, recallAtKOverTime, recallAtK,...
     EstimatedReturningTime, RealReturningTime,diff] = ...
FGPPEvaluator(trainEvents,testEvents, outedges,inedges,eventsMatrix,...
    theta, beta, tau, g, params, RecListSize)
%%%%%%%%%%%%%%%%%%% Initialization %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
U = params.U;
P = params.P;
alpha = params.alpha;

RecommendationListSize = RecListSize;
socialIntensity = zeros(U,P);
NTest = length(testEvents);
NTrain = length(trainEvents);

ranks = zeros(NTest,1);
ndcgOverTime = zeros(NTest,1);
ndcgAtK = zeros(P,1);
EstimatedReturningTime = zeros(U,1);

recallAtKOverTime = zeros(NTest,1);
recallAtK = zeros(P,1);

%% 
%%%%%%%%%%%%%%%%%%%%% Computing ReturningTime Of Events %%%%%%%%%%%%%
isInTrain = zeros(U,1);
lastOccurenceInTrain = zeros(U,1);
for i = 1:NTrain
    event = trainEvents{i};
    ti = event.time;
    ui = event.user;
    pi = event.product;
    isInTrain(ui) = isInTrain(ui)+1;
    lastOccurenceInTrain(ui) = max(lastOccurenceInTrain(ui),ti);
end
isInTest = zeros(U,1);
firstOccurenceInTest = inf(U,1);
for i = 1:NTest
    event = testEvents{i};
    ti = event.time;
    ui = event.user;
    pi = event.product;
    isInTest(ui) = isInTest(ui)+1;
    firstOccurenceInTest(ui) = min(firstOccurenceInTest(ui),ti);
end
RealReturningTime = firstOccurenceInTest-lastOccurenceInTrain;


NumGeneratingPoints = 100;
diff = [];
for u=1:U
    if (isInTest(u) && isInTrain(u))
         t0 = lastOccurenceInTrain(u);
         socialIntensityInitial = computeSocialIntensityUser(u,t0,...
             P,inedges,eventsMatrix,tau,g);
         baseIntensity = theta(u,1:end-1)*sum(beta(:,1:end-1))'+sum(beta(:,end))*theta(u,end)*(1+alpha);
         generatedPoints = zeros(NumGeneratingPoints,1);
        for i=1:NumGeneratingPoints
            generatedPoints(i) = generatePoint(t0,baseIntensity,socialIntensityInitial,g);
        end
        EstimatedReturningTime(u) = median(generatedPoints)-t0;
        diff(end+1) = abs(EstimatedReturningTime(u)-RealReturningTime(u));
    end
    if (mod(u,100) == 0)
        fprintf('u = %d , mse diff = %f\n',u,sqrt(sum(mean(diff.^2))) );
    end
end
%% %%%%%%%%%%%%%%%%%%% Computing SocialIntensity after 
for i = 1:NTrain
    event = trainEvents{i};
    ti = event.time;
    ui = event.user;
    pi = event.product;
    if i > 1
        lastEventTime = trainEvents{i-1}.time;
        socialIntensity = socialIntensity * g(ti-lastEventTime);
    end
    socialIntensity(outedges{ui},pi) = socialIntensity(outedges{ui},pi) + ...
        tau(ui,outedges{ui})'*g(0);
end

for i = 1:NTest
    event = testEvents{i};
    ti = event.time;
    ui = event.user;
    pi = event.product;
    if i > 1
        lastEventTime = testEvents{i-1}.time;
        socialIntensity = socialIntensity * g(ti-lastEventTime);
    end    
    
    baseIntensity = beta(:,1:end-1)*theta(ui,1:end-1)'+beta(:,end)*theta(ui,end)*(1+alpha); 
    intensity = baseIntensity + socialIntensity(ui,:)';
    ranks(i) = sum(intensity>intensity(pi))+1;
    recallAtK(ranks(i):P) = recallAtK(ranks(i):P)+1;
    if ranks(i)<RecommendationListSize
        isAtK = 1;
    else
        isAtK = 0;
    end
    ndcgAtK(ranks(i):P) = ndcgAtK(ranks(i):P) + 1/log2(1+ranks(i));
    if i>1
        ndcgOverTime(i) = ndcgOverTime(i-1)+1/log2(1+ranks(i));
        recallAtKOverTime(i) = recallAtKOverTime(i-1)+isAtK;
    else
        ndcgOverTime(i) = 1/log2(1+ranks(i));
        recallAtKOverTime(i) =isAtK;
    end
    socialIntensity(outedges{ui},pi) = socialIntensity(outedges{ui},pi) + tau(ui,outedges{ui})'*g(0);
end
ndcgAtK = ndcgAtK/NTest;
ndcgOverTime = ndcgOverTime./(1:NTest)';
recallAtKOverTime=recallAtKOverTime./(1:NTest)';
recallAtK = recallAtK/NTest;
fprintf('NDCG is %f\n',ndcgOverTime(end));
fprintf('recallAt%d is %f\n',RecommendationListSize,recallAtKOverTime(end));
end

function [time] = generatePoint(t0,baseIntensity,...
    socialIntensityInitial,g)
time = t0;
isGeneratedPoint = false;
while (isGeneratedPoint == false)
    I = baseIntensity+socialIntensityInitial*g(time-t0);
    time = time+exprnd(1/I);
    Is = baseIntensity+socialIntensityInitial*g(time-t0);
   % disp(['I= ',num2str(I),', Is= ',num2str(Is)]);
    c = rand();
    if (c*I <= Is)
        isGeneratedPoint = true;            
      %  disp('generated point');
    end
end
%disp('---------------------------------------');
end


%% computeSocialIntensityUser
function [socialIntensity] = computeSocialIntensityUser(u,t,P,inedges,eventsMatrix,tau,g)
    socialIntensity = 0.0;
    for v=inedges{u}
        for p=1:P
            for ti=eventsMatrix{v,p}
                if (ti >= t)
                    break;
                end
                socialIntensity = socialIntensity+tau(v,u)*g(t-ti);
            end
        end
    end
end