function RunFGPP(datasetName,methodName,isSocial,isSelfExciting,alpha, K,w,forceRun,maxNumberOfIterations)
if nargin<9
    maxNumberOfIterations = 1000;
end
if nargin < 8
    forceRun = 0;
end
if nargin < 7
    w = 1;
end
if nargin<6
    K = 20;
end
if nargin < 5
    alpha = 1.0;
end

prior = struct;
prior.alpha = alpha;
prior.shape = struct;
prior.shape.m = alpha;
prior.shape.v = 1.0; % This parameter is fixed and always equal to 1.0
prior.shape.beta = 1.0;
prior.shape.tau =1.0;
prior.shape.ksi = 1.0;
prior.shape.eta = 1.0;
prior.shape.mu = 1.0;

prior.rate = struct;
prior.rate.v = alpha;
prior.rate.beta = 0.1;
prior.rate.ksi = 0.1;
prior.rate.eta = 0.1;
prior.rate.mu = 0.1;

events = cell(1,1);
load(sprintf('Datasets/%sDataset_%d_%d.mat',datasetName,isSocial,isSelfExciting));

N = size(events,1);
t0 = zeros(U,1);
%% train and test events
trainSize = ceil(0.8*N);
trainEvents = events(1:trainSize);
testEvents = events(size(trainEvents)+1:end);

trainEventsMatrix = computeUserProductEventsMatrix(U,P,trainEvents);
% utriggers_times_train = utriggers_times(1:trainSize);
% utriggers_users_train = utriggers_users(1:trainSize);
% ptriggers_times_train = ptriggers_times(1:trainSize);
% ptriggers_products_train = ptriggers_products(1:trainSize);

% testUserEventsMatrix = computeUserEventsMatrix(U,testEvents);
fprintf('Reading Dataset Completed.\n');

kernel = struct;
kernel.w = w;
kernel.g = @(x,w) w*exp(-w*x);
kernel.g_factorized = @(x,w,p) p*exp(-w*x);
kernel.g_log = @(x,w) log(w)-w*x;
kernel.G = @(x,w) 1-exp(-w*x) ;

params = struct;
params.U = U;
params.P = P;
params.K = K;
params.alpha = alpha;
params.maxNumberOfIterations = maxNumberOfIterations;
params.saveInterval = 50;
params.plottingInIteration = 0;
params.datasetName = datasetName;
params.methodName = methodName;
params.isInDebugMode = 0;
%% Training The Model
modelFileName = sprintf('LearnedModels/LearnedModel_%s_%s_K_%d_w_%.1f_iter_%d.mat',methodName,datasetName,params.K,w,maxNumberOfIterations);
if forceRun==1 || exist(modelFileName,'file')~=2
    [theta, beta, tau, gamma, prior, cuv] = FGPP(trainEvents,trainEventsMatrix,...
        events{N}.time,t0,inedges,outedges,prior,params,kernel,0,0);
    save(modelFileName,'theta','beta','tau','kernel','params','prior','gamma','cuv');
    fprintf('Learning Model Completed.\n');
end
fprintf('Learning Model Completed.\n');

%% evaluation
modelFileName = sprintf('LearnedModels/LearnedModel_%s_%s_K_%d_w_%.1f_iter_%d.mat',methodName,datasetName,params.K,w,maxNumberOfIterations);
load(modelFileName);
g = @(x) w*exp(-w*x);
RecListSize = 20;
metricFileName = sprintf('Results/Metrics_%s_%s_K_%d_w_%.1f_iter_%d.mat',methodName, datasetName,params.K, kernel.w,maxNumberOfIterations);
if exist(metricFileName,'file')~=2
    [ranks, ndcgOverTime, ndcgAtK, recallAtKOverTime, recallAtK,...
         EstimatedReturningTime, RealReturningTime,diff] = ...
    FGPPEvaluator(trainEvents,testEvents, outedges,inedges,eventsMatrix,...
        theta, beta, tau, g, params, RecListSize);
    save(metricFileName,...
        'ranks', 'ndcgOverTime', 'ndcgAtK', 'recallAtKOverTime', 'recallAtK',...
        'EstimatedReturningTime', 'RealReturningTime','diff');
end
fprintf('Evaluation Completed.\n');
end