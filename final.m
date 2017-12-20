%% Final 
% 
close all

labels = csvread('train_labels.csv',2);

FNC = csvread('train_FNC.csv',2);

SBM = csvread('train_SBM.csv',2);

labels = labels(:,2);

FNC = FNC(:,2:end);

[train, ~, test] = dividerand(size(FNC,1),.85,0,.15);

%% PCA
x = FNC;
pos = find(labels == 1);
neg = find(labels == 0);
[U,S,V] = svd(FNC);
figure('rend','painters','pos',[10 10 900 600])
hold on
scatter3(U(pos,1),U(pos,2),U(pos,3),'r','filled');
scatter3(U(neg,1),U(neg,2),U(neg,3),'k','filled');
title('PCA of all points');
legend('diagnosed schizophrenia','healthy patient');
hold off
%
[~,S,V] = svd(FNC(pos,:),'econ');
U = x * V ./ repmat(diag(S)',size(x,1),1);
figure('rend','painters','pos',[10 10 900 600])
hold on
scatter3(U(pos,1),U(pos,2),U(pos,3),'r','filled');
scatter3(U(neg,1),U(neg,2),U(neg,3),'k','filled');
title('All points mapped to PCA orthonormal vectors of positive data');
legend('diagnosed schizophrenia','healthy patient');
hold off

[~,S,V] = svd(FNC(neg,:),'econ');
U = x * V ./ repmat(diag(S)',size(x,1),1);
figure('rend','painters','pos',[10 10 900 600])
hold on
scatter3(U(pos,1),U(pos,2),U(pos,3),'r','filled');
scatter3(U(neg,1),U(neg,2),U(neg,3),'k','filled');
title('All points mapped to PCA orthonormal vectors of negitive data');
legend('diagnosed schizophrenia','healthy patient');
hold off

%% LDA

% setup training data
[x,~,~] = svd(FNC,'econ');
x = x(:,1:50);

pos = find(labels(train) == 1);
neg = find(labels(train) == 0);

% Setup LDA variables
N1 = length(pos);
N2 = length(neg);
N  = N1 + N2;

X1 = x(pos,:);
X2 = x(neg,:);

mu1 = mean(X1);
mu2 = mean(X2);

% Run LDA
S1 = (1/N1)*(X1 - ones(N1,1)*mu1)'*(X1 - ones(N1,1)*mu1); 
S2 = (1/N2)*(X2 - ones(N2,1)*mu2)'*(X2 - ones(N2,1)*mu2); 

Sw = (N1*S1 + N2*S2)/N;

w = pinv(Sw) * (mu1-mu2)'; % Output weights

% plot prediction quality
pred = (x * w)';

figure('rend','painters','pos',[10 10 900 300])
hold on
scatter(pred(pos),ones(1,length(pred(pos))),'r','filled');
scatter(pred(neg),ones(1,length(pred(neg))),'k','filled');
title('Best linear class division');
legend('diagnosed schizophrenia','healthy patient');
hold off

% Remove basis found from first analysis
temp_norm = (w)/norm(w);
for i = 1:size(x,1)
    x(i,:) = x(i,:) - (temp_norm * temp_norm' * x(i,:)')';
end

% Reset LDA with newly calculated variables
X1 = x(pos,:);
X2 = x(neg,:);

mu1 = mean(X1);
mu2 = mean(X2);

% run LDA again
S1 = (1/N1)*(X1 - ones(N1,1)*mu1)'*(X1 - ones(N1,1)*mu1); 
S2 = (1/N2)*(X2 - ones(N2,1)*mu2)'*(X2 - ones(N2,1)*mu2); 

Sw = (N1*S1 + N2*S2)/N;

w = pinv(Sw) * (mu1-mu2)';

pred_2 = (x * w)';

figure('rend','painters','pos',[10 10 900 600])
hold on
scatter(pred(pos),pred_2(pos),'r','filled');
scatter(pred(neg),pred_2(neg),'k','filled');
title('Best two linear class divisions');
legend('diagnosed schizophrenia','healthy patient');
hold off


%% LDA with validation

% Create data split
[x,~,~] = svd(FNC,'econ');
x = x(:,1:70);
%x = SBM;

% Setup LDA
pos = find(labels == 1);
neg = find(labels == 0);
train_pos = find(labels(train) == 1);
train_neg = find(labels(train) == 0);
test_pos  = find(labels(test)  == 1);
test_neg  = find(labels(test)  == 0);

N1 = length(train_pos);
N2 = length(train_neg);
N  = N1 + N2;

X1 = x(train_pos,:);
X2 = x(train_neg,:);

mu1 = mean(X1);
mu2 = mean(X2);

% Run LDA
S1 = (1/N1)*(X1 - ones(N1,1)*mu1)'*(X1 - ones(N1,1)*mu1); 
S2 = (1/N2)*(X2 - ones(N2,1)*mu2)'*(X2 - ones(N2,1)*mu2); 

Sw = (N1*S1 + N2*S2)/N;

w = pinv(Sw) * (mu1-mu2)';

% Plot LDA results
pred = (x * w)';

figure('rend','painters','pos',[10 10 900 300])
hold on
scatter(pred(test_pos),zeros(1,length(pred(test_pos))),'r','filled');
scatter(pred(test_neg),zeros(1,length(pred(test_neg))),'k','filled');
title('Best linear class division on test set');
legend('diagnosed schizophrenia','healthy patient');
hold off

% Remove basis
temp_norm = (w)/norm(w);
for i = [train_pos;train_neg]
    x(i,:) = x(i,:) - (temp_norm * temp_norm' * x(i,:)')';
end

% update LDA variables
X1 = x(train_pos,:);
X2 = x(train_neg,:);

mu  = mean(x);
mu1 = mean(X1);
mu2 = mean(X2);

% Run LDA
S1 = (1/N1)*(X1 - ones(N1,1)*mu1)'*(X1 - ones(N1,1)*mu1); 
S2 = (1/N2)*(X2 - ones(N2,1)*mu2)'*(X2 - ones(N2,1)*mu2); 

Sb = N1*N2/N^2 * (mu1-mu2)' * (mu1-mu2);
Sw = (N1*S1 + N2*S2)/N;

w = pinv(Sw) * (mu1-mu2)';

% Plot new values

pred_2 = (x * w)';

figure('rend','painters','pos',[10 10 900 600])
hold on
scatter(pred(test_pos),pred_2(test_pos),'r','filled');
scatter(pred(test_neg),pred_2(test_neg),'k','filled');
title('Best two linear class divisions on test set');
legend('diagnosed schizophrenia','healthy patient');
hold off


%% Inverse and pseudo inverse
y_train = labels(train);
y_test  = labels(test);

%% FNC - Psudo Inverse
x = FNC(train,:);
x_test  = FNC(test,:);
[U,S,V] = svd(x,'econ');
S = S .^ (-1);
S(S == inf) = 0;
w = V * S * U' * y_train;
% Is under constrained so should give perfect results.

%% Lasso
[train, ~, test] = dividerand(size(FNC,1),.85,0,.15);
[lasso_values, lasso_stats] = lasso(FNC(train,:),labels(train),'lambda',[.0602]);

ref = @(pred) sign(pred)/2 + .5;
error = @(pred, labels) abs(double(ref(pred)) - labels);

constants = -1 *ones(1,size(lasso_values,2));

for j = 1:size(lasso_values,2)
    for i = -1:10
        while error(FNC(train,:) * lasso_values(:,j) + constants(1,j), labels(train)) ...
                >= error(FNC(train,:) * lasso_values(:,j) + constants(1,j) + 10^(-i), labels(train))
            constants(1,j) = constants(1,j) + 10^(-i); 
        end
    end
end



guesses = FNC(train,:) * lasso_values + repmat(constants,size(FNC(train,:),1),1);
guesses(guesses > 0) = 1;
guesses(guesses < 0) = 0;
er = abs(guesses - repmat(labels(train),1,size(guesses,2)));
er = sum(er);
[v,i] = min(er);

lasso_ = lasso_values(:,i);

guesses = FNC(test,:) * lasso_values(:,i) + repmat(constants(i),size(FNC(test,:),1),1);

guesses(guesses > 0) = 1;
guesses(guesses < 0) = 0;
er = abs(guesses - repmat(labels(test),1,size(guesses,2)));
er = sum(er);

lam = lasso_stats.Lambda(min(i))
prob = er
lasso_stats.DF(i)

%% Gradient descent

[train, ~, test] = dividerand(size(FNC,1),.85,0,.15);
x_test = FNC(train,:);

output = makeNetwork(FNC(train,:), labels(train), [10,10], 100000, .01, 1);
figure('rend','painters','pos',[10 10 900 300])
hold on;
scatter(output(FNC(test(test_pos),:)),zeros(1,length(pred(test_pos))),'r','filled');
scatter(output(FNC(test(test_neg),:)),zeros(1,length(pred(test_neg))),'k','filled');
title('10 - 10 neural network predictions');
legend('diagnosed schizophrenia','healthy patient');
hold off


%%
% setup training data
x = FNC(:,find(lasso_ ~= 0));


pos = find(labels == 1);
neg = find(labels == 0);

% Setup LDA variables
N1 = length(pos);
N2 = length(neg);
N  = N1 + N2;

X1 = x(pos,:);
X2 = x(neg,:);

mu1 = mean(X1);
mu2 = mean(X2);

% Run LDA
S1 = (1/N1)*(X1 - ones(N1,1)*mu1)'*(X1 - ones(N1,1)*mu1); 
S2 = (1/N2)*(X2 - ones(N2,1)*mu2)'*(X2 - ones(N2,1)*mu2); 

Sw = (N1*S1 + N2*S2)/N;

w = pinv(Sw) * (mu1-mu2)'; % Output weights

% plot prediction quality
pred = (x * w)';

figure('rend','painters','pos',[10 10 900 300])
hold on
scatter(pred(pos),ones(1,length(pred(pos))),'r','filled');
scatter(pred(neg),ones(1,length(pred(neg))),'k','filled');
title('Best linear class division');
legend('diagnosed schizophrenia','healthy patient');
hold off

% Remove basis found from first analysis
temp_norm = (w)/norm(w);
for i = 1:size(x,1)
    x(i,:) = x(i,:) - (temp_norm * temp_norm' * x(i,:)')';
end

% Reset LDA with newly calculated variables
X1 = x(pos,:);
X2 = x(neg,:);

mu1 = mean(X1);
mu2 = mean(X2);

% run LDA again
S1 = (1/N1)*(X1 - ones(N1,1)*mu1)'*(X1 - ones(N1,1)*mu1); 
S2 = (1/N2)*(X2 - ones(N2,1)*mu2)'*(X2 - ones(N2,1)*mu2); 

Sw = (N1*S1 + N2*S2)/N;

w = pinv(Sw) * (mu1-mu2)';

pred_2 = (x * w)';

figure('rend','painters','pos',[10 10 900 600])
hold on
scatter(pred(pos),pred_2(pos),'r','filled');
scatter(pred(neg),pred_2(neg),'k','filled');
title('Best two linear class divisions');
legend('diagnosed schizophrenia','healthy patient');
hold off


%% LDA with validation

% Create data split
[x,~,~] = svd(FNC,'econ');
x = x(:,1:70);
%x = SBM;

% Setup LDA
pos = find(labels == 1);
neg = find(labels == 0);
train_pos = find(labels(train) == 1);
train_neg = find(labels(train) == 0);
test_pos  = find(labels(test)  == 1);
test_neg  = find(labels(test)  == 0);

N1 = length(train_pos);
N2 = length(train_neg);
N  = N1 + N2;

X1 = x(train_pos,:);
X2 = x(train_neg,:);

mu1 = mean(X1);
mu2 = mean(X2);

% Run LDA
S1 = (1/N1)*(X1 - ones(N1,1)*mu1)'*(X1 - ones(N1,1)*mu1); 
S2 = (1/N2)*(X2 - ones(N2,1)*mu2)'*(X2 - ones(N2,1)*mu2); 

Sw = (N1*S1 + N2*S2)/N;

w = pinv(Sw) * (mu1-mu2)';

% Plot LDA results
pred = (x * w)';

figure('rend','painters','pos',[10 10 900 300])
hold on
scatter(pred(test_pos),zeros(1,length(pred(test_pos))),'r','filled');
scatter(pred(test_neg),zeros(1,length(pred(test_neg))),'k','filled');
title('Best linear class division on test set');
legend('diagnosed schizophrenia','healthy patient');
hold off

% Remove basis
temp_norm = (w)/norm(w);
for i = [train_pos;train_neg]
    x(i,:) = x(i,:) - (temp_norm * temp_norm' * x(i,:)')';
end

% update LDA variables
X1 = x(train_pos,:);
X2 = x(train_neg,:);

mu  = mean(x);
mu1 = mean(X1);
mu2 = mean(X2);

% Run LDA
S1 = (1/N1)*(X1 - ones(N1,1)*mu1)'*(X1 - ones(N1,1)*mu1); 
S2 = (1/N2)*(X2 - ones(N2,1)*mu2)'*(X2 - ones(N2,1)*mu2); 

Sb = N1*N2/N^2 * (mu1-mu2)' * (mu1-mu2);
Sw = (N1*S1 + N2*S2)/N;

w = pinv(Sw) * (mu1-mu2)';

% Plot new values

pred_2 = (x * w)';

figure('rend','painters','pos',[10 10 900 600])
hold on
scatter(pred(test_pos),pred_2(test_pos),'r','filled');
scatter(pred(test_neg),pred_2(test_neg),'k','filled');
title('Best two linear class divisions on test set');
legend('diagnosed schizophrenia','healthy patient');
hold off
