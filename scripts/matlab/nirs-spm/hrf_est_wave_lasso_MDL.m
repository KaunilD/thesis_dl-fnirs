function [Bout, theta, index] = hrf_est_wave_lasso(Y,H,flag)

n_B = size(H, 2); % # of points in one block
m_W = 9; % support size of wavelet filter;
N = size(Y,1);
decJ = fix(log2(n_B/(m_W-1)));

B = zeros(n_B, n_B);
for kk = 1:n_B
    tmp = zeros(n_B,1);
    tmp(kk,1) = 1;
    B(:,kk) = waveletcdf97(tmp, -decJ);
end

A = H * B;

Nw = n_B;
x = 1:Nw;
logs = zeros(Nw,1); 

for i = 1:Nw
    tmpy = log2(x(i));
    logs(i) = tmpy;
    while(1)
        tmpy2 = log2(tmpy);
        if tmpy2 <= 0
            break;
        else
            logs(i) = logs(i) + tmpy2;
            tmpy = tmpy2;
        end
    end
end

logsc = 2.865064;           % Constant for log*
logs = logs + log2(logsc);  % Final log*
Punv = 2.^(-logs);          % universal prior of intergers
Punv = Punv / sum(Punv(:));

[theta, numIters, activationHist, duals] = SolveLasso(A, Y, size(A,2), 'lasso', 25, 0, 0, 1, 0, 1e-10);
if isempty(theta) == 1
    [theta, numIters, activationHist, duals] = SolveLasso(A, Y, size(A,2), 'lasso', 25, 0, 0, 1, 0, 1e-10);
end

for kk = 1:size(theta,2)
    n0 = length(find(theta(:,kk)) ~= 0);
    res = Y - A * theta(:,kk);
    MDL(kk) = (N./2)*log2((1/N)*res'*res) + (3/2)*n0*log2(n_B);
    SIC(kk) = (N./2)*log2((1/N)*res'*res) + (1/2)*n0*log2(n_B);
end

count = 1;

if flag == 0
    index = find(MDL == min(MDL(count:end)));
elseif flag == 1
    index = find(SIC == min(SIC(count:end)));
end

theta = theta(:,index);
act_index = find(theta ~= 0);
theta = theta(act_index);
Bout = B(:,act_index);

     