function [ch_MNI_mm] = optd2ch(optdMRI, ch_config, surf_raw)
if nargin == 2
    flag = 1;
elseif nargin == 3
    flag = 2;
end
Noptd = size(optdMRI, 2);
Nch = size(ch_config, 1);

if nargin == 2
    optdBrain = optdMRI;
elseif nargin == 3
    optdBrain = zeros(size(optdMRI));
    for i = 1:Noptd
        minDst = 1e5;
        minWhere = -1;
        for j = 1:length(surf_raw)
            dst = sum(sum(abs(optdMRI(:,i)-surf_raw(1:3,j))));
            if minDst > dst
                minDst = dst;
                minWhere = j;
            end
        end
        optdBrain(:,i) = surf_raw(1:3,minWhere);
    end
end

chBrain_tmp = zeros(3,length(ch_config));
for i = 1:length(ch_config)
    chBrain_tmp(:,i) = (optdBrain(1:3,ch_config(i,1))+optdBrain(1:3,ch_config(i,2)))/2;
end

if nargin == 2
    chBrain = chBrain_tmp;
elseif nargin == 3
    chBrain = zeros(3,Nch);
    for i = 1:Nch
        minDst = 1e5;
        minWhere = -1;
        for j = 1:length(surf_raw)
            dst = sum(sum(abs(chBrain_tmp(:,i)-surf_raw(1:3,j))));
            if minDst > dst
                minDst = dst;
                minWhere = j;
            end
        end
        chBrain(:,i) = surf_raw(1:3,minWhere);
    end
end

%%%chBrain_tal = ch_MNI_mm
%%%chMRIN = ch_MNI_vx

% Q = [1.0000 -0.0004 0.0008 0.0642;0.0004 1.0000 0.0031 0.6581;-0.0008 -0.0031 1.0000 -0.9514;0 0 0 1.0000];
%%% Raw -> Talairach
% ch_MNI_mm = Q * [chBrain;ones(1,Nch)];  %% unit : mm
ch_MNI_mm = [chBrain; ones(1,Nch)];

