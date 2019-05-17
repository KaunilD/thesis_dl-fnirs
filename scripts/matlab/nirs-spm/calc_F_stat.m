function [F_stat df] = calc_F_stat(SPM_nirs, B, chs, c)
% Calculation of interpolated F-statistic
% inputs:
% SPM_nirs: SPM_nirs structures
% B: interpolating kernel matrix
% c: contrast vector
% output
% F_stat: interpolated F-statistics
% df: degree of freedom
% df(1): trMVMV/(trMV^2)
% df(2): trRVRV/(trRV^2)

switch SPM_nirs.xX.K.HParam.type
    case 'Wavelet-MDL'
        SPM_nirs.xX.K.X = SPM_nirs.xX.X;
end

% generation of measurement vectors
load(SPM_nirs.nirs.fname); % load nirs_data file
switch SPM_nirs.nirs.Hb
    case 'HbO'
        Y = nirs_data.oxyData;
    case 'HbR'
        Y = nirs_data.dxyData;
    case 'HbT'
        try
            Y = nirs_data.tHbData;
        catch
            Y = nirs_data.oxyData + nirs_data.dxyData;
        end
end
try % filtered NIRS data
    if strcmp(nirs_data.cL.type, 'none') == 0 && strcmp(nirs_data.cH.type, 'none') == 0
        KY = Y;
    elseif strcmp(nirs_data.cL.type, 'none') == 0 && strcmp(nirs_data.cH.type, 'none') == 1
        K_tmp = SPM_nirs.xX.K;
        K_tmp.LParam.type = 'none';
        KY = spm_filter_HPF_LPF_WMDL(K_tmp, Y);
    elseif strcmp(nirs_data.cL.type, 'none') == 1 && strcmp(nirs_data.cH.type, 'none') == 0
        K_tmp = SPM_nirs.xX.K;
        K_tmp.HParam.type = 'none';
        KY = spm_filter_HPF_LPF_WMDL(K_tmp, Y);
    end
    clear K_tmp;
catch
    KY = spm_filter_HPF_LPF_WMDL(SPM_nirs.xX.K, Y);
end
clear Y;
clear nirs_data;

% load parameters from SPM_nirs structures
X = SPM_nirs.xX.xKXs.X; % design matrix
ResSS = SPM_nirs.nirs.ResSS;
nScan = size(SPM_nirs.xX.X,1);


% generation of S matrix (filtering) for calculation of trMV & trMVMV
switch SPM_nirs.xX.K.LParam.type
    case {'hrf', 'Gaussian'}
        S = SPM_nirs.xX.K.KL;
    case 'none'
        S = speye(nScan);
end

switch SPM_nirs.xX.K.HParam.type
    case 'DCT'
        try
            S = S - SPM_nirs.xX.K.X0 * (SPM_nirs.xX.K.X0' * S);
        catch % in case that 'out of memory' appears
            % writing S matrix as text file
            str = [];
            for kk = 1:nScan
                str = [str '%g '];
            end
            str = [str '\n'];
            fid = fopen('S.txt','wt');
            fprintf(fid, str, full(S));
            fclose(fid);
            clear fid;
            
            var1 = SPM_nirs.xX.K.X0 * (SPM_nirs.xX.K.X0' * S);
            fid = fopen('S2.txt', 'wt');
            fprintf(fid, str, full(var1));
            fclose(fid);
            clear var1;
            clear fid;
            
            S = [];
            fid1 = fopen('S.txt', 'rt');
            fid2 = fopen('S2.txt', 'rt');
            
            disp('Generating a filtering matrix... Please wait.');
            for kk = 1:nScan
                S = [S; str2num(fgetl(fid1)) - str2num(fgetl(fid2))];
            end
            disp('Completed.');
            fclose(fid1);
            fclose(fid2);
            S = S';
            delete('S.txt');
            delete('S2.txt');
        end
end
% end of calculation of S matrix

nCont = size(X, 2);
c0 = eye(nCont) - c * pinv(c);
X0 = X * c0;
cSigma = (KY' * KY) - (KY' * X0) * pinv(X0) * KY;

% calculation of trace(MV) & trace(MVMV)
trMV = 0;
trMVMV = 0;

disp('Calculating the trace(MV) & trace(MVMV)... Please wait.');
for kk = 1:nScan
    var1 = S * S(kk,:)';
    var2 = X * (pinv(X) * var1) - X0 * (pinv(X0) * var1);
    var3 = S * (S' * var2);
    var4 = X * (pinv(X) * var3) - X0 * (pinv(X0) * var3);
    trMV = trMV + var2(kk);
    trMVMV = trMVMV + var4(kk);
end
disp('Completed.');

% end of calculation of trace MV and trace MVMV
df(1) = trMVMV ./ (trMV.^2);
df(2) = SPM_nirs.xX.erdf;


F_stat = [];
disp('Calculating interpolated F-statistics ...');
% interpolating matrix dependent-term calculation
for kk = 1:length(B)
    if isempty(B{kk}) == 0 % if interpolating matrix exists,
        nch = length(chs{kk});
        ResSS_set = zeros(nch);
        cSigma_set = zeros(nch);
        for aa = 1:nch 
            for bb = 1:nch 
                ResSS_set(aa,bb) = ResSS(chs{kk}(aa), chs{kk}(bb));
                cSigma_set(aa,bb) = cSigma(chs{kk}(aa), chs{kk}(bb));
            end
        end
        [V_X D_X] = eig(ResSS_set);
        [V_X0 D_X0] = eig(cSigma_set);
        tmp = D_X.^(1/2) * V_X' * B{kk};
        ip_ResSS = sum(tmp.^2,1);
        tmp = D_X0.^(1/2) * V_X0' * B{kk};
        ip_ResSS_X0 = sum(tmp.^2,1);
        F_stat = [F_stat ((ip_ResSS_X0 - ip_ResSS)./trMV)./(ip_ResSS./SPM_nirs.xX.trRV)];
    end
end
disp('Completed');

