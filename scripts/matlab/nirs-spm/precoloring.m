function [SPM] = precoloring(SPM, Y)

% this function is used for estimation of GLM parameters.
% especially, 'precoloring' method is used for estimating temporal
% correlation. 
% 'out of memory' problem is sufficiently solved.
% last update : March 8th, 2010.

[nScan nBeta] = size(SPM.xX.X);

% Design space and projector matrix [pseudoinverse] for WLS
%==================================================================
switch SPM.xX.K.HParam.type
    case 'Wavelet-MDL'
        tmp_K = SPM.xX.K;
        tmp_K.HParam.type = '';
        SPM.xX.xKXs = spm_sp('Set', spm_filter_HPF_LPF_WMDL(tmp_K, SPM.xX.X)); % KX
        SPM.xX.K.X = SPM.xX.X;
        clear tmp_K;
    case 'DCT'
        SPM.xX.xKXs = spm_sp('Set', spm_filter_HPF_LPF_WMDL(SPM.xX.K, SPM.xX.X)); % KX
end

SPM.xX.xKXs.X = full(SPM.xX.xKXs.X);
SPM.xX.pKX = spm_sp('x-', SPM.xX.xKXs); % projector

load(SPM.nirs.fname);
try
    if strcmp(nirs_data.cL.type, 'none') == 0 & strcmp(nirs_data.cH.type, 'none') == 0
        KY = Y;
    elseif strcmp(nirs_data.cL.type, 'none') == 0 & strcmp(nirs_data.cH.type, 'none') == 1
        K_tmp = SPM.xX.K;
        K_tmp.LParam.type = 'none';
        KY = spm_filter_HPF_LPF_WMDL(K_tmp, Y);
    elseif strcmp(nirs_data.cL.type, 'none') == 1 & strcmp(nirs_data.cH.type, 'none') == 0
        K_tmp = SPM.xX.K;
        K_tmp.HParam.type = 'none';        
        KY = spm_filter_HPF_LPF_WMDL(K_tmp, Y);
    end    
    clear K_tmp;
catch
    KY = spm_filter_HPF_LPF_WMDL(SPM.xX.K, Y);
end
clear nirs_data;

SPM.nirs.beta = SPM.xX.pKX * KY; % beta : least square estimate
res = spm_sp('r', SPM.xX.xKXs, KY); % Residuals 

% update for calculating the channel-wise least-square residual correlation
% date: Aug 10, 2011

ResSS = (KY' * KY) - (KY' * SPM.xX.xKXs.X) * SPM.xX.pKX * KY;
SPM.nirs.ResSS = ResSS;
SPM.nirs.res = res;
% end of update

clear KY; 
clear Y; 
switch SPM.xX.K.LParam.type
    case {'hrf', 'Gaussian'}
        S = SPM.xX.K.KL;
    case 'none'
        S = speye(nScan);
end

switch SPM.xX.K.HParam.type 
    case 'DCT'
        try
            S = S - SPM.xX.K.X0 * (SPM.xX.K.X0' * S);                                    
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
            
            var1 = SPM.xX.K.X0 * (SPM.xX.K.X0' * S);
            fid = fopen('S2.txt', 'wt');
            fprintf(fid, str, full(var1));
            fclose(fid);
            clear var1;
            clear fid;
            
            S = [];
            fid1 = fopen('S.txt', 'rt');
            fid2 = fopen('S2.txt', 'rt');
            
            %h_wait = waitbar(0, 'Generating the filtering matrix... Please wait.');
            disp('Generating a filtering matrix... Please wait.');
            for kk = 1:nScan
             %   waitbar(kk/nScan, h_wait);
                S = [S; str2num(fgetl(fid1)) - str2num(fgetl(fid2))];
            end
            disp('Completed.');
            %close(h_wait);
            fclose(fid1);
            fclose(fid2);
            S = S';
            delete('S.txt');
            delete('S2.txt');
        end        
end

% if the 'out of memory' happens,
% new code for calculating the trace(RV) & trace(RVRV)
trRV = 0;
trRVRV = 0;
disp('Calculating the trace(RV) & trace(RVRV)... Please wait.');
%h_wait = waitbar(0, 'Calculating the trace(RV) & trace(RVRV)... Please wait.');
for kk = 1:nScan
%    waitbar(kk/nScan, h_wait);
    var1 = S * S(kk,:)';
    var2 = var1 - SPM.xX.xKXs.X * (SPM.xX.pKX * var1); % RV * e(kk)
    var3 = S * (S' * var2);
    var4 = var3 - SPM.xX.xKXs.X * (SPM.xX.pKX * var3);
    trRV = trRV + var2(kk);
    trRVRV = trRVRV + var4(kk);
end
%close(h_wait);
disp('Completed.');

SPM.xX.trRV = trRV; % <R'*y'*y*R>
SPM.xX.trRVRV = trRVRV; %- Satterthwaite
SPM.xX.erdf = trRV^2/trRVRV; % effective degrees of freedom
% SPM.xX.Bcov = SPM.xX.pKX*V*SPM.xX.pKX';
SPM.xX.Bcov = (SPM.xX.pKX * S);
SPM.xX.Bcov = SPM.xX.Bcov * SPM.xX.Bcov';
SPM.nirs.step = 'estimation';

try
    K = SPM.xX.K;
    K = rmfield(K, 'X');
    SPM.xX.K = K;
    clear K;
end

disp('Model parameter estimation has been completed');

