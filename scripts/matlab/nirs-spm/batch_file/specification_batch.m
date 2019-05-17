function [SPM_nirs] = specification_batch(fname_nirs, hb, HPF, LPF, method_cor, dir_save, flag_window, hrf_type, units,  names, onsets, durations)

% NIRS-SPM batch script for 'specify 1st level' routine, which specifies
% the general linear model (GLM) such as the design matrix, temporal
% filtering, and temporal correlation estimation.
%
% Input variables
% 1. fname_nirs : the name of the nirs file.
% e.g., '...\NIRS_SPM_v3_2\Sample_data\converted_NIRS.mat';
% 2. hb : specific hemoglobin, e.g., 'hbo' or 'hbr', or 'hbt'
% 3. HPF : detrending method (wavelet MDL or DCT based method), e.g., 
% 1) 'wavelet' : Wavelet-MDL based 
% 2) 'DCT, 128' : DCT-based detrending with cut-off 128 (sec).
% 4. LPF : low pass filtering (hrf or Gaussian smoothing), e.g.,
% 1) 'hrf' : hemodynamic response function smoothing or
% 2) 'gaussian, 4' : Gaussian smoothing with FWHM 4 (sec)
% 5. method_cor : specific method for estimating the temporal correlation
% (precoloring or prewhitening) 
% e.g., 0 : precoloring or 1 : prewhitening
% 6. dir_save : the directory to save the variable 'SPM_nirs'
% e.g., '...\NIRS_SPM_v3_2\Sample_data\categorical_indiv_HbO\';
% 7. flag_window : 0 : do not show the design matrix, 1 : show the design
% matrix
% 8. hrf_type : the basis function to model the hemodynamic response
% e.g., 0 - 'hrf', 1 - hrf (time der.), 2 - hrf( time & dispersion
% der.) 
% 9. units : the unit for design, e.g., 0 - scan, 1 - secs
% 10. names : name for condition, e.g. 'right finger tpping'
% Note that the input variable for multiple conditions should be cell
% array, 
% e.g., name{1} = 'finger tapping', name{2} = 'n-back test';
% 11. onsets : a vector of onset times, e.g. [42 93 144 195]
% 12. durations : the task durations, e.g. [21 21 21 21]
%
% example usage
% 1. 
% >> fname_nirs = 'C:\NIRS_SPM_v3_2\Sample_data\converted_NIRS.mat';
% >> hb = 'hbo';
% >> HPF = 'wavelet';
% >> LPF = 'hrf';
% >> method_cor = 0;
% >> dir_save = 'C:\NIRS_SPM_v3_2\Sample_data\categorical_indiv_HbO\';
% >> flag_window = 1;
% >> hrf_type = 2;
% >> units = 1;
% >> names{1} = 'right finger tapping';
% >> onsets{1} = 42:51:501;
% >> durations{1} = 21 * ones(10,1);
% >> [SPM_nirs] = specification_batch(fname_nirs, hb, HPF, LPF, method_cor, dir_save, flag_window, hrf_type, units,  names, onsets, durations);

% 2. In the NIRS data from Hitachi ETG-4000 and ISS Imagent system, there
% is marker column that contains the vector of onsets and durations. In
% that case, this function automatically read the vector of onsets and
% durations from the data. 
% >> fname_nirs = 'C:\NIRS_SPM_v3_2\Sample_data\converted_NIRS.mat';
% >> hb = 'hbo';
% >> HPF = 'wavelet';
% >> LPF = 'hrf';
% >> method_cor = 0;
% >> dir_save = 'C:\NIRS_SPM_v3_2\Sample_data\categorical_indiv_HbO\';
% >> flag_window = 1;
% >> hrf_type = 2;
% if you want to specify the name of condition,
% >> names{1} = 'right finger tapping';
% >> [SPM_nirs] = specification_batch(fname_nirs, hb, HPF, LPF, method_cor, dir_save, flag_window, hrf_type, names);
% if not,
% >> [SPM_nirs] = specification_batch(fname_nirs, hb, HPF, LPF, method_cor, dir_save, flag_window, hrf_type);
% 3. Simultaneous entry of multiple condition names, onsets, and durations
% using *.mat file is allowed. This option can be used to load all the
% required information (e.g. condition names, onset, and durations) in
% one-go. You will first need to create *.mat file containing the relevant
% information. This *.mat file must include the following cell arrays (each
% 1 x n) : name, onsets, and durations. Please refer to the sample file;
% e.g.,¡¦\Sample_data\NIRS_data_file\sample_multiple_condition.mat.
% >> fname_nirs = 'C:\NIRS_SPM_v3_2\Sample_data\converted_NIRS.mat';
% >> hb = 'hbo';
% >> HPF = 'wavelet';
% >> LPF = 'hrf';
% >> method_cor = 0;
% >> dir_save = 'C:\NIRS_SPM_v3_2\Sample_data\categorical_indiv_HbO\';
% >> flag_window = 1;
% >> hrf_type = 2;
% >> units = 1;
% >> names =
% 'C:\NIRS_SPM_v3_2\\Sample_data\NIRS_data_file\sample_multiple_condition.m
% at';
% >> [SPM_nirs] = specification_batch(fname_nirs, hb, HPF, LPF, method_cor,
% dir_save, flag_window, hrf_type, units,  names);


load(fname_nirs);
SPM.nscan = size(nirs_data.oxyData,1);
SPM.xY.RT = 1/nirs_data.fs;
SPM.xBF.T = 10;
SPM.xBF.T0 = 1;
SPM.xBF.dt = SPM.xY.RT/SPM.xBF.T;

if nargin == 8 % vector_onset
    vector_onset = nirs_data.vector_onset;
    SPM.xBF.UNITS = 'scans';
elseif nargin == 9 % vector onset and name
    vector_onset = nirs_data.vector_onset;
    SPM.xBF.UNITS = 'scans';
    for kk = 1:size(names,2)
        SPM.Sess.U(kk).name = names(kk);
    end
elseif nargin == 10 % multiple condition
    if units == 0
        SPM.xBF.UNITS = 'scans';
    elseif units == 1
        SPM.xBF.UNITS = 'secs';
    end    
    load(names);
    for kk = 1:size(names, 2)
        SPM.Sess.U(kk).name = names(kk);
        SPM.Sess.U(kk).ons = onsets{kk};
        SPM.Sess.U(kk).dur = durations{kk};
    end
elseif nargin == 12
    if units == 0
        SPM.xBF.UNITS = 'scans';
    elseif units == 1
        SPM.xBF.UNITS = 'secs';
    end
    for kk = 1:size(names, 2)
        SPM.Sess.U(kk).name = names(kk);
        SPM.Sess.U(kk).ons = onsets{kk};
        SPM.Sess.U(kk).dur = durations{kk};
    end
end
rep     = 0;
switch hrf_type
    case 0
        SPM.xBF.name = 'hrf';
    case 1
        SPM.xBF.name = 'hrf (with time derivative)';
    case 2
        SPM.xBF.name = 'hrf (with time and dispersion derivatives)';
end

SPM.xBF = nirs_spm_get_bf(SPM.xBF);

switch hb
    case 'hbo'
        bf = SPM.xBF.bf;
    case 'hbr'
        bf = SPM.xBF.bf * (-1);
    case 'hbt'
        bf = SPM.xBF.bf;
end

V = 1;
SPM.xBF.Volterra = V; % model interactions (Volterra) : no

Xx    = [];
Xb    = [];
Xname = {};
Bname = {};

for s = 1:length(SPM.nscan)
    if (s == 1) | ~rep
        k   = SPM.nscan(s);
        if nargin == 8 | nargin == 9
            tSPM = SPM;
            tSPM.vector_onset = vector_onset;
            U = nirs_spm_get_ons_batch(tSPM, s, 2);
        elseif nargin == 10 | nargin == 12
            U = nirs_spm_get_ons_batch(SPM, s, 1);
        end
        [X,Xn,Fc] = spm_Volterra(U,bf,V);

        try
            X = X([0:(k - 1)]*SPM.xBF.T + SPM.xBF.T0 + 32,:);
        end

        for i = 1:length(Fc)
            X(:,Fc(i).i) = spm_orth(X(:,Fc(i).i));
        end

        try
            C     = SPM.Sess(s).C.C;
            Cname = SPM.Sess(s).C.name;
        catch
%             str   = sprintf('Session %d',s);
%             spm_input('Other regressors',1,'d',str)
            C     = [];
            %             c     = spm_input('user specified','+1','w1',0);
            c = 0;
            while size(C,2) < c
                str = sprintf('regressor %i',size(C,2) + 1);
                C  = [C spm_input(str,2,'e',[],[k Inf])];
            end

            Cname = {};
            for i = 1:size(C,2)
                str      = sprintf('regressor %i',i);
                Cname{i} = spm_input('name of','+0','s',str);
            end
        end

        X      = [X spm_detrend(C)];
        Xn     = {Xn{:}   Cname{:}};

        B      = ones(k,1);
        Bn{1}  = sprintf('constant');

    end

    SPM.Sess(s).U      = U;
    SPM.Sess(s).C.C    = C;
    SPM.Sess(s).C.name = Cname;
    SPM.Sess(s).row    = size(Xx,1) + [1:k];
    SPM.Sess(s).col    = size(Xx,2) + [1:size(X,2)];
    SPM.Sess(s).Fc     = Fc;

    % Append names
    %---------------------------------------------------------------
    for i = 1:length(Xn)
        Xname{end + 1} = [sprintf('Sn(%i) ',s) Xn{i}];
    end
    for i = 1:length(Bn)
        Bname{end + 1} = [sprintf('Sn(%i) ',s) Bn{i}];
    end

    % append into Xx and Xb
    %===============================================================
    Xx    = blkdiag(Xx,X);
    Xb    = blkdiag(Xb,B);

end %- for s

% finished
%-----------------------------------------------------------------------
SPM.xX.X      = [Xx Xb];
SPM.xX.iH     = [];
SPM.xX.iC     = [1:size(Xx,2)];
SPM.xX.iB     = [1:size(Xb,2)] + size(Xx,2);
SPM.xX.iG     = [];
SPM.xX.name   = {Xname{:} Bname{:}};
% end

nscan = SPM.nscan;
nsess = length(nscan);

%%% updated for wavelet-MDL detrending 2009-03-19
str = 'Detrending?';

if isempty(strfind(HPF, 'wavelet')) == 0 % wavelet-MDL
    SPM.xX.K.HParam.type = 'Wavelet-MDL';
elseif isempty(strfind(HPF, 'DCT')) == 0 % DCT
    index_cutoff = find(HPF == ',');
    if isempty(index_cutoff) == 1
        cutoff = 128;
    else
        cutoff = str2num(HPF(index_cutoff+1:end));
    end
    SPM.xX.K.HParam.type = 'DCT';
    SPM.xX.K.HParam.M = cutoff;
end

if isempty(strfind(LPF, 'hrf')) == 0 % hrf smoothing
    SPM.xX.K.LParam.type = 'hrf';
elseif isempty(strfind(LPF, 'gaussian')) == 0 % Gaussian smoothing
    index_FWHM = find(LPF == ',');
    if isempty(index_FWHM) == 1
        FWHM = 4;
    else 
        FWHM = str2num(LPF(index_FWHM+1:end));
    end
    SPM.xX.K.LParam.FWHM = FWHM;
    SPM.xX.K.LParam.type = 'Gaussian';
else
    SPM.xX.K.LParam.type = 'none';
end

K = struct( 'HParam', SPM.xX.K.HParam,...
    'row', SPM.Sess.row,...
    'RT', SPM.xY.RT,...
    'LParam', SPM.xX.K.LParam);
SPM.xX.K = spm_filter_HPF_LPF_WMDL(K);

% related spm m-file : spm_fmri_spm_ui.m
if method_cor == 0
    cVi = 'none';
elseif method_cor == 1
    cVi = 'AR(1)';
end

if ~ischar(cVi)	% AR coeficient[s] specified
    SPM.xVi.Vi = spm_Ce(nscan,cVi(1:3));
    cVi        = ['AR( ' sprintf('%0.1f ',cVi) ')'];

else
    switch lower(cVi)
        case 'none'		%  xVi.V is i.i.d
            %---------------------------------------------------------------
            SPM.xVi.V  = speye(sum(nscan));
            cVi        = 'i.i.d';
        otherwise		% otherwise assume AR(0.2) in xVi.Vi
            %---------------------------------------------------------------
            SPM.xVi.Vi = spm_Ce(nscan,0.2);
            cVi        = 'AR(0.2)';
    end
end
SPM.xVi.form = cVi;
SPM.xsDes = struct('Basis_functions', SPM.xBF.name, 'Sampling_period_sec', num2str(SPM.xY.RT), 'Total_number_of_samples', num2str(SPM.nscan));
if flag_window == 1
    spm_DesRep('DesMtx',SPM.xX,[],SPM.xsDes)
end

SPM.nirs.step = 'specification';
SPM.nirs.fname = fname_nirs;

SPM_nirs = SPM;
switch hb
    case 'hbo'
        SPM_nirs.nirs.Hb = 'HbO';
        SPM_nirs.nirs.level = 'individual';
        save([dir_save filesep 'SPM_indiv_HbO.mat'], 'SPM_nirs');
    case 'hbr'
        SPM_nirs.nirs.Hb = 'HbR';
        SPM_nirs.nirs.level = 'individual';
        save([dir_save filesep 'SPM_indiv_HbR.mat'], 'SPM_nirs');
    case 'hbt'
        SPM_nirs.nirs.Hb = 'HbT';
        SPM_nirs.nirs.level = 'individual';
        save([dir_save filesep 'SPM_indiv_HbT.mat'], 'SPM_nirs');        
end



