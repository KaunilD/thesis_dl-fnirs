function [SPM] = spm_nirs_design_final(nirs_fname, flag, flag_type)
% Statistical analysis of NIRS data uses a mass-univariate approach based on the general
% linear model (GLM). GLM is a statistical linear model that explains data as a linear
% combination of an explanatory variable plus an error term.
% In ¡®Specify the 1st level¡¯ routine, (1) GLM design matrix, (2) hemodynamic basis
% function, (3) filter parameter, and (4) the method of temporal correlation estimation are
% specified.
% Several timing parameters used in constructing the design matrix are fixed as follows:
% (a) Interscan interval (sec) : 1/ Sampling frequency of NIRS data (Hz)
% (b) Microtime resolution : 10
% (c) Microtime onset : 1
% Last update : April 28th, 2009.


% flag : 1 (HbO only), 2 (HbR only)

load(nirs_fname);
SPM = [];

if strcmp(flag_type, 'GLM_specification') == 1
    try
        cL = nirs_data.cL;
        cH = nirs_data.cH;
    end
    try
        SPM = cH.SPM;
        if flag == 2 % HbR
            SPM.xX.X(:,1:end-1) = SPM.xX.X(:,1:end-1).*(-1);
        end
    end
end

if isempty(SPM) == 1
    SPM.nscan = size(nirs_data.oxyData,1);
    spm_input(['Total number of NIRS time series :' num2str(SPM.nscan)],1,'d','SPM-NIRS design')
    
    SPM.xY.RT = 1/nirs_data.fs;
    % SPM.xBF.T = spm_input('Microtime resolution','+1','r','10',1);
    % SPM.xBF.T0 = spm_input('Microtime onset','+1','r','1',1);
    SPM.xBF.T = 10;
    SPM.xBF.T0 = 1;
    SPM.xBF.dt = SPM.xY.RT/SPM.xBF.T;
    
    try
        vector_onset = nirs_data.vector_onset;
        spm_input('The column of onset vector exists in the data.', 1, 'd');
        str = 'specify the onset vector?';
        flag_onset = spm_input(str, '+1', 'automatically|manually');
        if strcmp(flag_onset, 'automatically') == 1
            SPM.xBF.UNITS = 'scans';
        elseif strcmp(flag_onset, 'manually') == 1
            str           = 'specify design in';
            SPM.xBF.UNITS = spm_input(str,'+1','scans|secs');
        end
    catch
        str           = 'specify design in';
        SPM.xBF.UNITS = spm_input(str,'+1','scans|secs');
        
        %%% load the multiple conditions (saved as '.mat')
        spm_input('(Multiple) Conditions - names, onsets, durations', 1, 'd');
        str           = 'load *.mat file?';
        flag_cond = spm_input(str,'+1','yes|no');
        if strcmp(flag_cond, 'yes') == 1
            [filen, pathn] = uigetfile('*.mat','Select file containing multiple conditions');
            path_file_n = [pathn filen];
            if filen(1) == 0 | pathn(1) == 0
                return;
            end
            load(path_file_n);
            for kk = 1:size(names,2)
                SPM.Sess.U(kk).name = names(kk);
                SPM.Sess.U(kk).ons = onsets{kk};
                SPM.Sess.U(kk).dur = durations{kk};
            end
        end
    end
    rep     = 0;
    
    switch flag_type
        case 'GLM_specification'
            SPM.xBF = nirs_spm_get_bf(SPM.xBF);
        case 'model_generation'
            SPM.xBF.name = 'hrf';
            SPM.xBF = spm_get_bf(SPM.xBF);
    end
    switch flag
        case 1
            bf = SPM.xBF.bf;
        case 2
            bf = SPM.xBF.bf * (-1);
    end
    
    % try
    % 	V   = SPM.xBF.Volterra;
    % catch
    % 	str = 'model interactions (Volterra)';
    % 	V   = spm_input(str,'+1','y/n',[2 1]);
    % 	SPM.xBF.Volterra  = V;
    % end
    V = 1;
    SPM.xBF.Volterra = V; % model interactions (Volterra) : no
    
    Xx    = [];
    Xb    = [];
    Xname = {};
    Bname = {};
    
    % cur_dir = cd;
    % cd(spm_dir);
    
    for s = 1:length(SPM.nscan)
        
        if (s == 1) | ~rep
            k   = SPM.nscan(s);
            try
                if strcmp(flag_onset, 'automatically') == 1
                    tSPM = SPM;
                    tSPM.vector_onset = vector_onset;
                    U = nirs_spm_get_ons(tSPM, s, 2);
                elseif strcmp(flag_onset, 'manually') == 1
                    U = nirs_spm_get_ons(SPM, s, 1);
                end
            catch
                U = nirs_spm_get_ons(SPM, s, 1);
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
                str   = sprintf('Session %d',s);
                spm_input('Other regressors',1,'d',str)
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
end

nscan = SPM.nscan;
nsess = length(nscan);

switch flag_type
    case 'GLM_specification'
        %%% updated for wavelet-MDL detrending 2009-03-19
        str = 'Detrending?';
        try
            SPM.xX.K.HParam.type = cH.type;
            if strcmp(SPM.xX.K.HParam.type, 'DCT') == 1
                SPM.xX.K.HParam.M = cH.M;
                spm_input([str ' ' cH.type ',  ' num2str(cH.M)], 1, 'd');
            elseif strcmp(SPM.xX.K.HParam.type, 'Wavelet-MDL') == 1
                spm_input([str ' ' cH.type], 1, 'd');
            end
        catch
            clear cH;
            cH = {'Wavelet-MDL', 'DCT'};
            cH = spm_input(str, 1, 'b', cH);
            SPM.xX.K.HParam.type = cH;
            if strcmp(SPM.xX.K.HParam.type, 'DCT') == 1
                SPM.xX.K.HParam.M = spm_input('High-pass filter cut-off [seconds]','+1','r', '128',1);
            end
        end
        str = 'Low-pass filter?';
        try
            if strcmp(cL.type, 'none') == 0
                SPM.xX.K.LParam.type = cL.type;
                if strcmp(cL.type, 'Gaussian') == 1
                    SPM.xX.K.LParam.FWHM = cL.FWHM;
                    spm_input([str ' ' cL.type ',  ' num2str(cL.FWHM)], '+1', 'd');
                else
                    spm_input([str ' ' cL.type], '+1', 'd');
                end
            else
                clear CL
                cL = {'none', 'Gaussian', 'hrf'};
                cL = spm_input(str, '+1', 'b', cL);
                SPM.xX.K.LParam.type = cL;
                switch SPM.xX.K.LParam.type
                    case 'Gaussian'
                        SPM.xX.K.LParam.FWHM = spm_input('Gaussian FWHM [seconds]','+1','r','4',1);
                end
            end
        catch
            clear CL
            cL = {'none', 'Gaussian', 'hrf'};
            cL = spm_input(str, '+1', 'b', cL);
            SPM.xX.K.LParam.type = cL;
            switch SPM.xX.K.LParam.type
                case 'Gaussian'
                    SPM.xX.K.LParam.FWHM = spm_input('Gaussian FWHM [seconds]','+1','r','4',1);
            end
        end
        K = struct( 'HParam', SPM.xX.K.HParam,...
            'row', SPM.Sess.row,...
            'RT', SPM.xY.RT,...
            'LParam', SPM.xX.K.LParam);
        SPM.xX.K = spm_filter_HPF_LPF_WMDL(K);
        
        % related spm m-file : spm_fmri_spm_ui.m
        str   = 'Correct for serial correlations?';
        try
            if strcmp(nirs_data.cH.type, 'none') == 0 | strcmp(nirs_data.cL.type, 'none') == 0
                cVi = 'none';
                spm_input([str ' ' cVi], '+1', 'd');
            else
                str   = 'Correct for serial correlations?';
                cVi   = {'none','AR(1)'};
                cVi   = spm_input(str,'+1','b',cVi);
            end
        catch
            str   = 'Correct for serial correlations?';
            cVi   = {'none','AR(1)'};
            cVi   = spm_input(str,'+1','b',cVi);
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
        spm_DesRep('DesMtx',SPM.xX,[],SPM.xsDes)
        SPM.nirs.step = 'specification';
    case 'model_generation'
        SPM.nirs.step = 'model_generation';
end
SPM.nirs.fname = nirs_fname;


