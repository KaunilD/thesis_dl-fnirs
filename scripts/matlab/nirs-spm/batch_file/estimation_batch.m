function [SPM_nirs] = estimation_batch(fname_SPM, fname_nirs)
% NIRS-SPM batch script for 'Estimation' routine, which estimates the GLM
% parameters and temporal correlation. 
% fname_SPM : the name of file which results from model specifiction
% e.g.,'...\NIRS_SPM_v3_3\Sample_data\categorical_indiv_HbO\SPM_indiv_HbO.mat';
% fname_nirs : the name of the nirs file
% e.g.,'...\NIRS_SPM_v3_3\Sample_data\converted_NIRS.mat';
% example usage
% >> fname_SPM =
% 'C:\NIRS_SPM_v3_3\Sample_data\categorical_indiv_HbO\SPM_indiv_HbO.mat';
% >> fname_nirs = 'C:\NIRS_SPM_v3_3\Sample_data\converted_NIRS.mat';
% >> [SPM_nirs] = estimation_batch(fname_SPM, fname_nirs);

try
    [pathn, name, ext, versn] = fileparts(fname_SPM);
    pathn = [pathn filesep];
catch
    index = find(fname_SPM == filesep);
    pathn = fname_SPM(1:pathn(end));
end
load(fname_nirs);
load(fname_SPM);

switch SPM_nirs.nirs.step
    case 'estimation'
        disp('The process of estimation has been already done. Please do the step of model specification, again.');
        SPM_nirs = [];
        return;
end

disp('Model parameter estimation starts...');
switch SPM_nirs.nirs.Hb
    case 'HbO'
        Y = nirs_data.oxyData;
    case 'HbR'
        Y = nirs_data.dxyData;
    case 'HbT'
        tf = isfield(nirs_data, 'tHbData');
        if tf == 0
            Y = nirs_data.oxyData + nirs_data.dxyData;
        elseif tf == 1
            Y = nirs_data.tHbData;
        end
end

% estimation of GLM parameters using either percoloring or prewhitening
if isfield(SPM_nirs.xVi, 'V') == 1 % precoloring method 
    SPM_nirs = rmfield(SPM_nirs, 'xVi');
    [SPM_nirs] = precoloring(SPM_nirs, Y);
elseif isfield(SPM_nirs.xVi, 'V') == 0
    [SPM_nirs] = prewhitening(SPM_nirs, Y, pathn);
end
disp('Completed.');
save([pathn 'SPM_indiv_' SPM_nirs.nirs.Hb '.mat'], 'SPM_nirs');

% delete precalculated files (e.g. interpolated beta, its
% covariance and t- or F-statistics)
fname_others = cellstr(spm_select('FPList', pathn, ['^interp.*\' SPM_nirs.nirs.Hb '.mat$']));
if strcmp(fname_others{1}, filesep) ~= 1
    delete(fname_others{:});
end
fname_others = cellstr(spm_select('FPList', pathn, '^interp_matrix.*\.mat$'));
if strcmp(fname_others{1}, filesep) ~= 1
    delete(fname_others{:});
end
disp('Estimation of model parameters has been completed.'); 
