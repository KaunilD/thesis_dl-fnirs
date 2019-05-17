function [gavg_beta, group_beta, xX, index_group, mask, brain_view, fname_interp_cov] = nirs_spm_group(fname_ginterp_betas, min_subj)

% inputs: 
% fname_ginterp_betas: file names of individual-level interpolated betas 
% min_subj: the minimum number of overlapped individual subjects 
% At the border areas of interpolated individual maps, the maximum number
% of the individual subjects is quite different. Thus, the region of group
% analysis is restricted to the areas where the number of subjects is over
% the input variable - min_subj

% obtain the image size of specific view of rendered brain 
str_brain{1,1} = 'ventral';
str_brain{2,1} = 'dorsal';
str_brain{3,1} = 'right';
str_brain{4,1} = 'left';
str_brain{5,1} = 'frontal';
str_brain{6,1} = 'occipital';

mtx_flag = zeros(6,1);
for kk = 1:6
    mtx_flag(kk,1) = isempty(strfind(fname_ginterp_betas{1}, str_brain{kk,1}));
end
index_brain = find(mtx_flag == 0);

% size of brain matrix (s1: row, s2, column)
if index_brain == 5 || index_brain == 6
    bs = [362 362];
else
    bs = [362 434];
end
nvox_brain = bs(1) * bs(2); % # of voxels of brain matrix 

% information of brain view 
brain_view.name = str_brain{index_brain,1};
brain_view.index = index_brain;
brain_view.size = bs;

% interpolation mask (group)
nsubj = length(fname_ginterp_betas);
mask = zeros(1, nvox_brain);

for kk = 1:nsubj
    load(fname_ginterp_betas{kk}, 'index_mask');
    mask(index_mask) = mask(index_mask) + 1;
end
index_group = find(mask > min_subj-1); 
nvox = length(index_group); % # of voxels within region of interest (group)

% calculation of group-average beta
% and concatenation of individual betas for group F-statistics 
load(fname_ginterp_betas{1});
nbeta = size(interp_beta, 1);

indiv_beta = zeros(nbeta, nvox_brain);
indiv_beta(:, index_mask) = interp_beta;
group_beta{1} = indiv_beta(:,index_group);
gavg_beta = group_beta{1};

for kk = 2:nsubj
    load(fname_ginterp_betas{kk});
    indiv_beta = zeros(nbeta, nvox_brain);
    indiv_beta(:,index_mask) = interp_beta;
    group_beta{kk} = indiv_beta(:, index_group);
    gavg_beta = gavg_beta + group_beta{kk};
end
gavg_beta = gavg_beta./(ones(nbeta,1) * mask(index_group));

% identifying the filename of individual covariances
for kk = 1:nsubj
    [path name] = fileparts(fname_ginterp_betas{kk});
    str_index = strfind(name, 'beta');
    name = [name(1:str_index-1) 'cov' name(str_index+4:end)];
    fname_interp_cov{kk} = [path filesep name '.mat'];
end



