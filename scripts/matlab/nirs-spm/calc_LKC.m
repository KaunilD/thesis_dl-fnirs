function [L2] = calc_LKC(B_volume, mask, res, level)
% calculation of Lipschitz-Killing curvatures (L2)
% input: 
% B_volume: interpolation matrix 
% mask: mask matrix 
% res: channel-residuals 

switch level 
    case 'individual'
        rshift_mask = circshift(mask, [0 -1]);
        dshift_mask = circshift(mask, [-1 0]);
        common_mask = rshift_mask.*dshift_mask.*mask;
        [data_row, data_col] = find(common_mask == 1);
        nvox = length(data_row);
        
        % calculating Lipschitz-Killing curvatures
        % Q = R/norm(R), R: interpolated residuals
        % S = (Q(r+delta_x)-Q(r), Q(r+delta_y)-Q(r))
        % L2 = sum(det(S'*S).^1/2)
        L2 = 0;
        for aa = 1:nvox
            b = squeeze(B_volume(data_row(aa), data_col(aa), :));
            b_right = squeeze(B_volume(data_row(aa), data_col(aa)+1,:));
            b_down = squeeze(B_volume(data_row(aa)+1, data_col(aa), :));
            R = res * b;
            R_right = res * b_right;
            R_down = res * b_down;
            Q = R./norm(R);
            S = [(R_right./norm(R_right) - Q) (R_down./norm(R_down) - Q)];
            L2 = L2 + sqrt(abs(det(S'*S)));
        end
    case 'group'
        % rename of input variables 
        index_group = B_volume; % index of group data
        bs = mask; % size of brain matrix
        nsubj = size(res, 1); % # of subjects
        
        gmask = zeros(bs);
        gmask(index_group) = 1;
        rshift_mask = circshift(gmask, [0 -1]);
        dshift_mask = circshift(gmask, [-1 0]);
        common_mask = rshift_mask .* dshift_mask .* gmask;
        [data_row, data_col] = find(common_mask == 1);
        nvox = length(data_row);
        
        gres = zeros(nsubj, bs(1), bs(2));
        gres(:, index_group) = res;
        gres = gres .* reshape(ones(nsubj, 1) * gmask(:)', [nsubj, bs(1), bs(2)]);
        
        L2 = 0;
        for aa = 1:nvox
            R = gres(:, data_row(aa), data_col(aa));
            R_right = gres(:, data_row(aa), data_col(aa) + 1);
            R_down = gres(:, data_row(aa)+1, data_col(aa));
            Q = R./norm(R);
            S = [(R_right./norm(R_right) - Q) (R_down./norm(R_down) - Q)];
            L2 = L2 + sqrt(abs(det(S'*S)));
        end
end

