function [kappa] = calc_kappa(B, Bx, By, cov_beta_r, contrast)
% calculation of kappa value for tube formula corrected p-value 
% input: 
% B,Bx,By: interpolation matrix, and its gradient in x- & y-direction 
% cov_beta_r: (covariance of beta.^(1/2))
% contrast: contrast vector
mtx_eye = eye(size(contrast, 1));
P = cov_beta_r * kron(B, mtx_eye * contrast);
Px = cov_beta_r * kron(Bx, mtx_eye * contrast);
Py = cov_beta_r * kron(By, mtx_eye * contrast);

one_vec = ones(size(P,1),1);

% P' * P
term1 = one_vec * sum(P.^2);
% P * P' * Px
term2 = P.*(one_vec * sum(P.*Px));
term3 = P.*(one_vec * sum(P.*Py));

u_derx = Px./(term1.^(0.5)) - term2./(term1.^(1.5));
u_dery = Py./(term1.^(0.5)) - term3./(term1.^(1.5));

u_derxx = sum(u_derx.^2);
u_deryy = sum(u_dery.^2);
u_derxy = sum(u_derx.*u_dery);

kappa = 0;
nvox = size(B, 2);
for kk = 1:nvox
    vox_kappa = sqrt(abs(det([u_derxx(kk) u_derxy(kk); u_derxy(kk) u_deryy(kk)])));
    kappa = kappa + vox_kappa;
end







