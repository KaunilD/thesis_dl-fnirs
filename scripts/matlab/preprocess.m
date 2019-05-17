function [argout] = preprocess(nirs_data)
    % HIGH WORKLOAD, LOW WORKLOAD
    % INIT
    index_start = 1;
    index_end = 1;
    ntime = size(nirs_data.oxyData, 1);
    % DCT
    M = 128;
    % LPF
    RT = 1./nirs_data.fs;
    row = 1:size(nirs_data.oxyData,1);
    k = length(row);
    h = spm_hrf(RT);
    h = [h; zeros(size(h))];
    g = abs(fft(h));
    h = real(ifft(g));
    h = fftshift(h)';
    n = length(h);
    d = [1:n] - n/2 -1;
    KL = spdiags(ones(k,1)*h, d, k,k);
    KL = spdiags(1./sum(KL')',0,k,k)*KL;

    n  = fix(2*(k*RT)/M + 1);
    X0 = spm_dctmtx(k,n);
    X0 = X0(:,2:end);

    % Baseline Correction
    oxyData = nirs_data.oxyData - ones(ntime,1)*mean(nirs_data.oxyData(index_start:index_end,:),1);
    dxyData = nirs_data.dxyData - ones(ntime,1)*mean(nirs_data.dxyData(index_start:index_end,:),1);
%     ftotal_Hb = ftotal_Hb - ones(ntime,1) * mean(ftotal_Hb(index_start:index_end,:),1);


    oxyData = KL * nirs_data.oxyData;
    dxyData = KL * nirs_data.dxyData;
%     ftotal_Hb = KL * total_Hb;
    biasM_oxy = X0 * (X0' * oxyData);
    biasM_dxy = X0 * (X0' * dxyData);
%     biasM_tHb = X0 * (X0' * ftotal_Hb);

    oxyData = oxyData - biasM_oxy;
    dxyData = dxyData - biasM_dxy;
%     ftotal_Hb = ftotal_Hb - biasM_tHb;

    % Copy rest of the fields over to argout
    for fn = fieldnames(nirs_data)'
        argout.(fn{1}) = nirs_data.(fn{1});
    end

    argout.oxyData = oxyData;
    argout.dxyData = dxyData;
end
