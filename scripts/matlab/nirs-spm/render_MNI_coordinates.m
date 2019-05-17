function [rend, rendered_MNI] = render_MNI_coordinates(vx_MNI, wT1_info)
dat.mat = wT1_info.mat;
dat.dim = wT1_info.dim;

Nmark = size(vx_MNI, 2);
dat.XYZ = zeros(3,Nmark);
for kk = 1:Nmark
    dat.t(1,(kk-1)+1:kk) = kk*10;
end

dat.XYZ(:,:) = vx_MNI(1:3,:);
tmp_mm_MNI = wT1_info.mat * vx_MNI;
dat.mm_MNI(:,:) = tmp_mm_MNI(1:3,:);
clear tmp_MNI;

load([spm('dir') filesep 'rend' filesep 'render_single_subj.mat']);
brt = NaN;

if (exist('rend') ~= 1), % Assume old format...
    rend = cell(size(Matrixes,1),1);
    for i=1:size(Matrixes,1),
        rend{i}=struct('M',eval(Matrixes(i,:)),...
            'ren',eval(Rens(i,:)),...
            'dep',eval(Depths(i,:)));
        rend{i}.ren = rend{i}.ren/max(max(rend{i}.ren));
    end;
end;


for i=1:length(rend),
    rend{i}.max=0;
    rend{i}.data = cell(size(dat,1),1);
    if issparse(rend{i}.ren),
        % Assume that images have been DCT compressed
        % - the SPM99 distribution was originally too big.
        d = size(rend{i}.ren);
        B1 = spm_dctmtx(d(1),d(1));
        B2 = spm_dctmtx(d(2),d(2));
        rend{i}.ren = B1*rend{i}.ren*B2';
        % the depths did not compress so well with
        % a straight DCT - therefore it was modified slightly
        rend{i}.dep = exp(B1*rend{i}.dep*B2')-1;
    end;
    msk = find(rend{i}.ren>1);rend{i}.ren(msk)=1;
    msk = find(rend{i}.ren<0);rend{i}.ren(msk)=0;
end;

mx = zeros(length(rend),1)+eps;
mn = zeros(length(rend),1);

for j=1:length(dat),
    XYZ = dat(j).XYZ;
    mm_MNI = dat(j).mm_MNI;
    t  = dat(j).t;
    dim = dat(j).dim;
    mat = dat(j).mat;
    % transform from Taliarach space to space of the rendered image
    %---------------------------------------------------------------
    for i=1:length(rend),
        M1  = rend{i}.M*dat(j).mat;
        zm  = sum(M1(1:2,1:3).^2,2).^(-1/2);
        M2  = diag([zm' 1 1]);
        M  = M2*M1;
        cor = [1 1 1 ; dim(1) 1 1 ; 1 dim(2) 1; dim(1) dim(2) 1 ;
            1 1 dim(3) ; dim(1) 1 dim(3) ; 1 dim(2) dim(3); dim(1) dim(2) dim(3)]';
        tcor= M(1:3,1:3)*cor + M(1:3,4)*ones(1,8);
        off = min(tcor(1:2,:)');
        M2  = spm_matrix(-off+1)*M2;
        M  = M2*M1;
        xyz = (M(1:3,1:3)*XYZ + M(1:3,4)*ones(1,size(XYZ,2)));
        d2  = ceil(max(xyz(1:2,:)'));
        % calculate 'depth' of values
        %-------------------------------------------------------
        dep = spm_slice_vol(rend{i}.dep,spm_matrix([0 0 1])*inv(M2),d2,1);
%         z1  = dep(round(xyz(1,:))+round(xyz(2,:)-1)*size(dep,1));

        switch i
            case 1
                msk = find(mm_MNI(3,:) < -24 | mm_MNI(3,:) == -24); % ventral view
            case 2
                msk = find(mm_MNI(3,:) > -24 | mm_MNI(3,:) == -24);  %dorsal view
            case 3
                msk = find(mm_MNI(1,:) > 0 | mm_MNI(1,:) == 0); % lateral view (right hemisphere)
            case 4
                msk = find(mm_MNI(1,:) < 0 | mm_MNI(1,:) == 0); % lateral view (left hemisphere)
            case 5
                msk = find(mm_MNI(2,:) > -44 | mm_MNI(2,:) == -44); % frontal view
            case 6
                msk = find(mm_MNI(2,:) < -44 | mm_MNI(2,:) == -44); % occipital view
        end
        if ~isempty(msk),
            xyz =  xyz(:,msk);
            if ~isfinite(brt)
                t0  = t(msk);
            end;
            
            %%% updated 2009-02-05
            msk = find(xyz(1,:) > 0  & xyz(2,:) > 0 & xyz(3,:) > 0);
            xyz = xyz(:,msk);
            t0  = t0(:,msk);
            %%% 
            
            X0  = full(sparse(round(xyz(1,:)), round(xyz(2,:)), t0, d2(1), d2(2)));
            hld = 1; if ~isfinite(brt), hld = 0; end;
            X   = spm_slice_vol(X0,spm_matrix([0 0 1])*M2,size(rend{i}.dep),hld);
            msk = find(X<0);
            X(msk) = 0;
        else,
            X = zeros(size(rend{i}.dep));
        end;
        % Brighten the blobs
        if isfinite(brt), X = X.^brt; end;
        mx(j) = max([mx(j) max(max(X))]);
        mn(j) = min([mn(j) min(min(X))]);
        rend{i}.data{j} = X;
     end;
end;

rend{1}.mxmx = max(mx);
rend{1}.mnmn = min(mn);

for i=1:length(rend)
    temp_dep = rend{i}.dep;
    temp_ren = rend{i}.ren;
    temp_data = rend{i}.data{1};
    rend{i}.dep = temp_dep(size(temp_dep,1):-1:1,:);
    rend{i}.ren = temp_ren(size(temp_ren,1):-1:1,:);
    rend{i}.data{1} = temp_data(size(temp_data,1):-1:1,:);
end

for kk = 1:length(rend)
    for i = 1:Nmark
        [rows, cols, vals] = find(rend{kk}.data{1} == i*10);
        if isempty(rows) == 1 | isempty(cols) == 1
            rendered_MNI{kk}.rchn(i,1) = -1;
            rendered_MNI{kk}.cchn(i,1) = -1;
        elseif isempty(rows) == 0 & isempty(cols) == 0
            rendered_MNI{kk}.rchn(i,1) = round(mean(rows));
            rendered_MNI{kk}.cchn(i,1) = round(mean(cols));
        end
    end  
end