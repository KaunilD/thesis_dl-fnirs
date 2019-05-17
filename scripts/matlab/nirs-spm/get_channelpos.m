function [rend rendered_MNI ch_MNI_mm] = get_channelpos(Affine, VF, VG, wT1_info, tocoMRI, tocoDGT, optdDGT, ch_config);
Q = VG(1).mat*inv(Affine)/VF.mat;
Ntoco = size(tocoMRI,2);
Noptd = size(optdDGT,2);
Nch = size(ch_config, 1);

tmpCombi = perms(1:Ntoco);
Ncombi = size(tmpCombi,1);
errVec = zeros(Ncombi,1);

y = tocoMRI;
x = tocoDGT;

for i = 1:Ncombi
    for j = 1:Ntoco
        x(:,j) = tocoDGT(:,tmpCombi(i,j));
    end

    % solve an absolute orientation prob.-------------------------
    [s R t] = abs_orientation(x,y);     % y = s*R(x) + t

    % validate
    estY = zeros(size(y));
    for iterData = 1:Ntoco
        estY(:,iterData) = s*R*x(:,iterData) + t;
    end;
    err = y - estY;
    errVal = sum(err(:).^2);
    errVec(i) = errVal;
end

[useless minIND] = min(errVec);
for j = 1:Ntoco
    x(:,j) = tocoDGT(:,tmpCombi(minIND,j));
end
tocoDGT = x;
[s R t] = abs_orientation(x,y);     % y = s*R(x) + t
estY = zeros(size(y));
for iterData = 1:Ntoco
    estY(:,iterData) = s*R*x(:,iterData) + t;
end;
err = y - estY;
errVal = sum(err(:).^2)

optdMRI = zeros(size(optdDGT));

for iterCh = 1:Noptd
    optdMRI(:,iterCh) = s*R*optdDGT(:,iterCh) + t;
end;


% surf_raw = space2brain2;
% optdBrain = zeros(size(optdMRI));
% for i = 1:Noptd
%     minDst = 1e5;
%     minWhere = -1;
%     for j = 1:length(surf_raw)
%         dst = sum(sum(abs(optdMRI(:,i)-surf_raw(1:3,j))));
%         if minDst > dst
%             minDst = dst;
%             minWhere = j;
%         end
%     end
%     optdBrain(:,i) = surf_raw(1:3,minWhere);
% end

% chBrain_tmp = zeros(3,length(ch_config));
% for i = 1:length(ch_config)
%     chBrain_tmp(:,i) = (optdBrain(1:3,ch_config(i,1))+optdBrain(1:3,ch_config(i,2)))/2;
% end

% update
ch_HS = zeros(3,length(ch_config));
for i = 1:length(ch_config)
    ch_HS(:,i) = (optdMRI(1:3,ch_config(i,1))+optdMRI(1:3,ch_config(i,2)))/2;
end

% chBrain = zeros(3,Nch);
% for i = 1:Nch
%     minDst = 1e5;
%     minWhere = -1;
%     for j = 1:length(surf_raw)
%         dst = sum(sum(abs(chBrain_tmp(:,i)-surf_raw(1:3,j))));
%         if minDst > dst
%             minDst = dst;
%             minWhere = j;
%         end
%     end
%     chBrain(:,i) = surf_raw(1:3,minWhere);
% end

%%%chBrain_tal = ch_MNI_mm
%%%chMRIN = ch_MNI_vx

%%% Raw -> Talairach
ch_HS_MNI = Q * [ch_HS;ones(1,Nch)];  %% unit : mm
ch_MNI_mm = projection_CS(ch_HS_MNI);

ch_MNI_vx = inv(wT1_info.mat)*ch_MNI_mm; %% unit : vx
[rend, rendered_MNI] = render_MNI_coordinates(ch_MNI_vx, wT1_info); % transform the MNI space to spac of rendered image



function [s R t] = abs_orientation(x,y);
% y = sR(x) + t
% find s,  R,  t

N = size(x,2);

%% centroid
cntrX = (sum(x')/N)';
cntrY = (sum(y')/N)';


%% new coordinate
xNew = x - cntrX*ones(1,N);
yNew = y - cntrY*ones(1,N);

%% make 'NN' matrix
NN = zeros(4,4);
for(iterData = 1:N)
    NN = NN + quaternion2(xNew(:,iterData))'*quaternion1(yNew(:,iterData));
end;

%% find maximun eigenvector
[eigv eigd] = eig(NN);
qtn = eigv(:,4);

%% estimate R
R = qtn2rtm(qtn);

%% find scale factor
s = 0;
for iterData = 1:N
    ry_dot_R_rx = sum(yNew(:,iterData) .* (R * xNew(:,iterData)));
    s = s + ry_dot_R_rx;
end;
s = s / sum(xNew(:).^2);

%% find translation factor
t = cntrY - s * R*cntrX;


function [R] = quaternion1(r);
if(length(r) == 3)
    R = [0 -r(1) -r(2) -r(3);...
        r(1) 0 -r(3) r(2);...
        r(2) r(3) 0 -r(1);...
        r(3) -r(2) r(1) 0;];
elseif(length(r) == 4)
    R = [r(1) -r(2) -r(3) -r(4);...
        r(2) r(1) -r(4) r(3);...
        r(3) r(4) r(1) -r(2);...
        r(4) -r(3) r(2) r(1)];
else
    disp('Error @ quaternion1');
end;


function [R] = quaternion2(r);
if(length(r) == 3)
    R = [0 -r(1) -r(2) -r(3);...
        r(1) 0 r(3) -r(2);...
        r(2) -r(3) 0 r(1);...
        r(3) r(2) -r(1) 0];
elseif(length(r) == 4)
    R = [r(1) -r(2) -r(3) -r(4);...
        r(2) r(1) r(4) -r(3);...
        r(3) -r(4) r(1) r(2);...
        r(4) r(3) -r(2) r(1)];
else
    disp('Error @ quaternion2');
end;

function [R] = qtn2rtm(q);
% quaternion to rotation matrix
R = zeros(3,3);
s = q(1);
x = q(2);
y = q(3);
z = q(4);
R(1,1) = s^2 + x^2 - y^2 - z^2;
R(1,2) = 2 * (x*y - s*z);
R(1,3) = 2 * (x*z + s*y);
R(2,1) = 2 * (y*x + s*z);
R(2,2) = s^2 - x^2 + y^2 - z^2;
R(2,3) = 2 * (y*z - s*x);
R(3,1) = 2 * (z*x - s*y);
R(3,2) = 2 * (z*y + s*x);
R(3,3) = s^2 - x^2 - y^2 + z^2;


function ind2 = space2brain2;

% File select (gray & white matter images) -> segmented @ preprocessing of fMRI Data
gw_select = spm_select(2,'image','Select original gray and white matter image',[],0);

% File read % ---------------------------------------------------
if ischar(gw_select)
    P1 = spm_vol(gw_select(1,:));
    P2 = spm_vol(gw_select(2,:));
end;

% find axis (mm)
bb = mmAxis(P1);

% mm start
planeChange = 40;
mmStr = -30;
if bb(end,end) < planeChange
    mmEnd = bb(end,end);
    mmStr2 = -1;
else
    mmEnd = planeChange;
    mmStr2 = planeChange;
    mmEnd2 = bb(end,end);
end

ind = [];

invP1 = inv(P1.mat);

mmVec = mmStr:2:mmEnd;
Dims = round(diff(bb)'+1);
cntrRect = zeros(Dims(2),Dims(1));
cntrR = floor((Dims(2)+1)/2);
cntrC = floor((Dims(1)+1)/2);
cntrRect(cntrR-5:cntrR+5,cntrC-5:cntrC+5) = 1;
sftR = floor(Dims(1)*0.05*0.5);
sftC = floor(Dims(2)*0.05*0.5);
% using XY Plane --------------------------------------------------
for iMM = 1:length(mmVec);
    mm = mmVec(iMM);
    cent = [0;0;mm;1];
    cvx = invP1*cent;


    [imgt1 imgc1 imgs1 TM] = modi_spm_slice_vol(P1, bb, cent(1:3));
    [imgt2 imgc2 imgs2] = modi_spm_slice_vol(P2, bb, cent(1:3));
    img = imgt1+imgt2+cntrRect;
    imgr = imresize(img,0.95);
    contour_org = getContour2(imgr,0.1);

    % shift
    contour_org(:,1) = contour_org(:,1) + sftR;
    contour_org(:,2) = contour_org(:,2) + sftC;


    %     figure(1); imagesc(imgt1+imgt2); colormap(gray);
    %     hold on; plot(contour_org(:,2),contour_org(:,1),'r'); hold off;
    %     pause(0.5);
    contour = zeros(size(contour_org));
    contour(:,1) = contour_org(:,2);     % x,y change;
    contour(:,2) = contour_org(:,1);



    contour(:,1) = (Dims(1)-contour(:,1)+1);
    contour_tmp = zeros(size(contour,1),4);
    contour_tmp(:,1:2) = contour;
    contour_tmp(:,3) = 1;
    contour_tmp(:,4) = 1;

    contour_coord = TM*contour_tmp';
    ind = [ind contour_coord(1:3,:)];


end

% close all;

% using YZ plane %----------------------------------------

[minX minXIND] = min(contour_org(:,2));
maxX = max(contour_org(:,2));
strP_ind = [contour_org(minXIND,2) contour_org(minXIND,1) 0 1]';
endP_ind = [contour_org(minXIND,2)+(maxX-minX) contour_org(minXIND,1) 0 1]';
strP_vx = strP_ind;
endP_vx = endP_ind;
strP_vx(1) = Dims(1)-strP_ind(1)+1;
endP_vx(1) = Dims(1)-endP_ind(1)+1;
strP_vx = TM*strP_vx;
endP_vx = TM*endP_vx;
strP_mm = P1.mat*strP_vx;
endP_mm = P1.mat*endP_vx;

if strP_mm(1) > strP_mm(2)
    tmp = endP_mm;
    endP_mm = strP_mm;
    strP_mm = tmp;
    tmp = endP_vx;
    endP_vx = strP_vx;
    strP_vx = tmp;
end
strP_mm(3) = strP_mm(3)-5;
mmVec = strP_mm(1):2:endP_mm(1);
cent = strP_mm;
cent2 = strP_mm;
cent3 = strP_mm;
zvx = strP_vx(3);

for iMM = 1:length(mmVec)
    cent(1) = mmVec(iMM);
    cent2(1) = mmVec(iMM)-1;
    cent3(1) = mmVec(iMM)+1;
    cvx = invP1*cent;

    [imgt1 imgc1 imgs1 TM SM] = modi_spm_slice_vol(P1, bb, cent(1:3));
    [imgt2 imgc2 imgs2] = modi_spm_slice_vol(P2, bb, cent(1:3));
    [imgt1 imgc1 imgs3] = modi_spm_slice_vol(P1, bb, cent2(1:3));
    [imgt2 imgc2 imgs4] = modi_spm_slice_vol(P2, bb, cent2(1:3));
    [imgt1 imgc1 imgs5] = modi_spm_slice_vol(P1, bb, cent3(1:3));
    [imgt2 imgc2 imgs6] = modi_spm_slice_vol(P2, bb, cent3(1:3));

    indyz = inv(SM)*cvx;
    indyz(1) = 220 - indyz(1) + 1;

    imgTmp = imgs1+imgs2+imgs3+imgs4+imgs5+imgs6;
    imgTmp = imgTmp / max(imgTmp(:));
    imgHead = imresize(imgTmp,1.0);

    contour_org = getContour2(imgHead,0.1);

    % shift..
    % contour_org(:,1) = contour_org(:,1)-5;

    bigIND = find(contour_org(:,1) > indyz(2));
    %bigIND = find(contour_org(:,1) > (Dims(3)/2));
    %     figure(4); imagesc(imgs1+imgs2); colormap(gray);
    %     hold on; plot(contour_org(bigIND,2),contour_org(bigIND,1),'r'); hold off;
    %     pause(0.5);
    if length(bigIND)
        contour = zeros(length(bigIND),2);
        contour(:,1) = contour_org(bigIND,2);
        contour(:,2) = contour_org(bigIND,1);

        contour(:,1) = (Dims(1)-contour(:,1)+1);
        contour_tmp = zeros(size(contour,1),4);
        contour_tmp(:,1:2) = contour;
        contour_tmp(:,3) = 1;
        contour_tmp(:,4) = 1;

        contour_coord = SM*contour_tmp';
        ind = [ind contour_coord(1:3,:)];
    end
end

ind2 = [ind; ones(1,length(ind))];
ind2 = P1.mat*ind2;


function bb = mmAxis(P);

% calculate 'bb' ------------------------------------------------
% bb : mm axis
mn = [Inf Inf Inf];
mx = -mn;
bb = [[1 1 1];P.dim(1:3)];
c = [	bb(1,1) bb(1,2) bb(1,3) 1
    bb(1,1) bb(1,2) bb(2,3) 1
    bb(1,1) bb(2,2) bb(1,3) 1
    bb(1,1) bb(2,2) bb(2,3) 1
    bb(2,1) bb(1,2) bb(1,3) 1
    bb(2,1) bb(1,2) bb(2,3) 1
    bb(2,1) bb(2,2) bb(1,3) 1
    bb(2,1) bb(2,2) bb(2,3) 1]';
tc = (P.mat)*c;
tc = tc(1:3,:)';
mx = max([tc ; mx]);
mn = min([tc ; mn]);

bb = [mn ; mx];

function [imgt imgc imgs TM SM] = modi_spm_slice_vol(P, bb, cent);
% P    : from spm_vol
% cent : input position (mm)

Dims = round(diff(bb)'+1);

M = P.mat;
TM0 = [	1 0 0 -bb(1,1)+1
    0 1 0 -bb(1,2)+1
    0 0 1 -cent(3)
    0 0 0 1];
TM = inv(TM0*M);
TD = Dims([1 2]);

CM0 = [	1 0 0 -bb(1,1)+1
    0 0 1 -bb(1,3)+1
    0 1 0 -cent(2)
    0 0 0 1];
CM = inv(CM0*M);
CD = Dims([1 3]);

SM0 = [	0  1 0 -bb(1,2)+1
    0  0 1 -bb(1,3)+1
    1  0 0 -cent(1)
    0  0 0 1];
SM0 = [	0 -1 0 +bb(2,2)+1
    0  0 1 -bb(1,3)+1
    1  0 0 -cent(1)
    0  0 0 1];
SM = inv(SM0*M);
SD = Dims([2 3]);
imgt  = spm_slice_vol(P,TM,TD,1);
imgc  = spm_slice_vol(P,CM,CD,1);
imgs  = spm_slice_vol(P,SM,SD,1);

% I don't know..
imgt = imrotate(imgt,270);
imgc = imrotate(imgc,270);
imgs = imrotate(imgs,270);

function [contour] = getContour2(img, thresh);

if nargin < 2
    BW = im2bw(I,graythresh(img));
    % find the start point
    wColvec = max(BW);
    sColvec = find(wColvec>0);
    sCol= sColvec(1);
    wRowvec = BW(:,sCol);
    sRowvec = find(wRowvec>0);
    sRow = sRowvec(1);

    % Trace
    connectivity = 8;
    contour = bwtraceboundary(BW,[sRow,sCol],'N',connectivity);
    return;
elseif nargin < 3
    BW = zeros(size(img));
    BW(find(img>=thresh)) = 1;
    BW = conv2(BW(1:end-2,1:end-2),ones(3));
    BW(find(BW>0)) = 1;

    strP = zeros(8,2);

    for iterR = 1:size(img,1)
        tmpIND = find(BW(iterR,:)>0);
        if length(tmpIND)
            strP(1,1) = iterR;
            strP(1,2) = tmpIND(1);
            strP(2,1) = iterR;
            strP(2,2) = tmpIND(end);
            break;
        end
    end
    for iterR = size(img,1):-1:1
        tmpIND = find(BW(iterR,:)>0);
        if length(tmpIND)
            strP(3,1) = iterR;
            strP(3,2) = tmpIND(1);
            strP(4,1) = iterR;
            strP(4,2) = tmpIND(end);
            break;
        end
    end
    for iterC = 1:size(img,2)
        tmpIND = find(BW(:,iterC)>0);
        if length(tmpIND)
            strP(5,1) = tmpIND(1);
            strP(5,2) = iterC;
            strP(6,1) = tmpIND(end);
            strP(6,2) = iterC;
            break;
        end
    end
    for iterC = size(img,2):-1:1
        tmpIND = find(BW(:,iterC)>0);
        if length(tmpIND)
            strP(7,1) = tmpIND(1);
            strP(7,2) = iterC;
            strP(8,1) = tmpIND(end);
            strP(8,2) = iterC;
            break;
        end
    end


    % Trace
    connectivity = 8;
    lengs = zeros(8,1);

    contour = bwtraceboundary(BW,strP(1,:),'N',connectivity);
    leng = length(contour);
    for i = 2:8
        tmp_contour = bwtraceboundary(BW,strP(i,:),'N',connectivity);
        if leng < length(tmp_contour)
            contour = tmp_contour;
            leng = length(tmp_contour);
        end
    end

    return
else
    disp('INVALID INPUT');
    return
end
%}
%}
%}
