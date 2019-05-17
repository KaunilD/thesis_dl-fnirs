function  [mni_CS] = projection_CS(mni_HS)
% This function was modified using 'nfri_mni_estimation' function (Singh et
% al., 2005). 

RefN = 17; 
PointN = size(mni_HS, 2);
OtherRefCList = cell(1,RefN);
OtherRefList = mni_HS';

for c2=1:RefN
    ProjectionListC = ones(PointN,3);
    % Read cortical surface data for reference brain used.
    DataName = [GetMatDir filesep 'CrtSrfMNISm',sprintf('%04d',c2)];
    load(DataName);
    % Now brain surface data are read as xallM, yallM, zallM

    for cc=1:PointN
        % This is a modified version with stable results
        ProjectionListC(cc,:) = ProjectionBS_f(xallM, yallM, zallM, ...
            OtherRefList(cc,1:3));
    end
    OtherRefCList{c2} = ProjectionListC;
end

CPListOverPoint=cell(1,PointN);

for c2 = 1:PointN; % Counting up other points
    CPList = ones(RefN,3);
    for c1 = 1:RefN % Counting up reference brain
        CPList(c1,:) = OtherRefCList{c1}(c2,:);
    end
    CPListOverPoint{c2} = CPList;
end

ThisScriptPath = GetMatDir;
% load MAT/BrainSurfEdgeMNI;
MatFile = [ThisScriptPath filesep 'BrainSurfEdgeMNI'];
load (MatFile);
OtherC = ones(PointN, 3);

for c3=1:PointN;
    AA              = mean(CPListOverPoint{c3},1);
    % Surface transformation
    BB              = BackProjectionf(xallBEM, yallBEM, zallBEM, AA);
    OtherC(c3,:)    = BB;  
    clear AA BB;
end
mni_CS = [OtherC';ones(1, size(OtherC, 1))];

function PATH = GetMatDir
PATH = fileparts(mfilename('fullpath'));
PATH = [PATH filesep 'nfri_functions' filesep 'mat' filesep 'nfri_mni_estimation'];

function Out=ProjectionBS_f(xall, yall, zall, P) % Later alive
%!!!This is beta version for ProjectionBS_f
%Stable projection function based on Balloon-inflation algorithm
%Last modified by I. Dan on 050706

%%%ProjectionDM%%%
%Modified version of Projection D
%This program needs the following variates
%FacetListR created by Conv program, a list of coordinates for facets and
%apices of the convex hull
%xall, yall, zall created by BrainImageRead3D program, a list of
%coordinates for brain surface

%P is a given point on the head surface or its outside space

%This is a function version for ProjectionD program
%Based on the balloon-inflation algorithm

% load CrtSrfMNISm0001.mat%%%%%%%
%
% xall=xallM;%%%%%
% yall=yallM;%%%%%
% zall=zallM;%%%%%
% P=[-62    60   -63]%%%%%
% %P=[30 40 100]%%%%%


%This part obtains a list of distances between P and cortical surfaces
XYZ=[xall', yall', zall'];

PP=ones(size(XYZ));
PP(:,1)=P(1);
PP(:,2)=P(2);
PP(:,3)=P(3);
PreD=XYZ-PP;
PreD2=PreD.^2;
PreD3=sum(PreD2,2);
D=PreD3.^0.5; % Now distance is found

[VD,ID]=sort(D);%ID gives indices
%Top=1000;%This value is only for visual presentation
Top=round(size(XYZ,1)*0.05);%Top5% are selected
%!!!This value used to be 1000
% but it sometimes do not cover the deep sulcus
% So now we use a relative number

IDTop=ID(1:Top);%Indices list for top# closest points
XYZTop=XYZ(IDTop,:); %x, y, z coordinate list for top# closest points

% Obtains the colosest point Pnear on the convex hull surfaces from P
% Uses average of several closest points (NClose) to estimate the closest points
NClose=200;
IDClose=ID(1:NClose);
XYZClose=XYZ(IDClose,:);
PNear=mean(XYZClose);
%plot3(PNear(1),PNear(2),PNear(3),'m*');
%plot3(XYZClose(:,1),XYZClose(:,2),XYZClose(:,3),'g.');

%%%Should this be Top instead of Close???

%Draw a line between P and Pnear
NXYZTop=size(XYZTop,1);
%Draw a line between Pnear and P
%First, define a vector
PVec=P-PNear;
A=PVec(1); B=PVec(2); C=PVec(3);
H=ones(size(XYZTop));
%We will get a list of foots H of normal line from brain surface points to the vector
for c=1:NXYZTop;
    xc=XYZTop(c,1); yc=XYZTop(c,2);zc=XYZTop(c,3);
    t=(A*(xc-P(1))+B*(yc-P(2))+C*(zc-P(3)))/(A^2+B^2+C^2);
    H(c,:)=[A*t+P(1) B*t+P(2) C*t+P(3)];
end

%Find diviation between points in XYZClose and H
PreDH=XYZTop-H;
PreDH2=PreDH.^2;
PreDH3=sum(PreDH2,2);
DH=PreDH3.^0.5;%This is a distance list for brain surface points to the vector
%[VDH,IDH]=min(DH)

%Rod Option
%Expand the line P-Pnear to a rod

%%%Rod is now incremented
Det=0;
RodR=0;
while Det==0
    RodR=RodR+1;
    Iless2=find(DH<=RodR);
    Rod=XYZTop(Iless2,:);
    Det=sum(sum(Rod.^2));
end

% Okamoto option
% Find brain surface points on the vicinity of P
PPB=ones(size(Rod));
PPB(:,1)=P(1);
PPB(:,2)=P(2);
PPB(:,3)=P(3);

PreVicD=Rod-PPB;
PreVicD2=PreVicD.^2;
PreVicD3=sum(PreVicD2,2);
VicD=PreVicD3.^0.5;%Distance list

[VVicD,IVicD]=sort(VicD);

NVic=3;%The number of points to be averaged
if size(Rod,1)<=NVic;
    NVic=size(Rod,1);
end;
NIVicD=IVicD(1:NVic);
Vic=Rod(NIVicD,:);%Select top NVic closest points from Pnear within the rod

CP=mean(Vic,1);
Out=CP;

function Out = BackProjectionf(xall, yall, zall, P)

%%%BackProjectionf%%%
% Last modified by Dan, 041104
% This function simply finds the closest points on the head or brain surface.
% Work only within a small region, and not suitable for large area search.
% Pick up three closest points and find the centroid.

% Input may be head or brain surface EDGE points
% P should be close enough to the surface edge points

% This part obtains a list of distances between P and cortical surfaces
XYZ = [xall', yall', zall'];
PP = ones(size(XYZ));
PP(:,1) = P(1);
PP(:,2) = P(2);
PP(:,3) = P(3);
PreD = XYZ-PP;
PreD2 = PreD.^2;
PreD3 = sum(PreD2,2);
D = PreD3.^0.5; %Now distance is found

[VD,ID]=sort(D); %ID gives indices
Top=3; % This value is only for visual presentation
IDTop=ID(1:Top); % Indices list for top# closest points
XYZTop=XYZ(IDTop,:); % x, y, z coordinate list for top# closest points
Closest=mean(XYZTop,1);
% Closest=XYZ(ID(1), :); % tsuzuki test: This might be correct.

Out = Closest;
