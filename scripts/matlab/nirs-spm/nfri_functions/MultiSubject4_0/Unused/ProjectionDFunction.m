%%%ProjectionDM%%%
%Modified version of Projection D
%This program needs the following variates
%FacetListR created by Conv program, a list of coordinates for facets and
%apices of the convex hull
%xall, yall, zall created by BrainImageRead3D program, a list of
%coordinates for brain surface

%P is a given point on the head surface or its outside space
function Out=ProjectionDFunction(xall, yall, zall, P)
%This is a function version for ProjectionD program
%Based on the balloon-inflation algorithm

%This part obtains a list of distances between P and cortical surfaces
XYZ=[xall', yall', zall'];
PP=ones(size(XYZ));
PP(:,1)=P(1);
PP(:,2)=P(2);
PP(:,3)=P(3);
PreD=XYZ-PP;
PreD2=PreD.^2;
PreD3=sum(PreD2,2);
D=PreD3.^0.5;%Now distance is found

[VD,ID]=sort(D);%ID gives indices
Top=1000;%This value is only for visual presentation
IDTop=ID(1:Top);%Indices list for top# closest points
XYZTop=XYZ(IDTop,:);%x, y, z coordinate list for top# closest points
%plot3(XYZTop(:,1),XYZTop(:,2),XYZTop(:,3),'b.');
%Obtains the colosest point Pneae on the convex hull surfaces from P
%Uses average of several closest points (NClose) to estimate the closest points
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
RodR=1;
Iless2=find(DH<=RodR);
Rod=XYZTop(Iless2,:);
%plot3(Rod(:,1),Rod(:,2),Rod(:,3),'r.');

%Okamoto option
%Find brain surface points on the vicinity of P
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
%plot3(Vic(:,1),Vic(:,2),Vic(:,3),'ch');
%plot3(CP(1),CP(2),CP(3),'.','Color',[1 0 0],'MarkerSize',30);

Out=CP;