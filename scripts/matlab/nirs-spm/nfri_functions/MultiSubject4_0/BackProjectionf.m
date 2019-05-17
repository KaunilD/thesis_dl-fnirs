%%%BackProjectionf%%%
%Last modified by Dan, 041104
%This function simply finds the closest points on the head or brain surface.
%Work only within a small region, and not suitable for large area search.
%Pick up three closest points and find the centroid.


%Input may be head or brain surface EDGE points
%P should be close enough to the surface edge points
function Out=BackProjectionf(xall, yall, zall, P)

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
Top=3;%This value is only for visual presentation
IDTop=ID(1:Top);%Indices list for top# closest points
XYZTop=XYZ(IDTop,:);%x, y, z coordinate list for top# closest points
Closest=mean(XYZTop,1);

Out=Closest;