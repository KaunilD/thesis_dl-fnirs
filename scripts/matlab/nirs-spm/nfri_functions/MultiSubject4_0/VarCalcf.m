%%%VarCalcf%%%
%Last modified by Dan on 11/02/2004
% This function calculates the variance from a given representative point
% to a list of points.
% List of points are fed as n by 3 matrix.
% A given point is fed as 1 by 3 matrix.
% This function returns variances in x, y, z axes and composite distance variance.

function Out=VarCalcf(AA, AV)

N=size(AA,1);

%First, creating a matrix for subtraction
SubMat=ones(size(AA));
for cc=1:N
    SubMat(cc,:)=AV;
end

DispEach=AA-SubMat;
DispEachSq=DispEach.^2;
XYZSS=sum(DispEachSq);
RSS=sum(XYZSS);

XYZVar=XYZSS/(N-1);
RSSVar=RSS/(N-1);
    
Out=[XYZVar,RSSVar];
