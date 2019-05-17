%%%SurfCenterf%%%
%Last modified by Dan on 10/06/2004
% This is a function to find the center of given points on head (or whatever) surface.
% Head surface is fed as column vectors and processes as xall yall zall.
% Given points are fed as n by 3 matrix.
% The point having least squared distance sums is defined as the center.
% This function returns center coordinates in the first three columns and
% total variances of distance in the fourth column.

function Out=SurfCenterf(xall, yall, zall, Data)
%The following parts servey the three neighboring points
DSqList=ones(size(xall,1),1);%Template for the list of distance from PB is made

for c=1:size(xall,1);
    
    %First, creating a matrix for subtraction
    SubMat=ones(size(Data));
    for cc=1:size(Data,1)
        SubMat(cc,:)=[xall(c) yall(c) zall(c)];
    end
    
    DispEach=Data-SubMat;
    DispEachSq=DispEach.^2;
    TotalDispSq=sum(sum(DispEachSq));
    DSqList(c)=TotalDispSq;%This is the list of total distance squared 
end

%Sort the distances and choose the 1st, 2nd and 3rd closest points.
[VDD, IDD]=sort(DSqList);%Distance list is sorted
Center1=[xall(IDD(1)), yall(IDD(1)), zall(IDD(1))];
Var=VDD(1)/(size(Data,1)-1);
Out=[Center1,Var];
