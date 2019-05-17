%This function is written by Archana-san on 8th Nov.)
%if I save the data set in csv format, the program reads x y z values in
%line wise manner, and return the cortical surface points corrsponding each
%deta set.
%To run this function, load "BrainSurfEdgeMNI.mat"
%Then select all the lines above and press F9 (or chose "conduct the selected
%command")

M=csvread('ref.csv')
for i=1:442%%%%%%%%%%%%%%%%%%%%%%%%%%number of the dataset
    P=M(i,:)
    B=BackProjectionf(xallBEM, yallBEM, zallBEM,P )
    if i==1
        CP=[B]
    else
        CP=[CP;B]
    end
end



