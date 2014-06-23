function [Trans,xxx]=randomwalkincome(Z,sig)

%%%%%FUNCTION This function computes the value and the Markov matrix of a random income process 
%%%%% It takes as argument the possible number of values that the process
%%%%% can take and the standard deviation of the permanent income shock


%%%%%% vector
xxx=linspace(-3,3,Z);

pe=(xxx(end)-xxx(end-1))/2;

%%%%%INDIVIDUAL MATRIX
Trans=zeros(Z,Z);

for i=1:Z,
    for j=1:Z,
        diff=abs(xxx(i)-xxx(j));
        Trans(i,j)=double(normcdf(diff+pe,0,sig))-double(normcdf(diff-pe,0,sig));
    end;
end;

for i=1:Z,
    Trans(:,i)=Trans(:,i)./(sum(Trans(:,i)));
end;


end
