%%%%%%%%%%%% CODE TO SOLVE A SIMPLE LIFE CYCLE SAVING PROBLEM 
%%%%%%%%%%% ALESSANDRA VOENA, JUNE 2014

clear all;
clc;

%%%%% life cycle problem

%%%% call script that sets up the parameters of the problem
parameters;

%%%%%% Value and policy functions
V           = zeros(N,T,I,B);         % matrix for the value function
Policy      = zeros(N,T,I,B);         % matrix for the value function
Sell        = zeros(N,T,I,B);         % matrix for the value function
EV          = zeros(N,T,I,B);         % matrix for the value function

%%%%TERMINAL PERIOD
cons2 = repmat(WealthGrid,[1,I])+repmat(Income_unc(:,T)',[N,1]);
V(:,T,:,2)=utility(cons2,Works(T));
Policy(:,T,:,2)=a0;
cons1 = repmat(WealthGrid,[1,I])+repmat(Income_unc(:,T)',[N,1])+AssetValue(T);
V(:,T,:,1)=utility(cons1,Works(T));
Policy(:,T,:,1)=a0;
Sell(:,T,:,1)=1;
clear cons2 cons1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  OTHER PERIODS  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for t = T-1:-1:1,
    %%%%%%%%%%%% if sold 
    for i=1:I,
        ev=zeros(N,1);
        for ii=1:I,
            ev=ev+TransP(ii,i)*squeeze(V(:,t+1,ii,2));
        end;
        ev(isnan(ev))=-Inf;
        EV2(:,t+1,i)=ev;
    end;
    vvv2=permute(repmat(EV2(:,t+1,:),[1,N,1]),[2 1 3]);
    %%%%%%%%%%%% if not sold 
    for i=1:I,
        ev=zeros(N,1);
        for ii=1:I,
            ev=ev+TransP(ii,i)*squeeze(V(:,t+1,ii,1));
        end;
        ev(isnan(ev))=-Inf;
        EV1(:,t+1,i)=ev;
    end;
    vvv1=permute(repmat(EV1(:,t+1,:),[1,N,1]),[2 1 3]);
    [t]
    %%%%% if have sold
    cons2 = permute(repmat(WealthGrid,[1,N,I]),[1 2 3]) + permute(repmat(Income_unc(:,t),[1,N,N]),[2 3 1]) - permute(repmat((WealthGrid)/(1+r),[1,N,I]),[2 1 3]);
    objective = utility(cons2,Works(t))+beta*vvv2;
    [V(:,t,:,2),Policy(:,t,:,2)] = max(objective,[],2);
    %%%%% if have not yet sold
    %%%% sell
    cons1 = permute(repmat(WealthGrid,[1,N,I]),[1 2 3]) + permute(repmat(Income_unc(:,t),[1,N,N]),[2 3 1]) - permute(repmat((WealthGrid)/(1+r),[1,N,I]),[2 1 3]) + AssetValue(t);
    objective = utility(cons1,Works(t))+beta*vvv2;
    [v1,p1] = max(objective,[],2);
    %%%% do not sell
    objective = utility(cons2,Works(t))+beta*vvv1;
    [v2,p2] = max(objective,[],2);
    V(:,t,:,1)=max(v1,v2);
    Sell(:,t,:,1)=(v1>v2);
    Policy(:,t,:,1)=(v1>v2).*p1+(v1<=v2).*p2;
    clear cons* objective*;    
end;

Policy(isinf(V))=NaN;

simulation;

bbb=mean(Liquidate,2);
%save prova.mat aaa;
load prova.mat;
plot([1:T],aaa,[1:T],bbb,'r-')

% aaa=mean(WealthProfile,2);
% plot([1:T],aaa)
% plot([1:T],WealthProfile(:,1:10))

