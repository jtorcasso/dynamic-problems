%%%%%%%%%%%% CODE TO SOLVE A SIMPLE LIFE CYCLE SAVING PROBLEM 
%%%%%%%%%%% ALESSANDRA VOENA, JUNE 2014

clear all;
clc;

%%%%% life cycle problem

%%%% call script that sets up the parameters of the problem
parameters

%%%%%% Value and policy functions
V           = zeros(N,T,I);         % matrix for the value function
Policy      = zeros(N,T,I);         % matrix for the value function
EV          = zeros(N,T,I);         % matrix for the value function

%%%%TERMINAL PERIOD
cons = repmat(WealthGrid,[1,I])+repmat(Income_unc(:,T)',[N,1]);
V(:,T,:)=utility(cons,Works(T));
Policy(:,T,:)=a0;
clear cons;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  OTHER PERIODS  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for t = T-1:-1:1,
    for i=1:I,
        ev=zeros(N,1);
        for ii=1:I,
            ev=ev+TransP(ii,i)*squeeze(V(:,t+1,ii));
        end;
        ev(isnan(ev))=-Inf;
        EV(:,t+1,i)=ev;
    end;
    vvv=permute(repmat(EV(:,t+1,:),[1,N,1]),[2 1 3]);
    [t]
    cons = permute(repmat(WealthGrid,[1,N,I]),[1 2 3]) + permute(repmat(Income_unc(:,t),[1,N,N]),[2 3 1]) - permute(repmat((WealthGrid)/(1+r),[1,N,I]),[2 1 3]);
    objective = utility(cons,Works(t))+beta*vvv;
    [V(:,t,:),Policy(:,t,:)] = max(objective,[],2);
    clear cons objective;    
end;

Policy(isinf(V))=NaN;

