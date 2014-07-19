%%%%%%% simulated income shocks for K people

    %% Simulated income profile with correlation
    K=5000;
    % replace the initial draws with the normal
    mynumbers=randn(T,K);
    value=zeros(T,K);
    value(1,:)=mynumbers(1,:);
    for t=2:T,
       value(t,:)= value(t-1,:)+mynumbers(t,:);
    end;
    DisI=zeros(T,K);
    for t=1:T,
        for k=1:K,
            [~,DisI(t,k)]=min(abs(RI-value(t,k)));
        end;
    end;

    
%%%%% assets profile simulation
State = zeros(T,K);
State(1,:)=Policy(a0+1,t,DisI(1,:));
for t=2:T,
    for k=1:K,
        State(t,k)=Policy(State(t-1,k),t,DisI(t,k));
    end;
end;

WealthProfile=WealthGrid(State);