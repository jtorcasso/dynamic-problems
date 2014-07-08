global gamma psi

%%%%SET UP THE PARAMETERS OF THE MODEL

N = 600;                                    % # of grid points of assets
I=10;                                       % # of grid points of permanent income
T = 45;                                     % # of periods in totakl
R = 15;                                     % # of periods in retirement

%%%%%%%%%%%% FIXED PARAMETERS %%%%%%%%

%%%%%%income
Income=15000*ones(1,T-R);
Income1=5000*ones(1,R);
Income=[Income,Income1];

%%%%%5 kabor supply
Works=[ones(1,T-R),zeros(1,R)];

%%%%%% RANDOM INCOME COMPONENT
sigma=0.5;
[TransP,RI]=randomwalkincome(I,sigma);
RandInc=exp(RI);

%%%%%%INCOME VARIABLES
for i=1:I,
    for t=1:T,
        Income_unc(i,t)=max(0,Income(t)*RandInc(i));
    end;
end;


%Other parameters
beta = single(0.98);                    % discount factor
r = single(0.03);                       % return on Assets
gamma = single(1.5);                    % RRA
psi = 0.003;                              % utility cost of working

%%%%%%%%% Wealth Grid 
WealthGrid = single(linspace(-200000,2000000,N)');    % N equal spaced grid points between a and b'
%%%% find on what point on this grid the 0 is
[~,a0]=min(abs((WealthGrid)));