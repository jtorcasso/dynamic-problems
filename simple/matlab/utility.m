function y=utility(x,part)

	global gamma psi
    
    gamma1=1-gamma;
    y=(x.^gamma1)./gamma1-psi*part;
    y(x<=0.00000001)=-Inf;
    
end
