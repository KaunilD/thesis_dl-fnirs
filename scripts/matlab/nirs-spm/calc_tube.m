function [threshold] = calc_tube(kappa, p_value)
% calculating a threshold for tube-formula corrected p-value 
% inputs: 
% kappa: kappa values 
% p_value: specified p-value
% output:
% threshold: calculated threshold for T-statistics 

z_value = 1:0.0001:7;
ini_ran = 10^(-10);
p_value_tube = ((kappa * gamma(3/2))/(2*(pi^(3/2))))*(1-gammainc((z_value(:).^2)/2, 3/2));
index_z = [];
n = 0;
while isempty(index_z) == 1
    ran = ini_ran * (10^n);
    n = n+1;
    index_z = find(p_value_tube > p_value - ran & p_value_tube < p_value + ran);
end
index_z = index_z(end);
threshold = z_value(index_z);
    
    
    