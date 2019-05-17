function [threshold] = calc_EC(LKC, p_value, stat, df)
% calculating a threshold for LKC-based expected Euler characteristics 
% inputs: 
% LKC: Lipschitz-Killing curvatures 
% p_value: specified p-value
% stat: T or F field 
% df: degrees of freedom 
% output: 
% threshold: calculated threshold according to T or F statistics 
L0 = LKC(1);
L1 = LKC(2);
L2 = LKC(3);
switch stat
    case 'T'
        t = 0:0.001:7;
        for i=1:length(t)
            p0 = 1-spm_Tcdf(t(i),df(2));
            p1 = power(1+t(i)*t(i)/df(2), (1-df(2))/2)/(2*pi);
            p2 = gamma((df(2)+1)/2)*t(i)*power(1+t(i)*t(i)/df(2), (1-df(2))/2) / power(2*pi,3/2) / sqrt(df(2)/2) / gamma(df(2)/2);
            p(i) = p0*L0 + p1*L1 + p2*L2;
        end
        index_th=[];
        dp = 10^(-8);
        while isempty(index_th) == 1;
            index_th=find(p>p_value - dp & p<p_value + dp);
            dp = dp*10;
        end
        threshold = t(index_th(end));
    case 'F'
        k       = df(1);
        v       = df(2);
        a       = 1/(2*pi);
        b       = gammaln(v/2) + gammaln(k/2);
        t = 0:0.01:70;
        for i=1:length(t)
            p0 = 1 - spm_Fcdf(t(i),[k,v]);
            p1 = a^(1/2)*exp(gammaln((v+k-1)/2)-b)*2^(1/2)...
                *(k*t(i)/v).^(1/2*(k-1)).*(1+k*t(i)/v).^(-1/2*(v+k-2));
            p2 = a*exp(gammaln((v+k-2)/2)-b)*(k*t(i)/v).^(1/2*(k-2))...
                .*(1+k*t(i)/v).^(-1/2*(v+k-2)).*((v-1)*k*t(i)/v-(k-1));
            p(i) = p0*L0 + p1*L1 + p2*L2;
        end
        index_th=[];
        dp = 10^(-8);
        while isempty(index_th) == 1;
            index_th=find(p>p_value - dp & p<p_value + dp);
            dp = dp*10;
        end
        threshold = t(index_th(end));
end
    