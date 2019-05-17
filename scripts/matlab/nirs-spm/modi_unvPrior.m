function P = modi_unvPrior(Nw,L)
% --------------------------------------------------
% Prior distribution of
%  the wavelet coefficients of unknown global trend
% Using 'universal prior for integers'
%
% Revised in 2008.11.11. Kwang Eun Jang
% --------------------------------------------------
   x = 1:Nw;
   logs = zeros(Nw,1);  % Container for log* (without c)

   % Calculate log* for x
   for i = 1:Nw
     tmpy = log2(x(i));
     logs(i) = tmpy;
     while(1)
       tmpy2 = log2(tmpy);
       if tmpy2 <= 0
	 break;
       else
	 logs(i) = logs(i) + tmpy2;
	 tmpy = tmpy2;
       end
     end
   end

   logsc = 2.865064;           % Constant for log*
   logs = logs + log2(logsc);  % Final log*

   Punv = 2.^(-logs);          % universal prior of intergers

   % Normalize..
   Punv = Punv / sum(Punv(:));
   
   
   % Adjusting... scale by scale
   P = zeros(Nw,1);
   where = 0;
   meanP = mean(Punv(where+1:where+L(1)));
   P(where+1:where+L(1)) = meanP;
   for i = 2:length(L)
     where = sum(L(1:i-1));
     meanP = mean(Punv(where+1:where+L(i)));
     P(where+1:where+L(i)) = meanP;
   end
