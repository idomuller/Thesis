function [U, d_A, v_A] = GetDiffuisionMap (Dis, ep_factor, ep, alpha)

%% now the gaussian kernel
ker = exp(-Dis/(ep_factor*ep));

% normalization
sum_alpha = sum( ker,2).^alpha;
symmetric_sum = sum_alpha*sum_alpha';
ker = ker./symmetric_sum;
      
% second normalization to make it row - stochatic         
sum_c = (sum( ker,2));        
for i=1:size(ker,1)
    ker(i,:) = ker(i,:) / sum_c(i);
end

A = ker ;
disp('doing an eigen value decomposition\n\n')

[ U, d_A, v_A ] = svd( double( A)); %Yuri