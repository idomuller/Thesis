

function [v_A, d_A, A]= RunOneElecZone(P, WIN_LEN, ep, ep_factor, alpha)

[S1,S2] = size(P);

% in this example - starting from point num. 15, take the previous 14 days and compute the
% covariance matrix and then the inverse covariance. 
% For each day, save its inverse covariance matrix.

% in our application, each point can be the "mean/median" in the region of
% interest (one hand - distal+proxy, for example,  I am not sure it will work so good for distal only, since it is only 5 time series)

for j=(WIN_LEN+1):S1
    time_cloud = P(j-WIN_LEN:j,:);
    c = cov(time_cloud);
    inv_c(:,:,j-WIN_LEN) = pinv(c);  
end


% in this example - the data points are from 15 - end 
% (ignore the lines with the "waitbar")
% compute the pairwise distances based on the inverse covariance matrix of each point

data = P(1+WIN_LEN:end,:);
M = S1-WIN_LEN;
%h = waitbar(0, 'Please wait');
Dis = zeros(M, M);
for i = 1:M
%    waitbar(i/M, h);
    for j = 1:M
%          Dis(i,j) = [data(i,:) - data(j,:)] * inv_c(:,:,j) * [data(i,:) - data(j,:)]';
         Dis(i,j) = [data(i,:) - data(j,:)] * [inv_c(:,:,j)+inv_c(:,:,i)]*0.5 * [data(i,:) - data(j,:)]';

    end
end

%close(h);


% now the gaussian kernel
ker = exp(-Dis/(ep_factor*ep));

% normalization
sum_alpha = sum( ker,2).^alpha;
symmetric_sum = sum_alpha*sum_alpha';
ker = ker./symmetric_sum;
    
   
% second normalization to make it row - stochatic         
sum_c = (sum( ker,2));        
for i=1:size(ker,1)
    ker(i,:) = ker(i,:) / sum_c(i);
end;

A = ker ;
 disp('doing an eigen value decomposition\n\n')

[ U, d_A, v_A ] = svd( double( A)); %Yuri



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
%  Let's see if we can ignore this normalization below, I think so..

% % like the paper
% W_sml=A'*A;    
% d1=sum(W_sml,1);
% 
% A1=A./repmat(d1,M,1);
% W1=A1'*A1;
% 
% 
% 
% d2=sum(W1,1);
% A2=A1./repmat(sqrt(d2),M,1);
% W2=A2'*A2;
% 
% Ker = W2;
% 
% D = diag(sqrt(1./d2));
% 
% 
% 
% 
% 
% [V,E] = eigs(W2, 10);
% 
% E(2:end, 2:end);
% [srtdE,IE] = sort(sum(E),'descend');
% V_srt_clds = D*V(:,IE(1,2:10));
% 
% 
% % OR:
% % 
% % [ UU, DD, V ] = svd( double(W2));
% % for j=1:10
% %     V_DD(:,j) = DD(j,j)*V(:,j);
% % end;
% %     V_srt_clds = V_DD(:,2:10);
% % E=[];
