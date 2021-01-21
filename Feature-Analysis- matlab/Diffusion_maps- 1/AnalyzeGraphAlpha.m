% This function builds a kernel according to the distances in 'dist' and
% performs a spectral decomposition of the kernel
% accepts an optional number of input arguments. only "dist" is required,
% the rest has default values
%function [ U, d_A, v_A, sigma ] = AnalyzeGraph(  dist, num_of_eigs, epsilon_type,factor,Mass_Matrix)

function [ U, d_A, v_A, sigma ] = AnalyzeGraphAlpha(  dist,alpha, epsilon_type, factor, num_of_eigs)

n=99

    if nargin < 2  % nargin returns the number of input arguments passed in a call to the currently executing function.
        alpha = 0.5;
       disp('Setting alpha = 0.5')
    end
    if nargin < 3
        epsilon_type = 'max';
    end
     if nargin < 4
        factor = 2;
    end
     if nargin < 5
        num_of_eigs = 40;
    end
  
    
    disp('Building the gaussian kernel');

    [ ker, sigma] = BuildKernel( 'gaussian', dist, epsilon_type,factor);
    fprintf('Normalizing the GAUSSIAN kernel\n');
    
    

    % [ U, d_A, v_A ] = svd( double(W2)); %Yuri
   
    

    
    sum_alpha = sum( ker,2).^alpha;
    symmetric_sum = sum_alpha*sum_alpha';
    ker = ker./symmetric_sum;
    
   
         
      sum_c = (sum( ker,2));        
        for i=1:size(ker,1)
            ker(i,:) = ker(i,:) / sum_c(i);
        end;

      

        
    A = ker ;
    
    %twoA = 2.*A;
    %Asq = A*A;
    
    %A4 = twoA-Asq;
    
    cond(A)
    
    pack
    disp('doing an eigen value decomposition\n\n')
   % [ U, d_A, v_A ] = lansvd( double( A), num_of_eigs, 'L');
    [ U, d_A, v_A ] = svd( double( A)); %Yuri

    v_A = v_A(:,1:num_of_eigs);
     
   % v_A = v_A ./ repmat( v_A(:,1), 1, size(v_A,2));
return;
