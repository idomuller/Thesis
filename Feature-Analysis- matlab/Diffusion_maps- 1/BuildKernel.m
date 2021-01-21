%% This function builds the kernel from the Euclidean distance matrix
% dist is SQUARED i.e. ^2
%function [ker, sigmas] = BuildKernel( kernelName, dist, sigma_type, factor, nn, Mass_Matrix)
function [ker, sigmas] = BuildKernel( kernelName, dist, sigma_type, factor, nn)

    if ~exist('factor')
        factor = 2;
    end;
    
    if ~exist('sigma_type')
        sigma_type = 'max'; %% local function
    else
        sigma_type = lower( sigma_type);
    end
    [ bands, xSize] = size( dist);
    % Applying a kernel on the distance matrix tp create the diffusion operator
    disp('Building the kernel');
    if strcmp( upper( kernelName), 'GAUSSIAN')
        % find the radius of the ball in which each points looks for neighbors
        disp('GAUSSIAN');
        if (exist('nn') & (nn > 0))
               sigmas = FindSigmas( dist, nn); %% local function (here below)
        else
            if issparse(dist) % a sparse matrix is a matrix in which most of the elements are zero
                mins = spmin(dist, 2);
            else
                mins = min(dist + eye(xSize) * max(dist(:)));
            end
            sigma_mean = mean(mins);%avoid taking the digonal zeros as min
            sigma_max   = max(mins); %avoid taking the digonal zeros as min
            sigma_med   = median(mins) * 2;
            if issparse(dist) % a sparse matrix is a matrix in which most of the elements are zero
                [ i, j, v] = find( dist);%returns vectors of row indices,column indices,and values of the nonzero entries of 'dist' 
                sigma_med2 = median(v(:)) / 2;
            else
                sigma_med2 = median(dist(:)) / 2;
            end
            if strcmp( sigma_type, 'max')
                sigma = sigma_max;
            elseif strcmp( sigma_type, 'med')
                sigma = sigma_med;
            elseif strcmp( sigma_type, 'med2')
                sigma = sigma_med2;
            elseif strcmp( sigma_type, 'mean')
                sigma = sigma_mean;
            elseif strcmp( sigma_type, 'maxx')
                sigma = max( [sigma_mean sigma_med sigma_med2 sigma_max]) ;
            end
            fprintf(2, 'mean=%f   med=%f   med2=%f   max=%f   sigma=%f %s\n', ...
                            full( sigma_mean), full( sigma_med), full( sigma_med2), full( sigma_max), full( sigma), full( sigma_type));
%        sigma = sigma_med
%        sigma = min(mins)
%        sigma = max(max(dist));
            if 0
            figure; s = size( mins, 2); idx=1:s; plot(idx, mins, idx, sigma_mean*ones(1,s), idx, sigma_med*ones(1,s), ...
                                                                  idx, sigma_med2*ones(1,s), idx, sigma_max*ones(1,s))
            legend( 'min vals','sigma mean', 'sigma med', 'sigma med2', 'sigma max');
            title( sprintf( 'Minimum Distances for Each Row in The Kernel - chosen %s', upper(sigma_type)));
            end
            sigmas = sigma * factor;
           
          
            
        end
       % ker = gaussianKernel( dist, sigmas,Mass_Matrix);
       
        ker = gaussianKernel( dist, sigmas);
        
        
        
    elseif strcmp( upper( kernelName), 'BESSEL')
        disp('BESSEL');
%        ker = besselKernel( dist, bands, bands); 
        ker = besselKernel( sqrt(dist),2,1) + eye(bands); 
    elseif strcmp( upper( kernelName), 'POTENTIAL')
        disp('POTENTIAL');
        ker =potentialKernel(dist, data_dim); 
        sigmas = 0;
    else
        disp('taking the Euclidean distances as the kernel');
        ker = sqrt(dist);
%        ker = A_normalized_vec' * A_normalized_vec;
    end

%%% Find the sigmas for each row so sigma is equal to the distance between
%%% the rows point and the nn-th furthest neighbor
    function sigmas = FindSigmas( dist, nn)
        [rows, cols] = size(dist);
        [S,IX] = sort( dist, 1, 'ascend');
        sigmas_x = dist( IX( nn+1, :) + [0:cols:(cols-1)*rows] ) ;
        [S,IX] = sort( dist, 2, 'ascend');
        sigmas_y = dist( IX( :, nn+1) + [0:rows:(rows-1)*cols]' ) ;
        
        [ sxr, sxc] = size(sigmas_x);
        [ syr, syc] = size(sigmas_y);
        sigmas = repmat( sigmas_x, rows/sxr, cols/sxc) + repmat( sigmas_y, rows/syr, cols/syc);
