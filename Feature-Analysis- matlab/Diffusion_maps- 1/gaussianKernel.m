%function z = gaussianKernel( dist, sigmas , Mass_Matrix)
function z = gaussianKernel( dist, sigmas )

if issparse( dist)
    [ rows, cols] = size( dist);
    [ i, j, v] = find( -dist ./ sigmas);
    i = [i;(1:rows)'];
    j = [j;(1:rows)'];
    v = [v;zeros(rows,1)];
    dist = sparse( i, j, v);
    z = sparse( i, j, exp(v), rows, cols);
else
  
    
   z = exp( -dist.^2 ./ sigmas );  
 %   z = exp( -dist ./ 0.5 );  
 %   z = z*Mass_Matrix;
end

